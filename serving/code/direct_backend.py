"""Isolated synchronous dense vLLM tier integration, not hardware HBF.

Assumptions: TP=1, runner-v1, eager execution, synchronous scheduling, dense
16-token pages, immutable full prefill pages only. Both tiers occupy real GPU
memory and native attention reads HBF-designated pages directly. Completed
prefill copies are synchronous, chunked by the specified write buffer, and paced
by the specified write bandwidth. Read bandwidth is the physical GPU's bandwidth;
this is an optimistic HBF-read emulation, not a measurement of HBF hardware.
The diagnostic policy is forbidden for benchmark mode. Scientific runs require
an authenticated earlier controller state and explicit timing/features/limits.
No global package, other engine, or unrelated GPU process is modified.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from pathlib import Path
import time

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorWorkerMetadata,
    KVConnectorRole)
from vllm.v1.core.sched.scheduler import Scheduler
from tier_pool import TierPool


def plan():
    p=Path(os.environ['RANCHO_DIRECT_PLAN'])
    result=json.loads(p.read_text())
    if result['block_tokens'] != 16 or result['native_block_bytes'] != 1572864:
        raise ValueError('Unexpected native geometry')
    if result['mode'] == 'benchmark' and result['policy'] in ('diagnostic_admit_all','calibration_observe'):
        raise ValueError('A diagnostic admission fixture is not the current algorithm')
    return result


def log_event(config, name, event):
    with (Path(config['output'])/name).open('a') as handle:
        handle.write(json.dumps(event,allow_nan=False)+'\n')


@dataclass
class Step(KVConnectorMetadata):
    candidates: list = field(default_factory=list)
    destinations: list = field(default_factory=list)


@dataclass
class Copied(KVConnectorWorkerMetadata):
    pairs: list = field(default_factory=list)
    events: list = field(default_factory=list)
    gpu_completed: bool = True
    def aggregate(self, other):
        raise ValueError('This prototype is qualified only for one worker')


class DirectConnector(KVConnectorBase_V1):
    @property
    def requires_kv_delivery(self): return False
    def __init__(self,vllm_config,role,kv_cache_config):
        super().__init__(vllm_config,role,kv_cache_config)
        self.config=plan();self.pages=None;self.output=None;self.policy=None
        self.event_id=0;self.now_us=-1
        if role == KVConnectorRole.WORKER and self.config['policy'] not in ('diagnostic_admit_all','calibration_observe','baseline'):
            if self.config.get('updated_rancho'):
                from updated_policy import UpdatedAdmission
                self.policy=UpdatedAdmission(self.config)
                self.event_id=0;self.now_us=self.config['fit']['cut_us']-1
            else:
                raise ValueError("HBF serving requires the updated RANCHO policy")
    def register_kv_caches(self,kv_caches):
        from native_pages import NativeCachePages
        self.pages=NativeCachePages(self._kv_cache_config,kv_caches)
        if len(self.pages.groups)!=1 or len(self.pages.pages)!=48:
            raise ValueError('Expected all 48 full-attention Qwen3-30B layers')
        if sum(v.shape[1] for v in self.pages.pages.values())!=self.config['native_block_bytes']:
            raise ValueError('Native page byte count changed')
        log_event(self.config,'worker.jsonl',dict(event='REGISTERED_NATIVE_PAGES',
                  layout_sha256=self.pages.layout_sha256,layers=len(self.pages.pages)))

    def get_num_new_matched_tokens(self,request,num_computed_tokens): return 0,False
    def update_state_after_alloc(self,request,blocks,num_external_tokens): pass
    def build_connector_meta(self,scheduler_output): return Step()
    def start_load_kv(self,forward_context,**kwargs): pass
    def wait_for_layer_load(self,layer_name): pass
    def save_kv_layer(self,layer_name,kv_layer,attn_metadata,**kwargs): pass

    def wait_for_save(self):
        import torch
        metadata=self._get_connector_metadata()
        if not isinstance(metadata,Step): raise ValueError('Unexpected scheduler metadata')
        self.output=Copied()
        if not metadata.candidates:return
        if self.pages is None: raise ValueError('Native pages not registered')
        torch.cuda.synchronize()
        destinations=list(metadata.destinations)
        for candidate in metadata.candidates:
            sources=list(candidate['sources'])
            event=dict(request_id=candidate['request_id'],source_pages=len(sources),
                       prefill_GPU_completed=True,policy=self.config['policy'],
                       features=candidate['features'],policy_us=candidate['policy_us'])
            if self.config['policy']=='calibration_observe':
                event['action']='CALIBRATION_OBSERVATION_ONLY'
                self.output.events.append(event)
                continue
            if not sources:
                event['action']='NO_NEW_FULL_SUFFIX';self.output.events.append(event);continue
            if len(sources)>len(destinations):
                event['action']='NO_PHYSICAL_DESTINATION_CAPACITY';self.output.events.append(event);continue
            if self.policy is not None:
                self.now_us=max(self.now_us,candidate['policy_us'])
                decision=self.policy.hbf_prefill_complete(self.event_id,self.now_us,
                    candidate['features'],len(sources),len(sources),
                    computed_prompt_tokens=candidate['prompt_tokens'],
                    prompt_tokens=candidate['prompt_tokens'],gpu_completed=True)
                self.event_id+=1
                event['decision']=decision
                if not decision['admit']:
                    event['action']='POLICY_REJECT';self.output.events.append(event);continue
            targets=destinations[:len(sources)];del destinations[:len(sources)]
            size=self.config['native_block_bytes']
            per_chunk=self.config['write_buffer_bytes']//size
            if per_chunk<1:raise ValueError('Write buffer cannot hold one native page')
            started=time.monotonic_ns()
            for offset in range(0,len(sources),per_chunk):
                src=sources[offset:offset+per_chunk];dst=targets[offset:offset+per_chunk]
                begin=time.monotonic_ns()
                for view in self.pages.pages.values():
                    source_ids=torch.tensor(src,device=view.device,dtype=torch.long)
                    target_ids=torch.tensor(dst,device=view.device,dtype=torch.long)
                    payload=view.index_select(0,source_ids)
                    view.index_copy_(0,target_ids,payload)
                torch.cuda.synchronize()
                deadline=begin+(len(src)*size*1000000000+self.config['write_bandwidth_bytes_s']-1)//self.config['write_bandwidth_bytes_s']
                delay=(deadline-time.monotonic_ns())/1e9
                if delay>0:time.sleep(delay)
            self.output.pairs.extend(zip(sources,targets))
            event.update(action='COPIED',native_bytes=len(sources)*size,
                         copy_and_service_ns=time.monotonic_ns()-started)
            self.output.events.append(event)
        log_event(self.config,'worker.jsonl',dict(event='COMPLETED_PREFILL_OFFERS',
                  offers=self.output.events,gpu_copy_completed=True))

    def build_connector_worker_meta(self):
        result=self.output;self.output=None
        return result


class DirectScheduler(Scheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.direct_plan=plan()
        if self.scheduler_config.async_scheduling:
            raise ValueError('Direct migration requires synchronous scheduling')
        managers=self.kv_cache_manager.coordinator.single_type_managers
        self.tiers=TierPool(self.kv_cache_manager.block_pool,managers,
                           self.direct_plan['HBM_pages'],self.direct_plan['HBF_pages'])
        self.first_computed={};self.offered=set()

    def schedule(self,*args,**kwargs):
        output=super().schedule(*args,**kwargs)
        self.tiers.refresh_worker_tables(output)
        beginnings={r.req_id:r.num_computed_tokens for r in output.scheduled_new_reqs}
        cached=output.scheduled_cached_reqs
        beginnings.update(zip(cached.req_ids,cached.num_computed_tokens))
        meta=Step(destinations=self.tiers.available_hbf())
        tables=self.tiers.managers[0].req_to_blocks
        for rid,count in output.num_scheduled_tokens.items():
            request=self.requests[rid]
            before=beginnings[rid]
            first_seen=rid not in self.first_computed
            if rid in cached.resumed_req_ids and rid not in self.offered:
                self.first_computed[rid]=before
            self.first_computed.setdefault(rid,before)
            prompt=len(request.prompt_token_ids)
            if first_seen:
                prefix=tables[rid][:before//16]
                log_event(self.direct_plan,'scheduler.jsonl',dict(event='INITIAL_PREFIX_LOOKUP',
                    request_id=rid,prompt_tokens=prompt,cached_tokens=before,
                    cached_HBF_full_pages=sum(self.tiers.is_hbf(b.block_id) for b in prefix),
                    cached_HBM_full_pages=sum(self.tiers.is_hbm(b.block_id) for b in prefix)))
            if rid not in self.offered and before<prompt<=before+count:
                if before+count!=prompt:raise ValueError('Unqualified prefill/decode boundary')
                external=rid
                if external not in self.direct_plan['requests']:
                    match=re.fullmatch(r'(.+)-[0-9a-f]{8}',rid)
                    if match is None or match[1] not in self.direct_plan['requests']:
                        raise ValueError('Unbound native/external request identity: '+rid)
                    external=match[1]
                info=self.direct_plan['requests'][external]
                sources=[b.block_id for b in tables[rid][self.first_computed[rid]//16:prompt//16]
                         if self.tiers.is_hbm(b.block_id)]
                if len(set(sources))!=len(sources):raise ValueError('Duplicate native suffix page')
                feature_start=time.perf_counter_ns()
                prompt_keys=([k.hex() for k in __import__('controlled_workload').block_keys(request.prompt_token_ids)] if self.direct_plan.get('updated_rancho') else None)
                feature_cpu_ns=time.perf_counter_ns()-feature_start
                meta.candidates.append(dict(request_id=rid,prompt_tokens=prompt,
                    sources=sources,features=info.get('features'),policy_us=info.get('policy_us'),
                    prompt_keys=prompt_keys,feature_cpu_ns=feature_cpu_ns,
                    session=info.get('session'),suffix_begin=self.first_computed[rid]//16))
                self.offered.add(rid)
                log_event(self.direct_plan,'scheduler.jsonl',dict(event='PREFILL_SCHEDULED',
                    request_id=rid,prompt_tokens=prompt,initial_cached_tokens=self.first_computed[rid],
                    HBM_free_pages=self.tiers.hbm_free_count(),
                    HBF_available_pages=len(meta.destinations)))
        output.kv_connector_metadata=meta
        return output

    def update_from_output(self,scheduler_output,model_runner_output):
        conn=model_runner_output.kv_connector_output
        meta=conn.kv_connector_worker_meta if conn else None
        if meta is not None:
            if not isinstance(meta,Copied):raise ValueError('Unknown worker completion metadata')
            if meta.pairs:
                result=self.tiers.commit_completed_copies(meta.pairs,gpu_completed=meta.gpu_completed)
                log_event(self.direct_plan,'scheduler.jsonl',dict(event='PUBLISHED_NATIVE_HBF',**result))
        return super().update_from_output(scheduler_output,model_runner_output)
