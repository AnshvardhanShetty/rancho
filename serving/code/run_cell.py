"""One real-GPU matched-benefit fixed-duration serving cell and independent audit.

Physical-study extension: physical controller time, real daily accounting, and
zero initial post-warm-up write credits; original feature history remains causal.

Assumptions: immutable constructed workload, actual serial cache-fill warm-up,
one fresh engine, no preloaded KV, same offline eligibility denominator in all
arms. Native contiguous-prefix lookup supplies hits. Throughput counts all
measured outputs over max(arrival window, last completion), including drain.
One controlled HBM baseline retains declared hot prefixes after actual GPU
computation; this is explicitly not stock-vLLM eviction. Live frozen offered-suffix GBT, filtered HBM-miss history,
cost-band C7 and half-behind make all HBF decisions. The archived predictor is
transferred to scaled synthetic serving; current history starts empty and is
serialized in actual completed-prefill order. HBF reads remain optimistic GPU-memory emulation.
No hit counters, admission masks, latency shapes or speedups are forced.
"""
import argparse,hashlib,json,math,os,re,time
from pathlib import Path
from controlled_workload import prompts,bound_requests,eligibility
from synthetic_workload import FIT
from updated_model import SuffixGBT
from campaign_state import save
from metrics import summarise,describe
from physical_time import PhysicalBudget

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def index(rid):
 m=re.fullmatch(r'row(\d+)(?:-[0-9a-f]{8})?',rid)
 if not m:raise ValueError('Unrecognized native request identity')
 return int(m[1])

def compute(root,setup):
 cfg=read(root/'PLAN.json');records=rows(root/'requests.jsonl');lookup=[x for x in rows(root/'scheduler.jsonl') if x['event']=='INITIAL_PREFIX_LOOKUP']
 n=setup['groups']+setup['requests'];assert len(records)==len(lookup)==n
 byrow={index(x['request_id']):x for x in lookup};assert set(byrow)==set(range(n))
 assert sorted(r['row'] for r in records)==list(range(n)) and all(r['status']=='COMPLETE' and r['generated_tokens']==64 for r in records)
 mask=eligibility(setup);measured=[v for v in records if not v['warmup']];assert len(measured)==setup['requests']
 hit=total=hbf=hbm=0;allhit=alltotal=0
 for i,bits in enumerate(mask):
  e=byrow[i];k=e['cached_tokens']//16
  assert e['cached_tokens']==16*k==16*(e['cached_HBF_full_pages']+e['cached_HBM_full_pages'])
  assert 0<=k<=len(bits) and all(bits[:k]),'Cached prefix eligibility differs from declared synthetic history'
  got=sum(bits[:k]);den=sum(bits);allhit+=got;alltotal+=den
  if i>=setup['groups']:
   hit+=got;total+=den;hbf+=e['cached_HBF_full_pages'];hbm+=e['cached_HBM_full_pages']
 assert hit==hbf+hbm and 0<=hit<=total
 m=summarise(measured);allm=summarise(records);begin=measured[0]['measurement_start_ns'];end=max(v['finish_ns'] for v in measured)
 interval=max(setup['measurement_seconds'],(end-begin)/1e9)
 m.update(output_tokens_per_second=m['output_tokens']/interval,completed_requests_per_second=len(measured)/interval,arrival_window_seconds=setup['measurement_seconds'],window_and_drain_seconds=interval,drain_seconds=max(0,(end-begin)/1e9-setup['measurement_seconds']))
 steps=rows(root/'concurrency.jsonl');active=[v for v in steps if v['start_ns']>=begin]
 offers=[o for v in rows(root/'worker.jsonl') for o in v.get('offers',[])] if cfg['HBF_pages'] else []
 decision_ns=[o['decision']['decision_cpu_ns'] for o in offers if 'decision' in o and index(o['request_id'])>=setup['groups']]
 admission=dict(live_calls=sum('decision' in o for o in offers),measured_live_calls=len(decision_ns),admitted=sum(bool(o.get('decision',{}).get('admit')) for o in offers),paid_pages=sum(o.get('decision',{}).get('paid_cost',0) for o in offers),copied_bytes=sum(o.get('native_bytes',0) for o in offers),decision_CPU_microseconds=describe([v/1000 for v in decision_ns]),decision_CPU_seconds=sum(decision_ns)/1e9)
 result=dict(status='COMPLETE_PENDING_AUDIT',job=read(root/'JOB.json'),metrics=m,cold_start_inclusive_metrics=allm,offline_hit_percent=100*hit/total,eligible_accesses=total,eligible_hits=hit,HBF_hit_pp=100*hbf/total,HBM_hit_pp=100*hbm/total,cold_start_inclusive_offline_hit_percent=100*allhit/alltotal,hit_ratio=hit/total,admission=admission,concurrency=dict(peak_active=max((v['resident_requests'] for v in active),default=0),peak_waiting=max((v['waiting_requests'] for v in active),default=0),peak_scheduled_decode=max((v['scheduled_decode_requests'] for v in active),default=0),minimum_free_HBM_pages=min((v['HBM_free_after_schedule'] for v in active),default=None),native_allocation_refusals=sum(v['native_KV_allocation_refusals'] for v in active)),payload_sha256=cfg['payload_sha256'],GPU_execution=True,synthetic=True,full_queue_drained=True,warmup_real_requests=setup['groups'],policy_version='filtered_linear_suffix_GBT_cost_band_half_behind_physical_v2',HBF_reads='unthrottled GPU-memory emulation')
 if cfg['HBF_pages']:
  physical_rows=[v for v in rows(root/'worker.jsonl') if v.get('event')=='PHYSICAL_CONTROLLER_CLOCK_START']
  assert len(physical_rows)==1
  origin=physical_rows[0]['origin_ns'];ledger=PhysicalBudget(cfg['HBF_pages'],origin)
  for offer in offers:
   if 'decision' not in offer:continue
   d=offer['decision'];warm=index(offer['request_id'])<setup['groups']
   allowance=ledger.limit(d['physical_decision_ns'],warm,None if warm else begin)
   assert d['physical_write_limit_pages']==allowance and d['physical_warmup']==warm
   assert d['physical_origin_ns']==origin
   assert d['controller_time_us']==cfg['fit']['cut_us']+(d['physical_decision_ns']-origin)//1000
   ledger.charge(d['paid_cost'],warm,allowance)
  assert ledger.total==admission['paid_pages']
  result['physical_writes']=ledger.summary(end,begin,interval)
  assert result['physical_writes']['steady_rate_respected'] and result['physical_writes']['daily_allowance_respected']
  total_ns=[o['decision']['total_admission_cpu_ns'] for o in offers if 'decision' in o and index(o['request_id'])>=setup['groups']]
  admission['predictor_controller_and_guard_CPU_microseconds']=describe([v/1000 for v in total_ns])
  history_ns=[o.get('history_cpu_ns',0) for o in offers if index(o['request_id'])>=setup['groups']]
  admission['filtered_history_CPU_microseconds']=describe([v/1000 for v in history_ns])
  admission['total_admission_CPU_microseconds']=describe([(o['decision']['total_admission_cpu_ns']+o.get('history_cpu_ns',0)+o.get('scheduler_feature_cpu_ns',0))/1000 for o in offers if 'decision' in o and index(o['request_id'])>=setup['groups']])
  for label,key in [('predictor_CPU_microseconds','predictor_cpu_ns'),('controller_CPU_microseconds','controller_cpu_ns')]:
   admission[label]=describe([o['decision'][key]/1000 for o in offers if 'decision' in o and index(o['request_id'])>=setup['groups']])
  admission['scheduler_feature_CPU_microseconds']=describe([o.get('scheduler_feature_cpu_ns',0)/1000 for o in offers if index(o['request_id'])>=setup['groups']])
  decisions=[o['decision'] for o in offers if 'decision' in o]
  admission['mechanisms']=dict(threshold_active=sum(d['have_threshold'] for d in decisions),history_boosted=sum(d['history_multiplier']>1 for d in decisions),half_behind=sum(d['half_behind'] for d in decisions),score_rejections=sum(not d['score_admit_before_physical_guard'] for d in decisions),physical_guard_rejections=sum(d['physical_guard_rejected'] for d in decisions))
 return result

def audit(root):
 from controllers import Wallet
 cfg=read(root/'PLAN.json');setup=read(root/'SETUP.json');saved=read(root/'RESULT.json');got=compute(root,setup)
 assert saved==got,'Independent timestamp/eligibility/metric recomputation mismatch'
 # Independent algebraic cross-check: measured contexts all occur in the
 # immutable earlier reference trace; native request results separately report
 # first-lookup counters are used for offline credit; internal reuses after
 # preemption are not counted again. No constant95 value enters a hit counter.
 actual=[v for v in rows(root/'requests.jsonl') if not v['warmup']]
 measured_lookups={index(v['request_id']):v for v in rows(root/'scheduler.jsonl') if v['event']=='INITIAL_PREFIX_LOOKUP' and index(v['request_id'])>=setup['groups']}
 expected_den=len(actual)*(setup['prefix_tokens']+setup['unique_suffix_tokens'])//16
 independent_hits=0
 for v in actual:
  q=measured_lookups[v['row']]
  assert 0<=q['cached_tokens']<=setup['prefix_tokens'] and q['cached_tokens']%16==0
  assert v['cached_tokens'] is not None and v['cached_tokens']>=0
  independent_hits+=q['cached_tokens']//16
 assert saved['eligible_accesses']==expected_den and saved['eligible_hits']==independent_hits
 assert abs(saved['offline_hit_percent']-100*independent_hits/expected_den)<1e-12
 if cfg['ratio']:
  assert cfg['HBF_pages']==cfg['ratio']*(cfg['baseline_HBM_pages']-cfg['HBM_pages'])
  from updated_audit import audit_policy
  saved['algorithm_audit']=audit_policy(root,cfg,saved)
 elif cfg['baseline_prefix_retention']:
  from retention_audit import audit_retention
  saved['prefix_retention']=audit_retention(root,cfg,saved)
 save(root/'AUDIT.json',dict(status='PASS_INDEPENDENT_SERVING_AND_POLICY_AUDIT',all_requests_checked=True,offline_eligibility_recomputed=True,actual_native_hits_checked=True,all_outputs64=True,full_queue_drained=True,ratio_checked=True,paid_writes_checked=True,policy_decisions_replayed=bool(cfg['ratio']),baseline_retention=saved.get('prefix_retention'),algorithm_audit=saved.get('algorithm_audit')))
 print('PASS_INDEPENDENT_SERVING_AND_POLICY_AUDIT',flush=True)

def run(setup,job,root,manifest):
 source=Path(__file__).resolve().parent
 if manifest:
  for n,h in read(manifest).items():assert digest(source/n)==h,n
 os.environ['RANCHO_DIAGNOSTIC_QUEUE']='indexed';root.mkdir(exist_ok=False)
 hbm,hbf=job['hbm'],job['hbf'];payload=prompts(setup);N=len(payload);P=setup['prefix_tokens']+setup['unique_suffix_tokens']
 assert P+64<=setup['context_limit'] and math.ceil((P+64)/16)<=hbm and hbm+hbf<=16384
 assert N==setup['groups']+setup['requests']<=setup['max_num_seqs']
 cfg=dict(mode='benchmark',policy='filtered_linear_suffix_costband_half_behind' if hbf else 'baseline',HBM_pages=hbm,HBF_pages=hbf,baseline_HBM_pages=setup['baseline_pages'],ratio=job['ratio'],block_tokens=16,native_block_bytes=setup['native_page_bytes'],output=str(root),requests=bound_requests(setup),write_bandwidth_bytes_s=setup['write_bandwidth_bytes_s'],write_buffer_bytes=setup['write_buffer_bytes'],write_budget_pages=hbf,physical_write_accounting=bool(hbf),n_requests=N,arrival_rate=job.get('rate'),payload_sha256=hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest(),input_tokens_per_request=P,baseline_prefix_retention=bool(job.get('retain_prefixes',not bool(hbf))),shared_prefix_tokens=setup['prefix_tokens'],shared_groups=setup['groups'])
 if hbf:cfg.update(controller_history=job['controller'],controller_history_sha256=digest(job['controller']),fit=job['fit'],updated_rancho=True,repetition=job['repetition'])
 save(root/'PLAN.json',cfg);save(root/'SETUP.json',setup);save(root/'JOB.json',job);os.environ['RANCHO_DIRECT_PLAN']=str(root/'PLAN.json')
 from vllm import LLM,SamplingParams
 from vllm.config import KVTransferConfig
 from vllm.sampling_params import RequestOutputKind
 model=__import__('model_location').model_path();llm=None
 try:
  llm=LLM(model=model,moe_backend='triton',tensor_parallel_size=1,dtype='bfloat16',enforce_eager=True,enable_prefix_caching=True,max_model_len=setup['context_limit'],max_num_seqs=setup['max_num_seqs'],max_num_batched_tokens=2048,block_size=16,kv_cache_memory_bytes=(1+hbm+hbf)*setup['native_page_bytes'],gpu_memory_utilization=.15,distributed_executor_backend='mp',async_scheduling=False,scheduler_cls='concurrency_backend.TieredScheduler' if hbf else 'retained_baseline.RetainedBaselineScheduler' if cfg['baseline_prefix_retention'] else 'concurrency_backend.BaselineScheduler',worker_extension_cls='worker_probe.MemoryProbe',kv_transfer_config=KVTransferConfig(kv_connector='HBFConnector' if hbf else 'DirectConnector',kv_connector_module_path='physical_hbf' if hbf else 'direct_backend',kv_role='kv_both'))
  geometry=llm.collective_rpc('rancho_memory_snapshot');save(root/'GEOMETRY.json',geometry);assert geometry[0]['native_cache_blocks']==1+hbm+hbf
  state={};aliases={};pending=set();completed=0;serial=0
  def submit(i,scheduled,begin):
   rid=f'row{i}';state[rid]=dict(row=i,expected_output_tokens=64,scheduled_ns=scheduled,dispatch_ns=time.monotonic_ns(),first_token_ns=None,last_token_ns=None,generated_tokens=0,finish_ns=None,status='RUNNING',cached_tokens=None,warmup=i<setup['groups'],measurement_start_ns=begin)
   internal=llm.llm_engine.add_request(rid,dict(prompt_token_ids=payload[i]),SamplingParams(temperature=0,max_tokens=64,min_tokens=64,ignore_eos=True,detokenize=False,seed=17,output_kind=RequestOutputKind.DELTA));aliases[internal]=rid;aliases[rid]=rid;pending.add(rid)
  def step(handle):
   nonlocal completed
   outputs=llm.llm_engine.step();now=time.monotonic_ns()
   for request in outputs:
    rid=aliases[request.request_id];r=state[rid];count=sum(len(item.token_ids) for item in request.outputs)
    if count:r['first_token_ns']=r['first_token_ns'] or now;r['last_token_ns']=now;r['generated_tokens']+=count
    r['cached_tokens']=request.num_cached_tokens
    if request.finished:
     r.update(finish_ns=now,status='COMPLETE');pending.remove(rid);completed+=1;handle.write(json.dumps(r)+'\n')
  with (root/'requests.jsonl').open('x',buffering=1) as handle:
   for g in range(setup['groups']):
    now=time.monotonic_ns();submit(g,now,now)
    while pending:step(handle)
   begin=time.monotonic_ns();save(root/'MEASUREMENT_CLOCK.json',dict(measurement_start_ns=begin,clock='monotonic_ns',startup_requests_completed=setup['groups']));dispatched=setup['groups'];lastprogress=begin
   while completed<N:
    now=time.monotonic_ns()
    while dispatched<N:
     at=dispatched-setup['groups'];scheduled=begin if job.get('pilot') else begin+round(at*1e9/job['rate'])
     if now<scheduled:break
     submit(dispatched,scheduled,begin);dispatched+=1;now=time.monotonic_ns()
    if not pending:
     at=dispatched-setup['groups'];next_ns=begin+round(at*1e9/job['rate']);time.sleep(max(0,min(.02,(next_ns-time.monotonic_ns())/1e9)));continue
    step(handle);now=time.monotonic_ns()
    if now-lastprogress>15_000_000_000 or completed==N:
     save(root/'PROGRESS.json',dict(dispatched=dispatched-setup['groups'],completed=completed-setup['groups'],pending=len(pending),elapsed_seconds=(now-begin)/1e9));lastprogress=now
 finally:
  if llm is not None:llm.llm_engine.engine_core.shutdown()
 result=compute(root,setup);save(root/'RESULT.json',result);print(json.dumps({'status':'CELL_COMPLETE_PENDING_AUDIT','offline_hit_percent':result['offline_hit_percent'],'TTFT_p95_s':result['metrics']['TTFT_seconds']['p95'],'output_tokens_s':result['metrics']['output_tokens_per_second']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--setup');p.add_argument('--job');p.add_argument('--output');p.add_argument('--manifest');p.add_argument('--audit');a=p.parse_args()
 if a.audit:audit(Path(a.audit))
 else:run(read(a.setup),read(a.job),Path(a.output),Path(a.manifest))
