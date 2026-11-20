"""Connect vLLM's scheduler and GPU worker to the two cache tiers"""

from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from pathlib import Path
import time

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.scheduler import Scheduler
from tier_pool import TierPool


def read_engine_settings():
    """Read the settings passed to this vLLM process"""
    settings_file = os.environ.get("RANCHO_HBF_ENGINE_SETTINGS")
    if settings_file is None:
        settings_file = os.environ["RANCHO_DIRECT_PLAN"]
    result = json.loads(Path(settings_file).read_text())
    if result["block_tokens"] != 16 or result["native_block_bytes"] != 1572864:
        raise ValueError("Unexpected native geometry")
    if result["mode"] == "benchmark" and result["policy"] in (
        "diagnostic_admit_all",
        "calibration_observe",
    ):
        raise ValueError("A diagnostic admission fixture is not the current algorithm")
    return result


def log_event(config, name, event):
    with (Path(config["output"]) / name).open("a") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")


@dataclass
class Step(KVConnectorMetadata):
    """The requests and available destinations sent to the worker"""

    candidates: list = field(default_factory=list)
    destinations: list = field(default_factory=list)


@dataclass
class Copied(KVConnectorWorkerMetadata):
    """The completed copies and decision events sent back by the worker"""

    pairs: list = field(default_factory=list)
    events: list = field(default_factory=list)
    gpu_completed: bool = True

    def aggregate(self, other):
        raise ValueError("This prototype is qualified only for one worker")


class DirectConnector(KVConnectorBase_V1):
    """Prepare GPU cache access and copy completed KV pages"""

    @property
    def requires_kv_delivery(self):
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.config = read_engine_settings()
        self.pages = None
        self.output = None
        self.policy = None
        self.event_id = 0
        self.now_us = -1
        if role == KVConnectorRole.WORKER and self.config["policy"] not in (
            "diagnostic_admit_all",
            "calibration_observe",
            "baseline",
        ):
            if self.config.get("updated_rancho"):
                from updated_policy import UpdatedAdmission

                self.policy = UpdatedAdmission(self.config)
                self.event_id = 0
                self.now_us = self.config["fit"]["cut_us"] - 1
            else:
                raise ValueError("HBF serving requires the updated RANCHO policy")

    def register_kv_caches(self, kv_caches):
        """Find the page layout and allocate the copy buffer"""
        from native_pages import NativeCachePages

        self.pages = NativeCachePages(self._kv_cache_config, kv_caches)
        if len(self.pages.groups) != 1 or len(self.pages.pages) != 48:
            raise ValueError("Expected all 48 full-attention Qwen3-30B layers")
        if (
            sum(layer_view.shape[1] for layer_view in self.pages.pages.values())
            != self.config["native_block_bytes"]
        ):
            raise ValueError("Native page byte count changed")
        log_event(
            self.config,
            "worker.jsonl",
            dict(
                event="REGISTERED_NATIVE_PAGES",
                layout_sha256=self.pages.layout_sha256,
                layers=len(self.pages.pages),
            ),
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return Step()

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        """Handle this step's write offers after the GPU finishes"""
        import torch

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, Step):
            raise ValueError("Unexpected scheduler metadata")
        self.output = Copied()
        if not metadata.candidates:
            return
        if self.pages is None:
            raise ValueError("Native pages not registered")
        torch.cuda.synchronize()
        destinations = list(metadata.destinations)
        for candidate in metadata.candidates:
            sources = list(candidate["sources"])
            event = dict(
                request_id=candidate["request_id"],
                source_pages=len(sources),
                prefill_GPU_completed=True,
                policy=self.config["policy"],
                features=candidate["features"],
                policy_us=candidate["policy_us"],
            )
            if self.config["policy"] == "calibration_observe":
                event["action"] = "CALIBRATION_OBSERVATION_ONLY"
                self.output.events.append(event)
                continue
            if not sources:
                event["action"] = "NO_NEW_FULL_SUFFIX"
                self.output.events.append(event)
                continue
            if len(sources) > len(destinations):
                event["action"] = "NO_PHYSICAL_DESTINATION_CAPACITY"
                self.output.events.append(event)
                continue
            if self.policy is not None:
                self.now_us = max(self.now_us, candidate["policy_us"])
                decision = self.policy.hbf_prefill_complete(
                    self.event_id,
                    self.now_us,
                    candidate["features"],
                    len(sources),
                    len(sources),
                    computed_prompt_tokens=candidate["prompt_tokens"],
                    prompt_tokens=candidate["prompt_tokens"],
                    gpu_completed=True,
                )
                self.event_id += 1
                event["decision"] = decision
                if not decision["admit"]:
                    event["action"] = "POLICY_REJECT"
                    self.output.events.append(event)
                    continue
            targets = destinations[: len(sources)]
            del destinations[: len(sources)]
            page_bytes = self.config["native_block_bytes"]
            pages_per_chunk = self.config["write_buffer_bytes"] // page_bytes
            if pages_per_chunk < 1:
                raise ValueError("Write buffer cannot hold one native page")
            started = time.monotonic_ns()
            for offset in range(0, len(sources), pages_per_chunk):
                source_pages = sources[offset : offset + pages_per_chunk]
                target_pages = targets[offset : offset + pages_per_chunk]
                begin = time.monotonic_ns()
                for view in self.pages.pages.values():
                    source_ids = torch.tensor(
                        source_pages, device=view.device, dtype=torch.long
                    )
                    target_ids = torch.tensor(
                        target_pages, device=view.device, dtype=torch.long
                    )
                    payload = view.index_select(0, source_ids)
                    view.index_copy_(0, target_ids, payload)
                torch.cuda.synchronize()
                deadline = (
                    begin
                    + (
                        len(source_pages) * page_bytes * 1000000000
                        + self.config["write_bandwidth_bytes_s"]
                        - 1
                    )
                    // self.config["write_bandwidth_bytes_s"]
                )
                delay = (deadline - time.monotonic_ns()) / 1e9
                if delay > 0:
                    time.sleep(delay)
            self.output.pairs.extend(zip(sources, targets))
            event.update(
                action="COPIED",
                native_bytes=len(sources) * page_bytes,
                copy_and_service_ns=time.monotonic_ns() - started,
            )
            self.output.events.append(event)
        log_event(
            self.config,
            "worker.jsonl",
            dict(
                event="COMPLETED_PREFILL_OFFERS",
                offers=self.output.events,
                gpu_copy_completed=True,
            ),
        )

    def build_connector_worker_meta(self):
        result = self.output
        self.output = None
        return result


class DirectScheduler(Scheduler):
    """Prepare page moves and publish the copies that finish"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_settings = read_engine_settings()
        if self.scheduler_config.async_scheduling:
            raise ValueError("Direct migration requires synchronous scheduling")
        managers = self.kv_cache_manager.coordinator.single_type_managers
        self.tiers = TierPool(
            self.kv_cache_manager.block_pool,
            managers,
            self.engine_settings["HBM_pages"],
            self.engine_settings["HBF_pages"],
        )
        self.first_computed = {}
        self.offered = set()

    def schedule(self, *args, **kwargs):
        """Plan page moves for completed prefill and send them to the worker"""
        output = super().schedule(*args, **kwargs)
        self.tiers.refresh_worker_tables(output)
        beginnings = {
            scheduled_request.req_id: scheduled_request.num_computed_tokens
            for scheduled_request in output.scheduled_new_reqs
        }
        cached = output.scheduled_cached_reqs
        beginnings.update(zip(cached.req_ids, cached.num_computed_tokens))
        meta = Step(destinations=self.tiers.available_hbf())
        tables = self.tiers.managers[0].req_to_blocks
        for request_id, count in output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            cached_tokens_before = beginnings[request_id]
            first_seen = request_id not in self.first_computed
            if request_id in cached.resumed_req_ids and request_id not in self.offered:
                self.first_computed[request_id] = cached_tokens_before
            self.first_computed.setdefault(request_id, cached_tokens_before)
            prompt_token_count = len(request.prompt_token_ids)
            if first_seen:
                prefix = tables[request_id][: cached_tokens_before // 16]
                log_event(
                    self.engine_settings,
                    "scheduler.jsonl",
                    dict(
                        event="INITIAL_PREFIX_LOOKUP",
                        request_id=request_id,
                        prompt_tokens=prompt_token_count,
                        cached_tokens=cached_tokens_before,
                        cached_HBF_full_pages=sum(
                            self.tiers.is_hbf(cached_block.block_id)
                            for cached_block in prefix
                        ),
                        cached_HBM_full_pages=sum(
                            self.tiers.is_hbm(cached_block.block_id)
                            for cached_block in prefix
                        ),
                    ),
                )
            if (
                request_id not in self.offered
                and cached_tokens_before
                < prompt_token_count
                <= cached_tokens_before + count
            ):
                if cached_tokens_before + count != prompt_token_count:
                    raise ValueError("Unqualified prefill/decode boundary")
                trace_request_id = request_id
                if trace_request_id not in self.engine_settings["requests"]:
                    match = re.fullmatch(r"(.+)-[0-9a-f]{8}", request_id)
                    if (
                        match is None
                        or match[1] not in self.engine_settings["requests"]
                    ):
                        raise ValueError(
                            "Unbound native/external request identity: " + request_id
                        )
                    trace_request_id = match[1]
                info = self.engine_settings["requests"][trace_request_id]
                sources = [
                    cached_block.block_id
                    for cached_block in tables[request_id][
                        self.first_computed[request_id] // 16 : prompt_token_count // 16
                    ]
                    if self.tiers.is_hbm(cached_block.block_id)
                ]
                if len(set(sources)) != len(sources):
                    raise ValueError("Duplicate native suffix page")
                feature_start = time.perf_counter_ns()
                prompt_keys = (
                    [
                        block_hash.hex()
                        for block_hash in __import__("controlled_workload").block_keys(
                            request.prompt_token_ids
                        )
                    ]
                    if self.engine_settings.get("updated_rancho")
                    else None
                )
                feature_cpu_ns = time.perf_counter_ns() - feature_start
                meta.candidates.append(
                    dict(
                        request_id=request_id,
                        prompt_tokens=prompt_token_count,
                        sources=sources,
                        features=info.get("features"),
                        policy_us=info.get("policy_us"),
                        prompt_keys=prompt_keys,
                        feature_cpu_ns=feature_cpu_ns,
                        session=info.get("session"),
                        suffix_begin=self.first_computed[request_id] // 16,
                    )
                )
                self.offered.add(request_id)
                log_event(
                    self.engine_settings,
                    "scheduler.jsonl",
                    dict(
                        event="PREFILL_SCHEDULED",
                        request_id=request_id,
                        prompt_tokens=prompt_token_count,
                        initial_cached_tokens=self.first_computed[request_id],
                        HBM_free_pages=self.tiers.hbm_free_count(),
                        HBF_available_pages=len(meta.destinations),
                    ),
                )
        output.kv_connector_metadata = meta
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        """Apply the worker's completed moves to the cache tables"""
        connector_output = model_runner_output.kv_connector_output
        completed_copies = (
            connector_output.kv_connector_worker_meta if connector_output else None
        )
        if completed_copies is not None:
            if not isinstance(completed_copies, Copied):
                raise ValueError("Unknown worker completion metadata")
            if completed_copies.pairs:
                result = self.tiers.commit_completed_copies(
                    completed_copies.pairs, gpu_completed=completed_copies.gpu_completed
                )
                log_event(
                    self.engine_settings,
                    "scheduler.jsonl",
                    dict(event="PUBLISHED_NATIVE_HBF", **result),
                )
        return super().update_from_output(scheduler_output, model_runner_output)
