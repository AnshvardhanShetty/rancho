"""Save full KV blocks to SSD before vLLM reuses their GPU slots

The scheduler records block history and asks the policy what to keep
The connector stages cache hits before allocation can reuse their old slots
Saved blocks are copied back before the model reads them
"""

import importlib
import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id
from native_pages import NativeCachePages
from ssd_storage import Storage
from ssd_policy import Offer


def read_serving_settings():
    """Read the plan supplied by the launcher."""
    settings_file = os.environ.get("RANCHO_SSD_SERVING_SETTINGS")
    if settings_file is None:
        settings_file = os.environ["RANCHO_SSD_PLAN"]
    return json.loads(Path(settings_file).read_text())


def emit(serving_settings, name, fields):
    """Append one engine event to the output log."""
    with (Path(serving_settings["output"]) / (name + ".jsonl")).open("a") as log_file:
        log_file.write(json.dumps(fields, allow_nan=False) + "\n")


store = None
loads = {}


@dataclass
class Transfers(KVConnectorMetadata):
    """List the GPU destinations to restore before model execution."""

    destinations: list = field(default_factory=list)


class Connector(KVConnectorBase_V1):
    """Coordinate decisions, SSD transfers and block history."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.serving_settings = read_serving_settings()
        self.worker_role = role == KVConnectorRole.WORKER
        self.pending = {}
        self.destinations = {}
        if role == KVConnectorRole.SCHEDULER:
            module, name = self.serving_settings["policy_factory"].split(":")
            self.decide = getattr(importlib.import_module(module), name)(
                self.serving_settings["policy_settings"]
            )
            self.clock_start = time.monotonic()
            self.history = {}
            self.resident = {}
            self.request_touched = {}
            self.evictions = self.absent = self.retained = self.rejected = (
                self.restored
            ) = self.local_islands = 0
            self.new_pages = set()
            self.now = 0.0

    @property
    def requires_kv_delivery(self):
        return False

    def register_kv_caches(self, caches):
        """Open the SSD store using the worker's KV page layout"""
        global store
        if store is not None:
            raise RuntimeError("One engine per serving process required")
        pages = NativeCachePages(self._kv_cache_config, caches)
        if pages.num_blocks != self.serving_settings["hbm_pages"] + 1:
            raise ValueError("HBM allocation mismatch")
        for name, view in pages.pages.items():
            tensor = caches[name]
            if (
                tuple(tensor.shape) != (pages.num_blocks, 4, 16, 256)
                or tensor.stride(0) * tensor.element_size() != view.stride(0)
                or tensor[0].numel() * tensor.element_size() != view.shape[1]
                or tensor.data_ptr() != view.data_ptr()
            ):
                raise ValueError("Native KV page address or geometry mismatch: " + name)
        store = Storage(
            self.serving_settings["ssd_directory"],
            self.serving_settings["ssd_pages"],
            pages,
            self.serving_settings.get("verify_payloads", False),
        )
        emit(
            self.serving_settings,
            "geometry",
            dict(
                page_bytes=store.page_bytes,
                hbm_pages=pages.num_blocks - 1,
                ssd_pages=store.capacity,
                mount=store.mount,
                direct=True,
                dsync=True,
            ),
        )

    def shutdown(self):
        global store
        if self.worker_role and store is not None:
            store.close()
            store = None

    def begin_allocation(self, request):
        """Select the request metadata and advance the policy clock."""
        self.trace_request_id = re.sub(r"-[0-9a-f]{8}$", "", request.request_id)
        info = self.serving_settings["requests"][self.trace_request_id]
        if self.serving_settings["clock"] == "monotonic":
            self.now = time.monotonic() - self.clock_start
        else:
            self.now = max(self.now, info["policy_seconds"])
        return info

    def record_pages(self, request, manager):
        """Record one access per block per request."""
        request_info = self.serving_settings["requests"][self.trace_request_id]
        touched = self.request_touched.setdefault(request.request_id, set())
        blocks = manager.get_blocks(request.request_id).blocks[0]
        for block, block_hash in zip(
            blocks, request.block_hashes[: len(request.prompt_token_ids) // 16]
        ):
            key = bytes(block_hash)
            if (
                block.block_hash is None
                or bytes(get_block_hash(block.block_hash)) != key
            ):
                continue
            page = block.block_id
            acquired = page not in self.resident
            if acquired:
                self.resident[page] = (key, self.now)
                self.new_pages.add(page)
            if self.resident[page][0] != key:
                raise RuntimeError("Unobserved HBM reuse")
            if key in touched:
                if acquired:
                    self.history[key].update(
                        last=self.now,
                        rid=self.trace_request_id,
                        features=request_info["features"],
                    )
                continue
            touched.add(key)
            state = self.history.setdefault(key, dict(count=0, evictions=0))
            state.update(
                count=state["count"] + 1,
                last=self.now,
                rid=self.trace_request_id,
                features=request_info["features"],
            )

    def evict(self, block):
        """Ask whether to save an unused GPU block before its slot is reused."""
        resident = self.resident.pop(block.block_id, None)
        if resident is None:
            return
        key, born = resident
        assert (
            block.ref_cnt == 0
            and block.block_hash
            and get_group_id(block.block_hash) == 0
            and get_block_hash(block.block_hash) == key
        )
        if block.block_id in self.new_pages:
            raise RuntimeError(
                "Attempted to evict unfinished KV in the same scheduler step"
            )
        block_history = self.history[key]
        prior_evictions = block_history["evictions"]
        block_history["evictions"] += 1
        self.evictions += 1
        if key in store.slots:
            store.slots.move_to_end(key)
            self.retained += 1
            return
        offer = Offer(
            key,
            self.evictions - 1,
            self.absent,
            self.now,
            store.page_bytes,
            store.writes,
            self.serving_settings["write_budget_pages"],
            store.capacity,
            prior_evictions,
            block_history["count"],
            self.now - born,
            self.now - block_history["last"],
            block_history["rid"],
            dict(block_history["features"]),
        )
        self.absent += 1
        decision = self.decide(offer)
        if type(decision) != bool:
            raise TypeError("Policy must return bool")
        if decision and store.capacity and store.writes < offer.write_budget_pages:
            store.put(key, block.block_id)
        else:
            self.rejected += 1

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Find how much more of the input is available from GPU or SSD."""
        begin = num_computed_tokens // 16
        matches = []
        for block_hash in request.block_hashes[
            begin : (len(request.prompt_token_ids) - 1) // 16
        ]:
            key = bytes(block_hash)
            local = self.pool.get_cached_block(block_hash, [0])
            if local and local[0].block_id not in self.new_pages:
                matches.append((key, local[0].block_id))
            elif key in store.slots:
                matches.append((key, None))
            else:
                break
        self.pending[request.request_id] = (begin, matches)
        return 16 * len(matches), False

    def stage_lookup(self, request):
        """Preserve needed KV before allocation can recycle its source."""
        staged = []
        for block_key, local_hbm_page in self.pending.get(request.request_id, (0, []))[
            1
        ]:
            if local_hbm_page is None:
                payload = store.read(block_key)
            else:
                payload = store.buffer.new_empty(store.page_bytes)
                store.copy(local_hbm_page, payload, True)
            staged.append((block_key, local_hbm_page, payload))
        self.staged = staged

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """Associate staged KV with the GPU slots allocated for this request."""
        begin, matches = self.pending.pop(request.request_id, (0, []))
        if not num_external_tokens:
            return
        assert num_external_tokens == 16 * len(matches) == 16 * len(self.staged)
        destination_pages = blocks.get_block_ids()[0][begin : begin + len(matches)]
        assert len(destination_pages) == len(matches)
        for page, (key, local_hbm_page, payload) in zip(destination_pages, self.staged):
            if page in self.destinations and self.destinations[page] != key:
                raise RuntimeError("Conflicting restore to a single HBM page")
            self.destinations[page] = key
            loads[page] = payload
            if local_hbm_page is None:
                if key in store.slots:
                    store.slots.move_to_end(key)
                self.restored += 1
            else:
                self.local_islands += 1
        self.staged = []

    def build_connector_meta(self, scheduler_output):
        meta = Transfers(list(self.destinations))
        self.destinations.clear()
        self.pending.clear()
        self.staged = []
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        """Restore the staged KV before the model reads it."""
        for page in self._get_connector_metadata().destinations:
            store.restore(page, loads.pop(page))

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


class SSDScheduler(Scheduler):
    """Call the SSD connector before GPU slots are reused."""

    def __init__(self, *args, **kwargs):
        """Connect the GPU allocation path to the SSD eviction and restore hooks"""
        super().__init__(*args, **kwargs)
        if (
            store is None
            or self.scheduler_config.async_scheduling
            or self.vllm_config.max_concurrent_batches != 1
        ):
            raise RuntimeError(
                "Requires one synchronous UniProcExecutor with registered local KV"
            )
        connector = self.connector
        manager = self.kv_cache_manager
        connector.pool = manager.block_pool
        if connector.pool.hash_block_size != 16:
            raise ValueError("Expected 16-token hashes")
        evict = connector.pool._maybe_evict_cached_block

        def before_reuse(block):
            if block.block_id in connector.resident:
                connector.evict(block)
            return evict(block)

        connector.pool._maybe_evict_cached_block = before_reuse
        allocate = manager.allocate_slots

        def tracked_allocate(request, *args, **kwargs):
            connector.begin_allocation(request)
            connector.stage_lookup(request)
            result = allocate(request, *args, **kwargs)
            if result is not None:
                connector.record_pages(request, manager)
            else:
                connector.staged = []
            return result

        manager.allocate_slots = tracked_allocate

    def schedule(self, *args, **kwargs):
        """Begin a new synchronous scheduler step."""
        if loads:
            raise RuntimeError(
                "Outstanding loads at next schedule; synchronous executor required"
            )
        self.connector.new_pages.clear()
        return super().schedule(*args, **kwargs)

    def update_from_output(self, scheduler_output, model_runner_output):
        """Finish the step and report changed storage counters."""
        result = super().update_from_output(scheduler_output, model_runner_output)
        connector = self.connector
        for request_id in list(connector.request_touched):
            if request_id not in self.requests:
                del connector.request_touched[request_id]
        counters = dict(
            written_pages=store.writes,
            read_pages=store.reads,
            restored_pages=connector.restored,
            replacements=store.replacements,
            resident_ssd_pages=len(store.slots),
            absent_offers=connector.absent,
            evictions=connector.evictions,
            free_retentions=connector.retained,
            unwritten_offers=connector.rejected,
            local_islands=connector.local_islands,
        )
        if counters != getattr(self, "_last_counters", None):
            emit(connector.serving_settings, "io", counters)
            self._last_counters = counters
        return result
