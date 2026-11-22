"""Save full KV blocks to SSD before vLLM reuses their GPU slots

The scheduler tracks fully computed blocks and saves them within the write budget
The connector stages cache hits before allocation can reuse their old slots
Saved blocks are copied back before the model reads them
"""

import json
import os
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


def read_serving_settings():
    """Read the SSD demo settings supplied by the launcher."""
    settings_file = os.environ["RANCHO_SSD_SERVING_SETTINGS"]
    return json.loads(Path(settings_file).read_text())


def write_serving_event(serving_settings, name, fields):
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
            self.resident_block_keys = {}
            self.evictions = self.absent = self.retained = self.rejected = self.restored = (
                self.local_islands
            ) = 0
            self.new_pages = set()

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
        write_serving_event(
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

    def track_resident_blocks(self, request, manager):
        """Track each cached prompt block and protect slots allocated this step."""
        blocks = manager.get_blocks(request.request_id).blocks[0]
        for block, block_hash in zip(
            blocks, request.block_hashes[: len(request.prompt_token_ids) // 16]
        ):
            key = bytes(block_hash)
            if block.block_hash is None or bytes(get_block_hash(block.block_hash)) != key:
                continue
            page_id = block.block_id
            if page_id not in self.resident_block_keys:
                self.resident_block_keys[page_id] = key
                self.new_pages.add(page_id)
            if self.resident_block_keys[page_id] != key:
                raise RuntimeError("Unobserved HBM reuse")

    def handle_hbm_eviction(self, block):
        """Save an unused GPU block if SSD writes are enabled and affordable."""
        key = self.resident_block_keys.pop(block.block_id, None)
        if key is None:
            return
        assert (
            block.ref_cnt == 0
            and block.block_hash
            and get_group_id(block.block_hash) == 0
            and get_block_hash(block.block_hash) == key
        )
        if block.block_id in self.new_pages:
            raise RuntimeError("Attempted to evict unfinished KV in the same scheduler step")
        self.evictions += 1
        if key in store.slots:
            store.slots.move_to_end(key)
            self.retained += 1
            return
        self.absent += 1
        if (
            self.serving_settings["save_to_ssd"]
            and store.capacity
            and store.writes < self.serving_settings["write_budget_pages"]
        ):
            store.write_block_to_ssd(key, block.block_id)
        else:
            self.rejected += 1

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Find how much more of the input is available from GPU or SSD."""
        begin = num_computed_tokens // 16
        matches = []
        for block_hash in request.block_hashes[begin : (len(request.prompt_token_ids) - 1) // 16]:
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

    def stage_cached_blocks_before_allocation(self, request):
        """Preserve needed KV before allocation can recycle its source."""
        staged = []
        for block_key, local_hbm_page in self.pending.get(request.request_id, (0, []))[1]:
            if local_hbm_page is None:
                payload = store.read_block_from_ssd(block_key)
            else:
                payload = store.buffer.new_empty(store.page_bytes)
                store.copy_block_between_gpu_and_cpu(local_hbm_page, payload, True)
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
            store.restore_block_to_gpu(page, loads.pop(page))

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
            if block.block_id in connector.resident_block_keys:
                connector.handle_hbm_eviction(block)
            return evict(block)

        connector.pool._maybe_evict_cached_block = before_reuse
        allocate = manager.allocate_slots

        def tracked_allocate(request, *args, **kwargs):
            connector.stage_cached_blocks_before_allocation(request)
            result = allocate(request, *args, **kwargs)
            if result is not None:
                connector.track_resident_blocks(request, manager)
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
            write_serving_event(connector.serving_settings, "io", counters)
            self._last_counters = counters
        return result
