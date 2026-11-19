"""Manage HBM and HBF block allocation across vLLM cache-manager versions.

New computation uses HBM. Completed HBF copies remain directly readable.
Block 0 is reserved; the remaining blocks are split at the supplied HBM limit.
The older interface is handled by legacy_tier_pool.py."""

from indexed_queue import IndexedQueue


class TierPool:
    def __new__(cls, *args, **kwargs):
        if __import__("os").environ.get("RANCHO_DIAGNOSTIC_QUEUE") == "legacy":
            from legacy_tier_pool import TierPool as Legacy

            return Legacy(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, pool, managers, hbm_pages, hbf_pages):
        if type(hbm_pages) is not int or hbm_pages <= 0:
            raise ValueError("Explicit positive HBM page count required")
        if type(hbf_pages) is not int or hbf_pages < 0:
            raise ValueError("Explicit nonnegative HBF page count required")
        if len(managers) != 1 or len(pool.blocks) != 1 + hbm_pages + hbf_pages:
            raise ValueError("Requires exact one-group allocation including null page")
        if pool.null_block.block_id != 0 or not pool.enable_caching:
            raise ValueError("Requires native null page 0 and prefix caching")
        self.pool, self.managers = pool, managers
        self.hbm_pages, self.hbf_pages = hbm_pages, hbf_pages
        self.dirty_requests = set()
        pool.free_block_queue = IndexedQueue(pool.free_block_queue, hbm_pages)
        self.original_new = pool.get_new_blocks
        self.original_free_count = pool.get_num_free_blocks
        pool.get_num_free_blocks = self.hbm_free_count
        pool.get_new_blocks = self.allocate_hbm
        pool.get_usage = self.active_usage
        # Native admission subtracts reusable cache-hit pages from the required
        # free budget. Only HBM hits consume pages in that HBM free budget.
        for manager in managers:
            manager._get_num_evictable_blocks = lambda blocks: sum(
                b.ref_cnt == 0 and self.is_hbm(b.block_id) and not b.is_null
                for b in blocks
            )

    def is_hbm(self, block_id):
        return 0 < block_id <= self.hbm_pages

    def is_hbf(self, block_id):
        return self.hbm_pages < block_id < len(self.pool.blocks)

    def free_order(self):
        return list(self.pool.free_block_queue.get_all_free_blocks())

    def hbm_free_count(self):
        return len(self.pool.free_block_queue.hbm)

    def active_usage(self):
        # Native usage describes referenced pages, not writable capacity.
        return 1 - (
            len(self.pool.free_block_queue.hbm) + len(self.pool.free_block_queue.hbf)
        ) / (self.hbm_pages + self.hbf_pages)

    def allocate_hbm(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("Invalid allocation count")
        selected = self.pool.free_block_queue.first_hbm(count)
        if len(selected) != count:
            raise ValueError("Insufficient free HBM; HBF cannot absorb new writes")
        for block in selected:
            self.pool.free_block_queue.remove(block)
            self.pool._maybe_evict_cached_block(block)
            if block.ref_cnt != 0:
                raise ValueError("Free queue contains a referenced HBM page")
            block.ref_cnt = 1
            if self.pool.metrics_collector:
                self.pool.metrics_collector.on_block_allocated(block)
        return selected

    def available_hbf(self):
        # Native free-queue order supplies the same retention eviction order;
        # active-reader pages are absent from this queue and cannot be replaced.
        return list(self.pool.free_block_queue.hbf)

    def commit_completed_copies(self, pairs, *, gpu_completed):
        if gpu_completed is not True:
            raise ValueError("Cannot publish a pending GPU copy")
        pairs = list(pairs)
        sources, destinations = [a for a, b in pairs], [b for a, b in pairs]
        if len(set(sources)) != len(sources) or len(set(destinations)) != len(
            destinations
        ):
            raise ValueError("Duplicate source/destination in one atomic migration")
        free_ids = {b.block_id for b in self.free_order()}
        tables = self.managers[0].req_to_blocks
        references = {}
        # Validate the entire transaction before changing a hash or a table.
        for src, dst in pairs:
            if not self.is_hbm(src) or not self.is_hbf(dst):
                raise ValueError("Migration must be from HBM to HBF")
            old, new = self.pool.blocks[src], self.pool.blocks[dst]
            if dst not in free_ids or new.ref_cnt != 0:
                raise ValueError("HBF destination has an active reader")
            refs = [
                (rid, i)
                for rid, blocks in tables.items()
                for i, block in enumerate(blocks)
                if block is old
            ]
            if not refs or old.ref_cnt != len(refs) or old.block_hash is None:
                raise ValueError(
                    "Source references/hash do not match immutable prefix state"
                )
            references[src] = refs
        for src, dst in pairs:
            old, new = self.pool.blocks[src], self.pool.blocks[dst]
            self.pool.free_block_queue.remove(new)
            self.pool._maybe_evict_cached_block(new)
            self.pool.move_block_hashes(old, new)
            refs = references[src]
            for rid, at in refs:
                tables[rid][at] = new
                self.dirty_requests.add(rid)
            new.ref_cnt, old.ref_cnt = len(refs), 0
            self.pool.free_block_queue.prepend_n([old])
        return dict(
            migrated_pages=len(pairs),
            freed_HBM_pages=len(pairs),
            requests_requiring_table_refresh=sorted(self.dirty_requests),
        )

    def refresh_worker_tables(self, output):
        cached = output.scheduled_cached_reqs
        tables = self.managers[0].req_to_blocks
        for i, rid in enumerate(cached.req_ids):
            if rid in self.dirty_requests:
                cached.new_block_ids[i] = ([b.block_id for b in tables[rid]],)
                cached.resumed_req_ids.add(rid)
                self.dirty_requests.remove(rid)
        for req in output.scheduled_new_reqs:
            self.dirty_requests.discard(req.req_id)
