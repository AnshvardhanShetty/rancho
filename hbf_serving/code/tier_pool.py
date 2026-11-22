"""Manage HBM and HBF blocks in vLLM.

New computation uses HBM. Completed HBF copies remain directly readable.
Block 0 is reserved; the remaining blocks are split at the supplied HBM limit."""

from collections import OrderedDict
from itertools import islice


class IndexedQueue:
    """Keep requests in order and find them by ID"""

    def __init__(self, native, hbm_pages):
        self.native = native
        self.hbm_pages = hbm_pages
        self.hbm = OrderedDict()
        self.hbf = OrderedDict()
        for block in native.get_all_free_blocks():
            self._index_for_block_tier(block)[block.block_id] = block

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _index_for_block_tier(self, block):
        if block.block_id <= 0:
            raise ValueError("Null/sentinel page cannot enter the free tier index")
        return self.hbm if block.block_id <= self.hbm_pages else self.hbf

    def _check_blocks_are_absent(self, blocks):
        ids = [block.block_id for block in blocks]
        if len(ids) != len(set(ids)) or any(
            block.block_id in self._index_for_block_tier(block) for block in blocks
        ):
            raise ValueError("Duplicate free-page insertion")

    def popleft(self):
        block = self.native.popleft()
        del self._index_for_block_tier(block)[block.block_id]
        return block

    def popleft_n(self, n):
        blocks = self.native.popleft_n(n)
        for block in blocks:
            del self._index_for_block_tier(block)[block.block_id]
        return blocks

    def remove(self, block):
        index = self._index_for_block_tier(block)
        if index.get(block.block_id) is not block:
            raise ValueError("Removal of an unindexed free page")
        self.native.remove(block)
        del index[block.block_id]

    def append(self, block):
        self._check_blocks_are_absent([block])
        self.native.append(block)
        self._index_for_block_tier(block)[block.block_id] = block

    def append_n(self, blocks):
        blocks = list(blocks)
        self._check_blocks_are_absent(blocks)
        self.native.append_n(blocks)
        for block in blocks:
            self._index_for_block_tier(block)[block.block_id] = block

    def prepend_n(self, blocks):
        blocks = list(blocks)
        self._check_blocks_are_absent(blocks)
        self.native.prepend_n(blocks)
        for block in reversed(blocks):
            index = self._index_for_block_tier(block)
            index[block.block_id] = block
            index.move_to_end(block.block_id, last=False)

    def first_free_hbm_blocks(self, count):
        return list(islice(self.hbm.values(), count))

    def assert_consistent(self):
        """Check that the queue and the request index agree"""
        native = self.native.get_all_free_blocks()
        for index, predicate in [
            (self.hbm, lambda block: block.block_id <= self.hbm_pages),
            (self.hbf, lambda block: block.block_id > self.hbm_pages),
        ]:
            expected = [block.block_id for block in native if predicate(block)]
            if list(index) != expected:
                raise ValueError("Native eviction order and tier index diverged")
        return True


class TierPool:
    """Give vLLM separate sets of page IDs for HBM and HBF"""

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
        pool.get_new_blocks = self.allocate_hbm_pages
        pool.get_usage = self.hbm_usage_fraction
        # Native admission subtracts reusable cache-hit pages from the required
        # free budget. Only HBM hits consume pages in that HBM free budget.
        for manager in managers:
            manager._get_num_evictable_blocks = lambda blocks: sum(
                block.ref_cnt == 0 and self.is_hbm_page(block.block_id) and not block.is_null
                for block in blocks
            )

    def is_hbm_page(self, block_id):
        return 0 < block_id <= self.hbm_pages

    def is_hbf_page(self, block_id):
        return self.hbm_pages < block_id < len(self.pool.blocks)

    def free_pages_in_queue_order(self):
        return list(self.pool.free_block_queue.get_all_free_blocks())

    def hbm_free_count(self):
        return len(self.pool.free_block_queue.hbm)

    def hbm_usage_fraction(self):
        # Native usage describes referenced pages, not writable capacity.
        return 1 - (len(self.pool.free_block_queue.hbm) + len(self.pool.free_block_queue.hbf)) / (
            self.hbm_pages + self.hbf_pages
        )

    def allocate_hbm_pages(self, count):
        """Keep new allocations in the HBM page range"""
        if type(count) is not int or count < 0:
            raise ValueError("Invalid allocation count")
        selected = self.pool.free_block_queue.first_free_hbm_blocks(count)
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

    def free_hbf_page_ids(self):
        # Native free-queue order supplies the same retention eviction order;
        # active-reader pages are absent from this queue and cannot be replaced.
        return list(self.pool.free_block_queue.hbf)

    def commit_completed_copies(self, pairs, *, gpu_completed):
        """Apply copies to the current cache tables"""
        if gpu_completed is not True:
            raise ValueError("Cannot publish a pending GPU copy")
        pairs = list(pairs)
        sources, destinations = [a for a, block in pairs], [block for a, block in pairs]
        if len(set(sources)) != len(sources) or len(set(destinations)) != len(destinations):
            raise ValueError("Duplicate source/destination in one atomic migration")
        free_ids = {block.block_id for block in self.free_pages_in_queue_order()}
        tables = self.managers[0].req_to_blocks
        references = {}
        # Validate the entire transaction before changing a hash or a table.
        for source_page_id, target_page_id in pairs:
            if not self.is_hbm_page(source_page_id) or not self.is_hbf_page(target_page_id):
                raise ValueError("Migration must be from HBM to HBF")
            source_block, target_block = (
                self.pool.blocks[source_page_id],
                self.pool.blocks[target_page_id],
            )
            if target_page_id not in free_ids or target_block.ref_cnt != 0:
                raise ValueError("HBF destination has an active reader")
            active_references = [
                (request_id, block_position)
                for request_id, blocks in tables.items()
                for block_position, block in enumerate(blocks)
                if block is source_block
            ]
            if (
                not active_references
                or source_block.ref_cnt != len(active_references)
                or source_block.block_hash is None
            ):
                raise ValueError("Source references/hash do not match immutable prefix state")
            references[source_page_id] = active_references
        for source_page_id, target_page_id in pairs:
            source_block, target_block = (
                self.pool.blocks[source_page_id],
                self.pool.blocks[target_page_id],
            )
            self.pool.free_block_queue.remove(target_block)
            self.pool._maybe_evict_cached_block(target_block)
            self.pool.move_block_hashes(source_block, target_block)
            active_references = references[source_page_id]
            for request_id, block_position in active_references:
                tables[request_id][block_position] = target_block
                self.dirty_requests.add(request_id)
            target_block.ref_cnt, source_block.ref_cnt = len(active_references), 0
            self.pool.free_block_queue.prepend_n([source_block])
        return dict(
            migrated_pages=len(pairs),
            freed_HBM_pages=len(pairs),
            requests_requiring_table_refresh=sorted(self.dirty_requests),
        )

    def refresh_worker_block_tables(self, output):
        """Tell the worker where the active pages are now"""
        cached = output.scheduled_cached_reqs
        tables = self.managers[0].req_to_blocks
        for request_index, request_id in enumerate(cached.req_ids):
            if request_id in self.dirty_requests:
                cached.new_block_ids[request_index] = (
                    [block.block_id for block in tables[request_id]],
                )
                cached.resumed_req_ids.add(request_id)
                self.dirty_requests.remove(request_id)
        for scheduled_request in output.scheduled_new_reqs:
            self.dirty_requests.discard(scheduled_request.req_id)


# Reuse an existing HBF page without another copy.
def find_existing_hbf_copy(tiers, source_id):
    """Find a saved HBF copy of this page if there is one"""
    source = tiers.pool.blocks[source_id]
    if source.block_hash is None:
        return None
    entry = tiers.pool.cached_block_hash_to_block._cache.get(source.block_hash)
    values = (
        list(entry.values()) if isinstance(entry, dict) else ([] if entry is None else [entry])
    )
    for block in values:
        if tiers.is_hbf_page(block.block_id):
            return block.block_id
    return None


def remap_to_existing_hbf_copies(tiers, pairs, *, gpu_completed):
    """Move active references to an HBF copy we already have"""
    if gpu_completed is not True:
        raise ValueError("Prefill GPU work must complete before remapping")
    pairs = list(pairs)
    if len(set(source_page_id for source_page_id, target_page_id in pairs)) != len(pairs):
        raise ValueError("Duplicate reuse source")
    pool = tiers.pool
    tables = tiers.managers[0].req_to_blocks
    references = {}
    for source_page_id, target_page_id in pairs:
        if not tiers.is_hbm_page(source_page_id) or not tiers.is_hbf_page(target_page_id):
            raise ValueError("Wrong reuse tier")
        source_block, target_block = (
            pool.blocks[source_page_id],
            pool.blocks[target_page_id],
        )
        active_references = [
            (request_id, block_position)
            for request_id, blocks in tables.items()
            for block_position, block in enumerate(blocks)
            if block is source_block
        ]
        if (
            source_block.block_hash is None
            or source_block.block_hash != target_block.block_hash
            or not active_references
            or source_block.ref_cnt != len(active_references)
        ):
            raise ValueError("Existing-copy identity or source references changed")
        references[source_page_id] = active_references
    for source_page_id, target_page_id in pairs:
        source_block, target_block = (
            pool.blocks[source_page_id],
            pool.blocks[target_page_id],
        )
        if target_block.ref_cnt == 0:
            pool.free_block_queue.remove(target_block)
        pool._maybe_evict_cached_block(source_block)
        active_references = references[source_page_id]
        for request_id, block_position in active_references:
            tables[request_id][block_position] = target_block
            tiers.dirty_requests.add(request_id)
        target_block.ref_cnt += len(active_references)
        source_block.ref_cnt = 0
        pool.free_block_queue.prepend_n([source_block])
    return dict(reused_HBF_pages=len(pairs), freed_HBM_pages=len(pairs), additional_writes=0)


# Keep the baseline's shared prompt pages in HBM.
def find_completed_shared_prefixes(scheduled_ids, requests, tables, seen, shared_pages):
    """Called only at synchronous GPU completion; exclude unique and output pages."""
    candidates = []
    complete = []
    for rid in scheduled_ids:
        if rid in seen:
            continue
        count = min(shared_pages, requests[rid].num_computed_tokens // 16)
        blocks = tables[rid]
        assert count <= len(blocks)
        for block in blocks[:count]:
            assert block.block_hash is not None
            candidates.append((block, block.block_hash))
        if count == shared_pages:
            complete.append(rid)
    return candidates, complete


class PrefixRetention:
    """Keep the shared prefixes in memory for the HBM comparison"""

    def __init__(self, pool, hbm_pages, expected_pages):
        self.pool = pool
        self.hbm_pages = hbm_pages
        self.expected_pages = expected_pages
        self.by_hash = {}
        self.by_id = {}
        self.eviction_attempts = 0
        original = pool._maybe_evict_cached_block

        def guarded_evict(block):
            if block.block_id in self.by_id:
                self.eviction_attempts += 1
                raise RuntimeError("Attempted eviction of a retained shared-prefix page")
            return original(block)

        pool._maybe_evict_cached_block = guarded_evict

    def retain_shared_prefix_blocks(self, candidates, *, gpu_completed):
        """Candidates are (physical block, hash captured after scheduling) pairs."""
        if gpu_completed is not True:
            raise ValueError("Cannot retain uncomputed KV")
        added = []
        for block, expected_hash in candidates:
            if block.is_null or not 0 < block.block_id <= self.hbm_pages:
                raise ValueError("Retention requires a real HBM page")
            if expected_hash is None or block.block_hash != expected_hash:
                raise ValueError("Shared-prefix hash missing or changed before retention")
            if expected_hash in self.by_hash:
                # One retained copy is enough even if computation produced duplicates.
                continue
            if block.block_id in self.by_id or len(self.by_id) >= self.expected_pages:
                raise ValueError("Unexpected duplicate page or excess shared prefix")
            self.pool.touch([block])
            self.by_hash[expected_hash] = block
            self.by_id[block.block_id] = expected_hash
            added.append(dict(block_id=block.block_id, block_hash=repr(expected_hash)))
        return added

    def verify_retained_prefixes(self):
        """Full invariant check at new pin events and completion, not every token."""
        free_ids = {block.block_id for block in self.pool.free_block_queue.get_all_free_blocks()}
        for expected_hash, block in self.by_hash.items():
            if (
                block.block_hash != expected_hash
                or block.ref_cnt < 1
                or block.block_id in free_ids
            ):
                raise RuntimeError("Retained prefix was changed, released or made evictable")
        return dict(
            retained_pages=len(self.by_id),
            expected_shared_pages=self.expected_pages,
            eviction_attempts=self.eviction_attempts,
            protected_pages_in_free_queue=0,
        )
