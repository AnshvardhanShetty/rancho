"""Protect computed shared prefixes in the HBM-only baseline.

Each retained full page gets an extra reference and remains charged to HBM.
The caller supplies the shared-prefix length and waits for GPU computation
before retaining pages. This is the benchmark's prefix-retaining baseline."""


def completed_shared_candidates(scheduled_ids, requests, tables, seen, shared_pages):
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
                raise RuntimeError(
                    "Attempted eviction of a retained shared-prefix page"
                )
            return original(block)

        pool._maybe_evict_cached_block = guarded_evict

    def retain(self, candidates, *, gpu_completed):
        """Candidates are (physical block, hash captured after scheduling) pairs."""
        if gpu_completed is not True:
            raise ValueError("Cannot retain uncomputed KV")
        added = []
        for block, expected_hash in candidates:
            if block.is_null or not 0 < block.block_id <= self.hbm_pages:
                raise ValueError("Retention requires a real HBM page")
            if expected_hash is None or block.block_hash != expected_hash:
                raise ValueError(
                    "Shared-prefix hash missing or changed before retention"
                )
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

    def verify(self):
        """Full invariant check at new pin events and completion, not every token."""
        free_ids = {
            b.block_id for b in self.pool.free_block_queue.get_all_free_blocks()
        }
        for expected_hash, block in self.by_hash.items():
            if (
                block.block_hash != expected_hash
                or block.ref_cnt < 1
                or block.block_id in free_ids
            ):
                raise RuntimeError(
                    "Retained prefix was changed, released or made evictable"
                )
        return dict(
            retained_pages=len(self.by_id),
            expected_shared_pages=self.expected_pages,
            eviction_attempts=self.eviction_attempts,
            protected_pages_in_free_queue=0,
        )
