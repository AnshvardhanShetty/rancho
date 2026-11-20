"""Reuse existing HBF copies and transfer their active-reader references.

Copies are identified by vLLM prefix hashes and complete 16-token pages.
An existing immutable copy requires no further write. Reference updates use
the single synchronous scheduler."""


def find_existing_hbf_copy(tiers, source_id):
    """Find a saved HBF copy of this page if there is one"""
    source = tiers.pool.blocks[source_id]
    if source.block_hash is None:
        return None
    entry = tiers.pool.cached_block_hash_to_block._cache.get(source.block_hash)
    values = (
        list(entry.values())
        if isinstance(entry, dict)
        else ([] if entry is None else [entry])
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
    if len(set(source_page_id for source_page_id, target_page_id in pairs)) != len(
        pairs
    ):
        raise ValueError("Duplicate reuse source")
    pool = tiers.pool
    tables = tiers.managers[0].req_to_blocks
    references = {}
    for source_page_id, target_page_id in pairs:
        if not tiers.is_hbm_page(source_page_id) or not tiers.is_hbf_page(
            target_page_id
        ):
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
    return dict(
        reused_HBF_pages=len(pairs), freed_HBM_pages=len(pairs), additional_writes=0
    )
