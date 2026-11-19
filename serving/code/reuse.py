"""Reuse immutable HBF copies without charging or performing another write.

Assumptions: the same pinned model and native prefix hash namespace, complete
16-token pages, synchronous single scheduler. Existing HBF payloads are never
overwritten. The native hash identity is the same correctness assumption used
by vLLM prefix caching. Active-reader references are transferred atomically.
No admission decision or scientific budget is chosen by this helper.
"""


def existing_hbf(tiers,source_id):
    source=tiers.pool.blocks[source_id]
    if source.block_hash is None:return None
    entry=tiers.pool.cached_block_hash_to_block._cache.get(source.block_hash)
    values=list(entry.values()) if isinstance(entry,dict) else ([] if entry is None else [entry])
    for block in values:
        if tiers.is_hbf(block.block_id):return block.block_id
    return None


def commit_existing(tiers,pairs,*,gpu_completed):
    if gpu_completed is not True:raise ValueError('Prefill GPU work must complete before remapping')
    pairs=list(pairs)
    if len(set(src for src,dst in pairs))!=len(pairs):raise ValueError('Duplicate reuse source')
    pool=tiers.pool;tables=tiers.managers[0].req_to_blocks;references={}
    for src,dst in pairs:
        if not tiers.is_hbm(src) or not tiers.is_hbf(dst):raise ValueError('Wrong reuse tier')
        old,new=pool.blocks[src],pool.blocks[dst]
        refs=[(rid,i) for rid,blocks in tables.items() for i,b in enumerate(blocks) if b is old]
        if old.block_hash is None or old.block_hash!=new.block_hash or not refs or old.ref_cnt!=len(refs):
            raise ValueError('Existing-copy identity or source references changed')
        references[src]=refs
    for src,dst in pairs:
        old,new=pool.blocks[src],pool.blocks[dst]
        if new.ref_cnt==0:pool.free_block_queue.remove(new)
        pool._maybe_evict_cached_block(old)
        refs=references[src]
        for rid,i in refs:tables[rid][i]=new;tiers.dirty_requests.add(rid)
        new.ref_cnt+=len(refs);old.ref_cnt=0
        pool.free_block_queue.prepend_n([old])
    return dict(reused_HBF_pages=len(pairs),freed_HBM_pages=len(pairs),additional_writes=0)
