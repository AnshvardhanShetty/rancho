"""Maintain request order and membership for the synchronous scheduler."""

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
