"""Index the native eviction queue by tier without changing its ordering.

Assumptions: one synchronous scheduler; all mutations use the pinned queue API.
The native linked list remains authoritative. Ordered indexes mirror each tier's
subsequence and provide constant-time counts. No capacity or admission changes.
"""

from collections import OrderedDict
from itertools import islice


class IndexedQueue:
    def __init__(self, native, hbm_pages):
        self.native = native
        self.hbm_pages = hbm_pages
        self.hbm = OrderedDict()
        self.hbf = OrderedDict()
        for block in native.get_all_free_blocks():
            self._index(block)[block.block_id] = block

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _index(self, block):
        if block.block_id <= 0:
            raise ValueError("Null/sentinel page cannot enter the free tier index")
        return self.hbm if block.block_id <= self.hbm_pages else self.hbf

    def _check_new(self, blocks):
        ids = [b.block_id for b in blocks]
        if len(ids) != len(set(ids)) or any(
            b.block_id in self._index(b) for b in blocks
        ):
            raise ValueError("Duplicate free-page insertion")

    def popleft(self):
        block = self.native.popleft()
        del self._index(block)[block.block_id]
        return block

    def popleft_n(self, n):
        blocks = self.native.popleft_n(n)
        for block in blocks:
            del self._index(block)[block.block_id]
        return blocks

    def remove(self, block):
        index = self._index(block)
        if index.get(block.block_id) is not block:
            raise ValueError("Removal of an unindexed free page")
        self.native.remove(block)
        del index[block.block_id]

    def append(self, block):
        self._check_new([block])
        self.native.append(block)
        self._index(block)[block.block_id] = block

    def append_n(self, blocks):
        blocks = list(blocks)
        self._check_new(blocks)
        self.native.append_n(blocks)
        for block in blocks:
            self._index(block)[block.block_id] = block

    def prepend_n(self, blocks):
        blocks = list(blocks)
        self._check_new(blocks)
        self.native.prepend_n(blocks)
        for block in reversed(blocks):
            index = self._index(block)
            index[block.block_id] = block
            index.move_to_end(block.block_id, last=False)

    def first_hbm(self, count):
        return list(islice(self.hbm.values(), count))

    def assert_consistent(self):
        native = self.native.get_all_free_blocks()
        for index, predicate in [
            (self.hbm, lambda b: b.block_id <= self.hbm_pages),
            (self.hbf, lambda b: b.block_id > self.hbm_pages),
        ]:
            expected = [b.block_id for b in native if predicate(b)]
            if list(index) != expected:
                raise ValueError("Native eviction order and tier index diverged")
        return True
