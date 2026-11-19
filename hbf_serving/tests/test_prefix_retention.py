"""Check shared-prefix reference counts using small fake cache blocks."""

from types import SimpleNamespace as NS
import unittest
from prefix_retention import PrefixRetention, completed_shared_candidates
from tier_pool import TierPool


class Queue:
    def __init__(self, blocks):
        self.values = list(blocks)

    def get_all_free_blocks(self):
        return list(self.values)

    def remove(self, b):
        self.values.remove(b)

    def prepend_n(self, bs):
        self.values = list(bs) + self.values

    def append_n(self, bs):
        self.values.extend(bs)


class Pool:
    def __init__(self):
        self.blocks = [
            NS(block_id=i, ref_cnt=0, block_hash=None, is_null=i == 0) for i in range(6)
        ]
        self.null_block = self.blocks[0]
        self.enable_caching = True
        self.metrics_collector = None
        self.free_block_queue = Queue(self.blocks[1:])
        self.get_num_free_blocks = lambda: len(
            self.free_block_queue.get_all_free_blocks()
        )
        self.get_new_blocks = lambda n: None

    def _maybe_evict_cached_block(self, b):
        b.block_hash = None

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0:
                self.free_block_queue.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free_block_queue.append_n([b])


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.pool = Pool()
        self.manager = NS(req_to_blocks={})
        self.tiers = TierPool(self.pool, [self.manager], 5, 0)
        self.retention = PrefixRetention(self.pool, 5, 2)

    def page(self, key):
        b = self.pool.get_new_blocks(1)[0]
        b.block_hash = key
        return b

    def test_pin_survives_request_release_and_is_charged_to_hbm(self):
        b = self.page("prefix")
        self.retention.retain([(b, "prefix")], gpu_completed=True)
        self.pool.free_blocks([b])
        self.assertEqual(b.ref_cnt, 1)
        self.assertEqual(self.pool.get_num_free_blocks(), 4)
        self.assertEqual(self.manager._get_num_evictable_blocks([b]), 0)
        with self.assertRaises(ValueError):
            self.pool.get_new_blocks(5)
        self.retention.verify()

    def test_pin_after_request_finished_removes_free_page(self):
        b = self.page("prefix")
        self.pool.free_blocks([b])
        self.assertEqual(self.pool.get_num_free_blocks(), 5)
        self.retention.retain([(b, "prefix")], gpu_completed=True)
        self.assertEqual(self.pool.get_num_free_blocks(), 4)
        self.retention.verify()

    def test_repeated_hits_do_not_leak_retention_references(self):
        b = self.page("prefix")
        self.retention.retain([(b, "prefix")], gpu_completed=True)
        self.pool.free_blocks([b])
        for _ in range(100):
            self.pool.touch([b])
            self.retention.retain([(b, "prefix")], gpu_completed=True)
            self.pool.free_blocks([b])
        self.assertEqual(b.ref_cnt, 1)
        self.assertEqual(len(self.retention.by_id), 1)

    def test_suffix_remains_reclaimable_and_prefix_eviction_raises(self):
        prefix = self.page("prefix")
        suffix = self.page("suffix")
        self.retention.retain([(prefix, "prefix")], gpu_completed=True)
        self.pool.free_blocks([prefix, suffix])
        self.pool._maybe_evict_cached_block(suffix)
        self.assertIsNone(suffix.block_hash)
        with self.assertRaises(RuntimeError):
            self.pool._maybe_evict_cached_block(prefix)
        self.assertEqual(prefix.block_hash, "prefix")

    def test_no_early_pin_and_corrupted_hash_detected(self):
        b = self.page("prefix")
        with self.assertRaises(ValueError):
            self.retention.retain([(b, "prefix")], gpu_completed=False)
        self.assertEqual(b.ref_cnt, 1)
        self.retention.retain([(b, "prefix")], gpu_completed=True)
        b.block_hash = "corrupted"
        with self.assertRaises(RuntimeError):
            self.retention.verify()

    def test_partial_computation_and_shared_boundary_are_respected(self):
        blocks = [NS(block_hash=str(i)) for i in range(5)]
        request = NS(num_computed_tokens=31)
        candidates, complete = completed_shared_candidates(
            ["r"], {"r": request}, {"r": blocks}, set(), 2
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(complete, [])
        request.num_computed_tokens = 80
        candidates, complete = completed_shared_candidates(
            ["r"], {"r": request}, {"r": blocks}, set(), 2
        )
        self.assertEqual([x[0] for x in candidates], blocks[:2])
        self.assertEqual(complete, ["r"])
        self.assertEqual(
            completed_shared_candidates(["r"], {"r": request}, {"r": blocks}, {"r"}, 2),
            ([], []),
        )

    def test_duplicate_physical_copy_of_hash_is_not_double_pinned(self):
        a = self.page("same")
        b = self.page("same")
        self.retention.retain([(a, "same"), (b, "same")], gpu_completed=True)
        self.assertEqual(a.ref_cnt, 2)
        self.assertEqual(b.ref_cnt, 1)
        self.assertEqual(len(self.retention.by_id), 1)


if __name__ == "__main__":
    unittest.main()
