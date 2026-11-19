"""Native queue parity under randomized mutations; no serving-performance claim.

Assumes the pinned vLLM queue API. Compare every ordered block after every
mutation, including mixed-tier prepends, cache touches and multi-page frees.
"""
import random
import unittest
from indexed_queue import IndexedQueue
try:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue
except ImportError as error:
    raise unittest.SkipTest("Native queue checks need vLLM") from error


class Checks(unittest.TestCase):
    def test_native_order_and_counts_survive_all_mutation_paths(self):
        rng = random.Random(17)
        ref_blocks = {i: KVCacheBlock(block_id=i) for i in range(1, 129)}
        new_blocks = {i: KVCacheBlock(block_id=i) for i in range(1, 129)}
        ref = FreeKVCacheBlockQueue(list(ref_blocks.values()))
        new = IndexedQueue(FreeKVCacheBlockQueue(list(new_blocks.values())), 91)
        held = set()
        for _ in range(3000):
            free = [b.block_id for b in ref.get_all_free_blocks()]
            mode = rng.choice(['pop', 'pop_n', 'remove'] if not held else
                              ['append', 'append_n', 'prepend'] if not free else
                              ['pop', 'pop_n', 'remove', 'append', 'append_n', 'prepend'])
            if mode == 'pop':
                a, b = ref.popleft(), new.popleft()
                self.assertEqual(a.block_id, b.block_id)
                held.add(a.block_id)
            elif mode == 'pop_n':
                n = rng.randrange(min(9, len(free)) + 1)
                a, b = ref.popleft_n(n), new.popleft_n(n)
                self.assertEqual([x.block_id for x in a], [x.block_id for x in b])
                held.update(x.block_id for x in a)
            elif mode == 'remove':
                i = rng.choice(free)
                ref.remove(ref_blocks[i]); new.remove(new_blocks[i]); held.add(i)
            else:
                ids = rng.sample(sorted(held), 1 if mode == 'append' else rng.randrange(min(9, len(held)) + 1))
                a, b = [ref_blocks[i] for i in ids], [new_blocks[i] for i in ids]
                if mode == 'append': ref.append(a[0]); new.append(b[0])
                elif mode == 'append_n': ref.append_n(a); new.append_n(b)
                else: ref.prepend_n(a); new.prepend_n(b)
                held.difference_update(ids)
            self.assertTrue(new.assert_consistent())
            expected = [b.block_id for b in ref.get_all_free_blocks()]
            self.assertEqual(expected, [b.block_id for b in new.get_all_free_blocks()])
            self.assertEqual(list(new.hbm), [i for i in expected if i <= 91])
            self.assertEqual(list(new.hbf), [i for i in expected if i > 91])
            self.assertEqual([b.block_id for b in new.first_hbm(4)], list(new.hbm)[:4])


if __name__ == '__main__':
    unittest.main()
