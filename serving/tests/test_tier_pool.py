"""Allocator safety tests; synthetic pages establish no GPU performance claim."""
from types import SimpleNamespace as NS
import unittest
from tier_pool import TierPool
from reuse import existing_hbf,commit_existing


class Queue:
    def __init__(self, values): self.values=list(values)
    def get_all_free_blocks(self): return list(self.values)
    def remove(self, value): self.values.remove(value)
    def prepend_n(self, values): self.values=list(values)+self.values


class Pool:
    def __init__(self):
        self.blocks=[NS(block_id=i, ref_cnt=0, block_hash=None, is_null=i==0) for i in range(7)]
        self.null_block=self.blocks[0]
        self.enable_caching=True
        self.metrics_collector=None
        self.free_block_queue=Queue(self.blocks[1:])
        self.get_num_free_blocks=lambda:len(self.free_block_queue.values)
        self.get_new_blocks=lambda n:None
    def _maybe_evict_cached_block(self,b): b.block_hash=None
    def move_block_hashes(self,a,b): b.block_hash,a.block_hash=a.block_hash,None


class Tests(unittest.TestCase):
    def setUp(self):
        self.pool=Pool()
        self.manager=NS(req_to_blocks={})
        self.tiers=TierPool(self.pool,[self.manager],4,2)
    def active_page(self):
        b=self.pool.get_new_blocks(1)[0]
        b.block_hash=b'prefix'
        self.manager.req_to_blocks['a']=[b]
        return b
    def test_new_computation_never_allocates_hbf(self):
        self.assertEqual([b.block_id for b in self.pool.get_new_blocks(4)],[1,2,3,4])
        self.assertEqual(self.pool.get_num_free_blocks(),0)
        with self.assertRaises(ValueError):self.pool.get_new_blocks(1)
        self.assertEqual(self.tiers.available_hbf(),[5,6])
    def test_copy_migrates_all_active_readers_and_frees_source(self):
        old=self.active_page()
        old.ref_cnt+=1
        self.manager.req_to_blocks['b']=[old]
        before=self.pool.get_num_free_blocks()
        self.tiers.commit_completed_copies([(old.block_id,5)],gpu_completed=True)
        self.assertEqual(self.pool.get_num_free_blocks(),before+1)
        self.assertEqual(self.pool.blocks[5].ref_cnt,2)
        self.assertEqual(old.ref_cnt,0)
        self.assertIs(self.manager.req_to_blocks['a'][0],self.pool.blocks[5])
        self.assertIs(self.manager.req_to_blocks['b'][0],self.pool.blocks[5])
        self.assertEqual(self.tiers.available_hbf(),[6])
    def test_no_publication_before_copy_completion(self):
        old=self.active_page()
        with self.assertRaises(ValueError):
            self.tiers.commit_completed_copies([(old.block_id,5)],gpu_completed=False)
        self.assertEqual(old.block_hash,b'prefix')
        self.assertIsNone(self.pool.blocks[5].block_hash)
    def test_active_hbf_victim_rejected(self):
        old=self.active_page()
        self.pool.free_block_queue.remove(self.pool.blocks[5])
        self.pool.blocks[5].ref_cnt=1
        with self.assertRaises(ValueError):
            self.tiers.commit_completed_copies([(old.block_id,5)],gpu_completed=True)
        self.assertEqual(old.block_hash,b'prefix')
    def test_whole_transaction_checked_before_first_publication(self):
        old=self.active_page()
        with self.assertRaises(ValueError):
            self.tiers.commit_completed_copies([(old.block_id,5),(2,6)],gpu_completed=True)
        self.assertEqual(old.block_hash,b'prefix')
        self.assertIsNone(self.pool.blocks[5].block_hash)
    def test_worker_receives_full_replacement_table(self):
        old=self.active_page()
        self.tiers.commit_completed_copies([(old.block_id,5)],gpu_completed=True)
        cached=NS(req_ids=['a'],new_block_ids=[None],resumed_req_ids=set())
        output=NS(scheduled_cached_reqs=cached,scheduled_new_reqs=[])
        self.tiers.refresh_worker_tables(output)
        self.assertEqual(cached.new_block_ids,[([5],)])
        self.assertEqual(cached.resumed_req_ids,{'a'})
    def test_hbf_hit_does_not_claim_evictable_hbm(self):
        self.assertEqual(self.manager._get_num_evictable_blocks([self.pool.blocks[5]]),0)
        self.assertEqual(self.manager._get_num_evictable_blocks([self.pool.blocks[1]]),1)
    def test_existing_copy_can_be_reused_while_another_reader_is_active(self):
        old=self.active_page();target=self.pool.blocks[5]
        target.block_hash=old.block_hash;target.ref_cnt=1
        self.pool.free_block_queue.remove(target)
        self.manager.req_to_blocks['other']=[target]
        self.pool.cached_block_hash_to_block=NS(_cache={old.block_hash:{1:old,5:target}})
        self.assertEqual(existing_hbf(self.tiers,old.block_id),5)
        result=commit_existing(self.tiers,[(old.block_id,5)],gpu_completed=True)
        self.assertEqual(result['additional_writes'],0)
        self.assertEqual(target.ref_cnt,2)
        self.assertEqual(old.ref_cnt,0)
        self.assertIs(self.manager.req_to_blocks['a'][0],target)
        self.assertIs(self.manager.req_to_blocks['other'][0],target)
    def test_existing_copy_identity_mismatch_is_rejected(self):
        old=self.active_page();self.pool.blocks[5].block_hash=b'different'
        with self.assertRaises(ValueError):commit_existing(self.tiers,[(old.block_id,5)],gpu_completed=True)
        self.assertEqual(old.ref_cnt,1)


if __name__=='__main__': unittest.main()
