"""HBM-only baseline retaining the declared shared prefix after GPU computation.

Assumptions: unchanged synchronous engine and native queue order, no speculation,
no HBF pages. Native admission queues requests when retained pages leave too
little space. The retained references are genuine ownership charged to HBM,
not additional active requests. Reported ref-count-based HBM occupancy therefore
includes idle retained prefixes. Empty starting cache, all cold fills and pinning
overhead are timed. Never retains per-request unique suffixes or output tokens.
"""
from direct_backend import DirectScheduler, log_event
from concurrency_observer import Observer
from prefix_retention import PrefixRetention, completed_shared_candidates


class RetainedPrefixScheduler(DirectScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.direct_plan
        assert cfg['baseline_prefix_retention'] is True and cfg['HBF_pages'] == 0
        assert cfg['shared_prefix_tokens'] % cfg['block_tokens'] == 0
        self.shared_pages = cfg['shared_prefix_tokens']//cfg['block_tokens']
        self.retention = PrefixRetention(self.tiers.pool, cfg['HBM_pages'], self.shared_pages*cfg['shared_groups'])
        self.retention_seen = set()
        self.retention_steps = 0
        self.retention_completed = False
        log_event(cfg,'retention.jsonl',dict(event='RETENTION_START',retained_pages=0,
            expected_shared_pages=self.retention.expected_pages,empty_measured_cache=True,
            retention_rule='declared shared prefix only, after actual GPU computation',
            artificial_request_cap=False))

    def update_from_output(self, scheduler_output, model_runner_output):
        # This callback is entered after synchronous GPU execution. Save block
        # objects before the native completion path frees finished request tables.
        tables = self.tiers.managers[0].req_to_blocks
        candidates, complete_prefixes = completed_shared_candidates(
            scheduler_output.num_scheduled_tokens, self.requests, tables,
            self.retention_seen, self.shared_pages)
        result = super().update_from_output(scheduler_output, model_runner_output)
        added = self.retention.retain(candidates, gpu_completed=True)
        self.retention_seen.update(complete_prefixes)
        if added:
            invariant = self.retention.verify()
            log_event(self.direct_plan,'retention.jsonl',dict(event='PREFIX_PAGES_RETAINED',
                step=self.retention_steps, gpu_completed=True, added=added,
                HBM_free_pages=self.tiers.hbm_free_count(),
                resident_requests=len(self.running), **invariant))
        # Empty engine queues can occur between rate-paced arrivals. Only finish
        # validation after every measured request has computed its shared prefix
        # and all active work has completed; never reset or release retained KV.
        all_requests_seen = len(self.retention_seen) == self.direct_plan['n_requests']
        if all_requests_seen and self.get_num_unfinished_requests() == 0 and not self.retention_completed:
            invariant = self.retention.verify()
            assert invariant['retained_pages'] == invariant['expected_shared_pages']
            log_event(self.direct_plan,'retention.jsonl',dict(event='RETENTION_COMPLETE',
                step=self.retention_steps, **invariant))
            self.retention_completed = True
        self.retention_steps += 1
        return result


class RetainedBaselineScheduler(Observer, RetainedPrefixScheduler):
    pass
