"""Keep computed shared prefixes in HBM for the serving baseline.

Retained pages count towards HBM capacity and stay protected from replacement.
The scheduler queues requests when the remaining space cannot hold their KV.
Only the declared shared prefix is retained after GPU computation."""

from direct_backend import DirectScheduler, log_event
from concurrency_observer import Observer
from prefix_retention import PrefixRetention, completed_shared_candidates


class RetainedPrefixScheduler(DirectScheduler):
    """Keep shared prefixes in HBM once the model has computed them"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        engine_settings = self.engine_settings
        assert (
            engine_settings["baseline_prefix_retention"] is True
            and engine_settings["HBF_pages"] == 0
        )
        assert (
            engine_settings["shared_prefix_tokens"] % engine_settings["block_tokens"]
            == 0
        )
        self.shared_pages = (
            engine_settings["shared_prefix_tokens"] // engine_settings["block_tokens"]
        )
        self.retention = PrefixRetention(
            self.tiers.pool,
            engine_settings["HBM_pages"],
            self.shared_pages * engine_settings["shared_groups"],
        )
        self.retention_seen = set()
        self.retention_steps = 0
        self.retention_completed = False
        log_event(
            engine_settings,
            "retention.jsonl",
            dict(
                event="RETENTION_START",
                retained_pages=0,
                expected_shared_pages=self.retention.expected_pages,
                empty_measured_cache=True,
                retention_rule="declared shared prefix only, after actual GPU computation",
                artificial_request_cap=False,
            ),
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        # This callback is entered after synchronous GPU execution. Save block
        # objects before the native completion path frees finished request tables.
        tables = self.tiers.managers[0].req_to_blocks
        candidates, complete_prefixes = completed_shared_candidates(
            scheduler_output.num_scheduled_tokens,
            self.requests,
            tables,
            self.retention_seen,
            self.shared_pages,
        )
        result = super().update_from_output(scheduler_output, model_runner_output)
        added = self.retention.retain(candidates, gpu_completed=True)
        self.retention_seen.update(complete_prefixes)
        if added:
            invariant = self.retention.verify()
            log_event(
                self.engine_settings,
                "retention.jsonl",
                dict(
                    event="PREFIX_PAGES_RETAINED",
                    step=self.retention_steps,
                    gpu_completed=True,
                    added=added,
                    HBM_free_pages=self.tiers.hbm_free_count(),
                    resident_requests=len(self.running),
                    **invariant
                ),
            )
        # Empty engine queues can occur between rate-paced arrivals. Only finish
        # validation after every measured request has computed its shared prefix
        # and all active work has completed; never reset or release retained KV.
        all_requests_seen = (
            len(self.retention_seen) == self.engine_settings["n_requests"]
        )
        if (
            all_requests_seen
            and self.get_num_unfinished_requests() == 0
            and not self.retention_completed
        ):
            invariant = self.retention.verify()
            assert invariant["retained_pages"] == invariant["expected_shared_pages"]
            log_event(
                self.engine_settings,
                "retention.jsonl",
                dict(
                    event="RETENTION_COMPLETE", step=self.retention_steps, **invariant
                ),
            )
            self.retention_completed = True
        self.retention_steps += 1
        return result


class RetainedBaselineScheduler(Observer, RetainedPrefixScheduler):
    """Use the shared prefix scheduler for the HBM comparison"""

    pass
