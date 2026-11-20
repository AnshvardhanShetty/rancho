"""Record active requests and cache use at each scheduler step"""

import json, time
from pathlib import Path


class Observer:
    """Wrap the scheduler so both runs collect the same observations"""

    def __init__(self, *args, **kwargs):
        self.observer_preemptions = 0
        super().__init__(*args, **kwargs)
        self.observer_file = (
            Path(self.direct_plan["output"]) / "concurrency.jsonl"
        ).open("a", buffering=65536)
        self.observer_step = None
        self.observer_steps = 0
        self.observer_last_resident = None
        self.observer_refusals = 0
        self.observer_step_refusals = []
        native_allocate = self.kv_cache_manager.allocate_slots

        def observed_allocate(*args, **kwargs):
            result = native_allocate(*args, **kwargs)
            if result is None:
                free = self.tiers.hbm_free_count()
                request = args[0] if args else kwargs["request"]
                self.observer_refusals += 1
                self.observer_step_refusals.append(
                    dict(request_id=request.request_id, HBM_free_pages=free)
                )
            return result

        self.kv_cache_manager.allocate_slots = observed_allocate

    def _preempt_request(self, *args, **kwargs):
        self.observer_preemptions += 1
        return super()._preempt_request(*args, **kwargs)

    def schedule(self, *args, **kwargs):
        """Record the requests and cache space used for this step"""
        started = time.monotonic_ns()
        self.observer_step_refusals = []
        free_before = self.tiers.hbm_free_count()
        output = super().schedule(*args, **kwargs)
        host_schedule_ns = time.monotonic_ns() - started
        if self.observer_step is not None:
            raise ValueError("Only one synchronous observed GPU step is supported")
        before = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
        cached = output.scheduled_cached_reqs
        before.update(zip(cached.req_ids, cached.num_computed_tokens))
        scheduled = output.num_scheduled_tokens
        decode = sum(
            before[rid] >= len(self.requests[rid].prompt_token_ids) for rid in scheduled
        )
        prefill_tokens = sum(
            min(n, max(0, len(self.requests[rid].prompt_token_ids) - before[rid]))
            for rid, n in scheduled.items()
        )
        self.observer_step = dict(
            event="GPU_STEP",
            step=self.observer_steps,
            start_ns=started,
            host_schedule_ns=host_schedule_ns,
            resident_requests=len(self.running),
            waiting_requests=len(self.waiting) + len(self.skipped_waiting),
            scheduled_requests=len(scheduled),
            scheduled_decode_requests=decode,
            scheduled_prefill_requests=len(scheduled) - decode,
            scheduled_tokens=sum(scheduled.values()),
            scheduled_prefill_tokens=prefill_tokens,
            scheduled_decode_tokens=sum(scheduled.values()) - prefill_tokens,
            HBM_capacity_pages=self.tiers.hbm_pages,
            HBM_free_before_schedule=free_before,
            HBM_free_after_schedule=self.tiers.hbm_free_count(),
            native_KV_allocation_refusals=len(self.observer_step_refusals),
            native_KV_allocation_refusals_total=self.observer_refusals,
            refused_request_ids=sorted(
                {r["request_id"] for r in self.observer_step_refusals}
            ),
            refusal_samples=self.observer_step_refusals[:8],
        )
        return output

    def update_from_output(self, *args, **kwargs):
        """Record finished work after the model step"""
        host_start = time.monotonic_ns()
        result = super().update_from_output(*args, **kwargs)
        host_update_ns = time.monotonic_ns() - host_start
        r = self.observer_step
        if r is None:
            raise ValueError("GPU completion lacks its observed scheduling step")
        r.update(
            end_ns=time.monotonic_ns(),
            host_update_ns=host_update_ns,
            preemptions_total=self.observer_preemptions,
            HBM_free_after_completion=self.tiers.hbm_free_count(),
        )
        if (
            self.observer_steps % 64 == 0
            or r["resident_requests"] != self.observer_last_resident
        ):
            pool = self.tiers.pool
            r["tier_occupancy_sample"] = dict(
                active_HBM_pages=self.tiers.hbm_pages - self.tiers.hbm_free_count(),
                active_HBF_pages=sum(
                    b.ref_cnt > 0 for b in pool.blocks[self.tiers.hbm_pages + 1 :]
                ),
                retained_HBF_pages=sum(
                    b.block_hash is not None
                    for b in pool.blocks[self.tiers.hbm_pages + 1 :]
                ),
            )
        self.observer_file.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.observer_steps += 1
        self.observer_last_resident = r["resident_requests"]
        self.observer_step = None
        if self.observer_steps % 64 == 0 or self.get_num_unfinished_requests() == 0:
            self.observer_file.flush()
        return result
