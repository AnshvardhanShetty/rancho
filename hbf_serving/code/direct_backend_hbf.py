"""Move completed KV to HBF when RANCHO accepts a write

The scheduler finds full pages and protects any copies already on HBF
The worker waits for prefill to finish then makes the accepted copies
The scheduler updates the cache after those copies finish

HBF is emulated in GPU memory and the run limits write speed
Reuse from the current request is added after its write decision
"""

import time
from direct_backend import DirectConnector, DirectScheduler, Copied, log_event
from reuse import find_existing_hbf_copy, remap_to_existing_hbf_copies


class HBFConnector(DirectConnector):
    """Copy accepted KV pages and report completed transfers to the scheduler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.get("updated_rancho"):
            from updated_policy import FilteredHistory

            self.history = FilteredHistory(self.config["HBM_pages"])

    def wait_for_save(self):
        """Check each completed request and copy the KV that RANCHO accepts"""
        import torch

        metadata = self._get_connector_metadata()
        self.output = Copied()
        if not metadata.candidates:
            return
        torch.cuda.synchronize()
        destinations = list(metadata.destinations)
        copied_hashes = {}
        for candidate in metadata.candidates:
            history_start = time.perf_counter_ns()
            reuse_history = self.history.collect_request_reuse(
                candidate["request_id"],
                candidate["prompt_keys"],
                candidate["session"],
                candidate["suffix_begin"],
            )
            history_prepare_ns = time.perf_counter_ns() - history_start
            try:
                # Reuse any pages already copied so only new pages cost a write
                all_sources = list(candidate["sources"])
                reuse = list(candidate["reuse"])
                reused_sources = {source_page for source_page, target_page in reuse}
                for source_page in all_sources:
                    key = candidate["source_hashes"][source_page]
                    if source_page not in reused_sources and key in copied_hashes:
                        reuse.append((source_page, copied_hashes[key]))
                        reused_sources.add(source_page)
                sources = [
                    source_page
                    for source_page in all_sources
                    if source_page not in reused_sources
                ]
                distinct, cost = len(all_sources), len(sources)
                assert distinct == len(
                    set(candidate["prompt_keys"][candidate["suffix_begin"] :])
                )
                event = dict(
                    request_id=candidate["request_id"],
                    source_pages=distinct,
                    write_cost_pages=cost,
                    existing_HBF_pages=len(reuse),
                    prefill_GPU_completed=True,
                    policy=self.config["policy"],
                    features=candidate["features"],
                    policy_us=candidate["policy_us"],
                    scheduler_feature_cpu_ns=candidate["feature_cpu_ns"],
                    history=dict(
                        shadow_cached_pages=reuse_history["shadow_cached_pages"],
                        past_filtered_reuses=reuse_history["past_filtered_reuses"],
                        suffix_begin=reuse_history["suffix_begin"],
                    ),
                )
                if not all_sources:
                    event["action"] = "NO_NEW_FULL_SUFFIX"
                    self.output.events.append(event)
                    continue
                if cost > len(destinations):
                    event["action"] = "NO_PHYSICAL_DESTINATION_CAPACITY"
                    self.output.events.append(event)
                    continue
                self.now_us = max(self.now_us, candidate["policy_us"])
                if self.config.get("physical_write_accounting"):
                    self.policy.set_request_context(candidate["request_id"])
                # Score using only reuse from before this request
                self.policy.set_suffix_reuse_count(
                    reuse_history["past_filtered_reuses"]
                )
                decision = self.policy.decide_hbf_write_after_prefill(
                    self.event_id,
                    self.now_us,
                    candidate["features"],
                    distinct,
                    cost,
                    computed_prompt_tokens=candidate["prompt_tokens"],
                    prompt_tokens=candidate["prompt_tokens"],
                    gpu_completed=True,
                )
                # Comparisons with the calibrated NumPy threshold can return a
                # NumPy boolean. Preserve its truth value at this JSON boundary.
                decision["admit"] = bool(decision["admit"])
                self.event_id += 1
                event["decision"] = decision
                if not decision["admit"]:
                    event["action"] = "POLICY_REJECT"
                    self.output.events.append(event)
                    continue
                # Copy in chunks that fit the buffer and wait for the write time
                targets = destinations[:cost]
                del destinations[:cost]
                page_bytes = self.config["native_block_bytes"]
                pages_per_chunk = self.config["write_buffer_bytes"] // page_bytes
                if pages_per_chunk < 1:
                    raise ValueError("Write buffer cannot hold a native page")
                started = time.monotonic_ns()
                for offset in range(0, cost, pages_per_chunk):
                    source_page = sources[offset : offset + pages_per_chunk]
                    target_page = targets[offset : offset + pages_per_chunk]
                    begin = time.monotonic_ns()
                    for view in self.pages.pages.values():
                        indices = torch.tensor(
                            source_page, device=view.device, dtype=torch.long
                        )
                        target = torch.tensor(
                            target_page, device=view.device, dtype=torch.long
                        )
                        view.index_copy_(0, target, view.index_select(0, indices))
                    torch.cuda.synchronize()
                    deadline = (
                        begin
                        + (
                            len(source_page) * page_bytes * 1000000000
                            + self.config["write_bandwidth_bytes_s"]
                            - 1
                        )
                        // self.config["write_bandwidth_bytes_s"]
                    )
                    delay = (deadline - time.monotonic_ns()) / 1e9
                    if delay > 0:
                        time.sleep(delay)
                self.output.pairs.extend(zip(sources, targets))
                for source_page, target_page in zip(sources, targets):
                    copied_hashes[candidate["source_hashes"][source_page]] = target_page
                event.update(
                    action="COPIED" if cost else "REUSED_EXISTING_ONLY",
                    reuse_pairs=reuse,
                    native_bytes=cost * page_bytes,
                    copy_and_service_ns=time.monotonic_ns() - started,
                )
                self.output.events.append(event)
            finally:
                # Record this request for future offers even if the write was declined
                history_start = time.perf_counter_ns()
                self.history.record_request_reuse(reuse_history)
                event["history_cpu_ns"] = (
                    history_prepare_ns + time.perf_counter_ns() - history_start
                )
        log_event(
            self.config,
            "worker.jsonl",
            dict(
                event="COMPLETED_PREFILL_OFFERS",
                offers=self.output.events,
                gpu_copy_completed=True,
            ),
        )


class HBFScheduler(DirectScheduler):
    """Identify existing HBF copies and publish newly completed writes."""

    def schedule(self, *args, **kwargs):
        """Protect existing copies and prepare the next worker transfer."""
        output = super().schedule(*args, **kwargs)
        meta = output.kv_connector_metadata
        protected = set()
        for candidate in meta.candidates:
            reuse = []
            candidate["source_hashes"] = {}
            for source in candidate["sources"]:
                key = self.tiers.pool.blocks[source].block_hash
                if key is None:
                    raise ValueError(
                        "An offered full native page lacks its content hash"
                    )
                candidate["source_hashes"][source] = repr(key)
                target = find_existing_hbf_copy(self.tiers, source)
                if target is not None:
                    reuse.append((source, target))
                    protected.add(target)
            candidate["reuse"] = reuse
        meta.destinations = [dst for dst in meta.destinations if dst not in protected]
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        """Make completed copies visible to future cache lookups."""
        conn = model_runner_output.kv_connector_output
        meta = conn.kv_connector_worker_meta if conn else None
        if meta is not None:
            if meta.pairs:
                result = self.tiers.commit_completed_copies(
                    meta.pairs, gpu_completed=meta.gpu_completed
                )
                log_event(
                    self.engine_settings,
                    "scheduler.jsonl",
                    dict(event="PUBLISHED_NATIVE_HBF", **result),
                )
                meta.pairs = []
            reuse = [
                pair for event in meta.events for pair in event.get("reuse_pairs", [])
            ]
            if reuse:
                result = remap_to_existing_hbf_copies(
                    self.tiers, reuse, gpu_completed=meta.gpu_completed
                )
                log_event(
                    self.engine_settings,
                    "scheduler.jsonl",
                    dict(event="REUSED_NATIVE_HBF", **result),
                )
        return super().update_from_output(scheduler_output, model_runner_output)
