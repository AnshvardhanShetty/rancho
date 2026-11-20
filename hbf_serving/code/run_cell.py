"""Run one serving benchmark and check its recorded requests and writes

run sends the warmup and measured requests then saves the logs
compute gets hit counts and timings from those logs
audit checks the counts and the write budget again
"""

import argparse, hashlib, json, math, os, re, time
from pathlib import Path
from controlled_workload import prompts, bound_requests, eligibility
from synthetic_workload import FIT
from updated_model import SuffixGBT
from campaign_state import read_run_json, save
from metrics import summarise, describe
from physical_time import PhysicalBudget


def read(json_path):
    """Read one JSON file."""
    return json.loads(Path(json_path).read_text())


def rows(jsonl_path):
    """Read the JSON records stored one per line."""
    return [json.loads(line) for line in Path(jsonl_path).read_text().splitlines()]


def digest(file_path):
    """Calculate a file digest for later consistency checks."""
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def index(request_id):
    """Extract the request’s position from its engine ID."""
    request_id_match = re.fullmatch(r"row(\d+)(?:-[0-9a-f]{8})?", request_id)
    if not request_id_match:
        raise ValueError("Unrecognized native request identity")
    return int(request_id_match[1])


def compute(run_directory, workload_settings):
    """Calculate hits, timings and write charges from the saved engine events."""
    engine_settings = read_run_json(run_directory, "engine_settings.json", "PLAN.json")
    request_records = rows(run_directory / "requests.jsonl")
    first_cache_lookups = [
        x
        for x in rows(run_directory / "scheduler.jsonl")
        if x["event"] == "INITIAL_PREFIX_LOOKUP"
    ]
    request_count = workload_settings["groups"] + workload_settings["requests"]
    assert len(request_records) == len(first_cache_lookups) == request_count
    lookup_by_request_row = {index(x["request_id"]): x for x in first_cache_lookups}
    assert set(lookup_by_request_row) == set(range(request_count))
    assert sorted(r["row"] for r in request_records) == list(
        range(request_count)
    ) and all(
        r["status"] == "COMPLETE" and r["generated_tokens"] == 64
        for r in request_records
    )
    # Count eligible reads served by the first cache lookup for each request.
    mask = eligibility(workload_settings)
    measured = [v for v in request_records if not v["warmup"]]
    assert len(measured) == workload_settings["requests"]
    hit = total = hbf = hbm = 0
    allhit = alltotal = 0
    for i, bits in enumerate(mask):
        e = lookup_by_request_row[i]
        k = e["cached_tokens"] // 16
        assert (
            e["cached_tokens"]
            == 16 * k
            == 16 * (e["cached_HBF_full_pages"] + e["cached_HBM_full_pages"])
        )
        assert 0 <= k <= len(bits) and all(
            bits[:k]
        ), "Cached prefix eligibility differs from declared synthetic history"
        got = sum(bits[:k])
        den = sum(bits)
        allhit += got
        alltotal += den
        if i >= workload_settings["groups"]:
            hit += got
            total += den
            hbf += e["cached_HBF_full_pages"]
            hbm += e["cached_HBM_full_pages"]
    assert hit == hbf + hbm and 0 <= hit <= total
    # Measure output rate and delays, including time spent draining the queue.
    m = summarise(measured)
    allm = summarise(request_records)
    begin = measured[0]["measurement_start_ns"]
    end = max(v["finish_ns"] for v in measured)
    interval = max(workload_settings["measurement_seconds"], (end - begin) / 1e9)
    m.update(
        output_tokens_per_second=m["output_tokens"] / interval,
        completed_requests_per_second=len(measured) / interval,
        arrival_window_seconds=workload_settings["measurement_seconds"],
        window_and_drain_seconds=interval,
        drain_seconds=max(
            0, (end - begin) / 1e9 - workload_settings["measurement_seconds"]
        ),
    )
    # Summarize concurrent requests and the cost of admission decisions.
    steps = rows(run_directory / "concurrency.jsonl")
    active = [v for v in steps if v["start_ns"] >= begin]
    offers = (
        [o for v in rows(run_directory / "worker.jsonl") for o in v.get("offers", [])]
        if engine_settings["HBF_pages"]
        else []
    )
    decision_ns = [
        o["decision"]["decision_cpu_ns"]
        for o in offers
        if "decision" in o and index(o["request_id"]) >= workload_settings["groups"]
    ]
    admission = dict(
        live_calls=sum("decision" in o for o in offers),
        measured_live_calls=len(decision_ns),
        admitted=sum(bool(o.get("decision", {}).get("admit")) for o in offers),
        paid_pages=sum(o.get("decision", {}).get("paid_cost", 0) for o in offers),
        copied_bytes=sum(o.get("native_bytes", 0) for o in offers),
        decision_CPU_microseconds=describe([v / 1000 for v in decision_ns]),
        decision_CPU_seconds=sum(decision_ns) / 1e9,
    )
    result = dict(
        status="COMPLETE_PENDING_AUDIT",
        job=read_run_json(run_directory, "comparison_settings.json", "JOB.json"),
        metrics=m,
        cold_start_inclusive_metrics=allm,
        offline_hit_percent=100 * hit / total,
        eligible_accesses=total,
        eligible_hits=hit,
        HBF_hit_pp=100 * hbf / total,
        HBM_hit_pp=100 * hbm / total,
        cold_start_inclusive_offline_hit_percent=100 * allhit / alltotal,
        hit_ratio=hit / total,
        admission=admission,
        concurrency=dict(
            peak_active=max((v["resident_requests"] for v in active), default=0),
            peak_waiting=max((v["waiting_requests"] for v in active), default=0),
            peak_scheduled_decode=max(
                (v["scheduled_decode_requests"] for v in active), default=0
            ),
            minimum_free_HBM_pages=min(
                (v["HBM_free_after_schedule"] for v in active), default=None
            ),
            native_allocation_refusals=sum(
                v["native_KV_allocation_refusals"] for v in active
            ),
        ),
        payload_sha256=engine_settings["payload_sha256"],
        GPU_execution=True,
        synthetic=True,
        full_queue_drained=True,
        warmup_real_requests=workload_settings["groups"],
        policy_version="filtered_linear_suffix_GBT_cost_band_half_behind_physical_v2",
        HBF_reads="unthrottled GPU-memory emulation",
    )
    # Reconstruct actual write credit and verify every recorded payment.
    if engine_settings["HBF_pages"]:
        physical_rows = [
            v
            for v in rows(run_directory / "worker.jsonl")
            if v.get("event") == "PHYSICAL_CONTROLLER_CLOCK_START"
        ]
        assert len(physical_rows) == 1
        origin = physical_rows[0]["origin_ns"]
        ledger = PhysicalBudget(engine_settings["HBF_pages"], origin)
        for offer in offers:
            if "decision" not in offer:
                continue
            d = offer["decision"]
            warm = index(offer["request_id"]) < workload_settings["groups"]
            allowance = ledger.limit(
                d["physical_decision_ns"], warm, None if warm else begin
            )
            assert (
                d["physical_write_limit_pages"] == allowance
                and d["physical_warmup"] == warm
            )
            assert d["physical_origin_ns"] == origin
            assert (
                d["controller_time_us"]
                == engine_settings["fit"]["cut_us"]
                + (d["physical_decision_ns"] - origin) // 1000
            )
            ledger.charge(d["paid_cost"], warm, allowance)
        assert ledger.total == admission["paid_pages"]
        result["physical_writes"] = ledger.summary(end, begin, interval)
        assert (
            result["physical_writes"]["steady_rate_respected"]
            and result["physical_writes"]["daily_allowance_respected"]
        )
        total_ns = [
            o["decision"]["total_admission_cpu_ns"]
            for o in offers
            if "decision" in o and index(o["request_id"]) >= workload_settings["groups"]
        ]
        admission["predictor_controller_and_guard_CPU_microseconds"] = describe(
            [v / 1000 for v in total_ns]
        )
        history_ns = [
            o.get("history_cpu_ns", 0)
            for o in offers
            if index(o["request_id"]) >= workload_settings["groups"]
        ]
        admission["filtered_history_CPU_microseconds"] = describe(
            [v / 1000 for v in history_ns]
        )
        admission["total_admission_CPU_microseconds"] = describe(
            [
                (
                    o["decision"]["total_admission_cpu_ns"]
                    + o.get("history_cpu_ns", 0)
                    + o.get("scheduler_feature_cpu_ns", 0)
                )
                / 1000
                for o in offers
                if "decision" in o
                and index(o["request_id"]) >= workload_settings["groups"]
            ]
        )
        for label, key in [
            ("predictor_CPU_microseconds", "predictor_cpu_ns"),
            ("controller_CPU_microseconds", "controller_cpu_ns"),
        ]:
            admission[label] = describe(
                [
                    o["decision"][key] / 1000
                    for o in offers
                    if "decision" in o
                    and index(o["request_id"]) >= workload_settings["groups"]
                ]
            )
        admission["scheduler_feature_CPU_microseconds"] = describe(
            [
                o.get("scheduler_feature_cpu_ns", 0) / 1000
                for o in offers
                if index(o["request_id"]) >= workload_settings["groups"]
            ]
        )
        decisions = [o["decision"] for o in offers if "decision" in o]
        admission["mechanisms"] = dict(
            threshold_active=sum(d["have_threshold"] for d in decisions),
            history_boosted=sum(d["history_multiplier"] > 1 for d in decisions),
            half_behind=sum(d["half_behind"] for d in decisions),
            score_rejections=sum(
                not d["score_admit_before_physical_guard"] for d in decisions
            ),
            physical_guard_rejections=sum(
                d["physical_guard_rejected"] for d in decisions
            ),
        )
    return result


def audit(run_directory):
    """Recompute the results and independently check cache and policy accounting."""
    from controllers import Wallet

    engine_settings = read_run_json(run_directory, "engine_settings.json", "PLAN.json")
    workload_settings = read_run_json(
        run_directory, "workload_settings.json", "SETUP.json"
    )
    saved_results = read(run_directory / "RESULT.json")
    recomputed_results = compute(run_directory, workload_settings)
    assert (
        saved_results == recomputed_results
    ), "Independent timestamp/eligibility/metric recomputation mismatch"
    # Check first lookup hits against the earlier synthetic requests
    # Count each eligible block once even if a request was paused and resumed
    actual = [v for v in rows(run_directory / "requests.jsonl") if not v["warmup"]]
    measured_lookups = {
        index(v["request_id"]): v
        for v in rows(run_directory / "scheduler.jsonl")
        if v["event"] == "INITIAL_PREFIX_LOOKUP"
        and index(v["request_id"]) >= workload_settings["groups"]
    }
    expected_eligible_reads = (
        len(actual)
        * (
            workload_settings["prefix_tokens"]
            + workload_settings["unique_suffix_tokens"]
        )
        // 16
    )
    independent_hits = 0
    for v in actual:
        first_lookup = measured_lookups[v["row"]]
        assert (
            0 <= first_lookup["cached_tokens"] <= workload_settings["prefix_tokens"]
            and first_lookup["cached_tokens"] % 16 == 0
        )
        assert v["cached_tokens"] is not None and v["cached_tokens"] >= 0
        independent_hits += first_lookup["cached_tokens"] // 16
    assert (
        saved_results["eligible_accesses"] == expected_eligible_reads
        and saved_results["eligible_hits"] == independent_hits
    )
    assert (
        abs(
            saved_results["offline_hit_percent"]
            - 100 * independent_hits / expected_eligible_reads
        )
        < 1e-12
    )
    if engine_settings["ratio"]:
        assert engine_settings["HBF_pages"] == engine_settings["ratio"] * (
            engine_settings["baseline_HBM_pages"] - engine_settings["HBM_pages"]
        )
        from updated_audit import audit_policy

        saved_results["algorithm_audit"] = audit_policy(
            run_directory, engine_settings, saved_results
        )
    elif engine_settings["baseline_prefix_retention"]:
        from retention_audit import audit_retention

        saved_results["prefix_retention"] = audit_retention(
            run_directory, engine_settings, saved_results
        )
    save(
        run_directory / "AUDIT.json",
        dict(
            status="PASS_INDEPENDENT_SERVING_AND_POLICY_AUDIT",
            all_requests_checked=True,
            offline_eligibility_recomputed=True,
            actual_native_hits_checked=True,
            all_outputs64=True,
            full_queue_drained=True,
            ratio_checked=True,
            paid_writes_checked=True,
            policy_decisions_replayed=bool(engine_settings["ratio"]),
            baseline_retention=saved_results.get("prefix_retention"),
            algorithm_audit=saved_results.get("algorithm_audit"),
        ),
    )
    print("PASS_INDEPENDENT_SERVING_AND_POLICY_AUDIT", flush=True)


def run(workload_settings, comparison_settings, run_directory, source_checksums_file):
    """Start a fresh engine, issue requests, drain its queue and save metrics."""
    source = Path(__file__).resolve().parent
    if source_checksums_file:
        for n, h in read(source_checksums_file).items():
            assert digest(source / n) == h, n
    os.environ["RANCHO_DIAGNOSTIC_QUEUE"] = "indexed"
    run_directory.mkdir(exist_ok=False)
    # Prepare identical prompts and the selected cache allocation.
    hbm, hbf = comparison_settings["hbm"], comparison_settings["hbf"]
    payload = prompts(workload_settings)
    request_count = len(payload)
    prompt_tokens = (
        workload_settings["prefix_tokens"] + workload_settings["unique_suffix_tokens"]
    )
    assert (
        prompt_tokens + 64 <= workload_settings["context_limit"]
        and math.ceil((prompt_tokens + 64) / 16) <= hbm
        and hbm + hbf <= 16384
    )
    assert (
        request_count
        == workload_settings["groups"] + workload_settings["requests"]
        <= workload_settings["max_num_seqs"]
    )
    # Pass explicit capacities, workload metadata and write limits to the engine.
    engine_settings = dict(
        mode="benchmark",
        policy="filtered_linear_suffix_costband_half_behind" if hbf else "baseline",
        HBM_pages=hbm,
        HBF_pages=hbf,
        baseline_HBM_pages=workload_settings["baseline_pages"],
        ratio=comparison_settings["ratio"],
        block_tokens=16,
        native_block_bytes=workload_settings["native_page_bytes"],
        output=str(run_directory),
        requests=bound_requests(workload_settings),
        write_bandwidth_bytes_s=workload_settings["write_bandwidth_bytes_s"],
        write_buffer_bytes=workload_settings["write_buffer_bytes"],
        write_budget_pages=hbf,
        physical_write_accounting=bool(hbf),
        n_requests=request_count,
        arrival_rate=comparison_settings.get("rate"),
        payload_sha256=hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest(),
        input_tokens_per_request=prompt_tokens,
        baseline_prefix_retention=bool(
            comparison_settings.get("retain_prefixes", not bool(hbf))
        ),
        shared_prefix_tokens=workload_settings["prefix_tokens"],
        shared_groups=workload_settings["groups"],
    )
    if hbf:
        engine_settings.update(
            controller_history=comparison_settings["controller"],
            controller_history_sha256=digest(comparison_settings["controller"]),
            fit=comparison_settings["fit"],
            updated_rancho=True,
            repetition=comparison_settings["repetition"],
        )
    engine_settings_file = run_directory / "engine_settings.json"
    save(engine_settings_file, engine_settings)
    save(run_directory / "workload_settings.json", workload_settings)
    save(run_directory / "comparison_settings.json", comparison_settings)
    os.environ["RANCHO_HBF_ENGINE_SETTINGS"] = str(engine_settings_file)
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.sampling_params import RequestOutputKind

    model = __import__("model_location").model_path()
    llm = None
    try:
        # Install the custom scheduler and connector in a fresh vLLM engine.
        llm = LLM(
            model=model,
            moe_backend="triton",
            tensor_parallel_size=1,
            dtype="bfloat16",
            enforce_eager=True,
            enable_prefix_caching=True,
            max_model_len=workload_settings["context_limit"],
            max_num_seqs=workload_settings["max_num_seqs"],
            max_num_batched_tokens=2048,
            block_size=16,
            kv_cache_memory_bytes=(1 + hbm + hbf)
            * workload_settings["native_page_bytes"],
            gpu_memory_utilization=0.15,
            distributed_executor_backend="mp",
            async_scheduling=False,
            scheduler_cls=(
                "concurrency_backend.TieredScheduler"
                if hbf
                else (
                    "retained_baseline.RetainedBaselineScheduler"
                    if engine_settings["baseline_prefix_retention"]
                    else "concurrency_backend.BaselineScheduler"
                )
            ),
            worker_extension_cls="worker_probe.MemoryProbe",
            kv_transfer_config=KVTransferConfig(
                kv_connector="HBFConnector" if hbf else "DirectConnector",
                kv_connector_module_path="physical_hbf" if hbf else "direct_backend",
                kv_role="kv_both",
            ),
        )
        geometry = llm.collective_rpc("rancho_memory_snapshot")
        save(run_directory / "GEOMETRY.json", geometry)
        assert geometry[0]["native_cache_blocks"] == 1 + hbm + hbf
        state = {}
        aliases = {}
        pending = set()
        completed = 0
        serial = 0

        # Submit one request and remember its scheduled arrival time.
        def submit(i, scheduled, begin):
            rid = f"row{i}"
            state[rid] = dict(
                row=i,
                expected_output_tokens=64,
                scheduled_ns=scheduled,
                dispatch_ns=time.monotonic_ns(),
                first_token_ns=None,
                last_token_ns=None,
                generated_tokens=0,
                finish_ns=None,
                status="RUNNING",
                cached_tokens=None,
                warmup=i < workload_settings["groups"],
                measurement_start_ns=begin,
            )
            internal = llm.llm_engine.add_request(
                rid,
                dict(prompt_token_ids=payload[i]),
                SamplingParams(
                    temperature=0,
                    max_tokens=64,
                    min_tokens=64,
                    ignore_eos=True,
                    detokenize=False,
                    seed=17,
                    output_kind=RequestOutputKind.DELTA,
                ),
            )
            aliases[internal] = rid
            aliases[rid] = rid
            pending.add(rid)

        # Advance the engine and record generated tokens and completions.
        def step(handle):
            nonlocal completed
            outputs = llm.llm_engine.step()
            now = time.monotonic_ns()
            for request in outputs:
                rid = aliases[request.request_id]
                r = state[rid]
                count = sum(len(item.token_ids) for item in request.outputs)
                if count:
                    r["first_token_ns"] = r["first_token_ns"] or now
                    r["last_token_ns"] = now
                    r["generated_tokens"] += count
                r["cached_tokens"] = request.num_cached_tokens
                if request.finished:
                    r.update(finish_ns=now, status="COMPLETE")
                    pending.remove(rid)
                    completed += 1
                    handle.write(json.dumps(r) + "\n")

        # Fill each shared context, then start the timed arrival stream.
        with (run_directory / "requests.jsonl").open("x", buffering=1) as handle:
            for g in range(workload_settings["groups"]):
                now = time.monotonic_ns()
                submit(g, now, now)
                while pending:
                    step(handle)
            begin = time.monotonic_ns()
            save(
                run_directory / "MEASUREMENT_CLOCK.json",
                dict(
                    measurement_start_ns=begin,
                    clock="monotonic_ns",
                    startup_requests_completed=workload_settings["groups"],
                ),
            )
            dispatched = workload_settings["groups"]
            last_progress_ns = begin
            while completed < request_count:
                now = time.monotonic_ns()
                while dispatched < request_count:
                    at = dispatched - workload_settings["groups"]
                    scheduled = (
                        begin
                        if comparison_settings.get("pilot")
                        else begin + round(at * 1e9 / comparison_settings["rate"])
                    )
                    if now < scheduled:
                        break
                    submit(dispatched, scheduled, begin)
                    dispatched += 1
                    now = time.monotonic_ns()
                if not pending:
                    at = dispatched - workload_settings["groups"]
                    next_ns = begin + round(at * 1e9 / comparison_settings["rate"])
                    time.sleep(max(0, min(0.02, (next_ns - time.monotonic_ns()) / 1e9)))
                    continue
                step(handle)
                now = time.monotonic_ns()
                if (
                    now - last_progress_ns > 15_000_000_000
                    or completed == request_count
                ):
                    save(
                        run_directory / "PROGRESS.json",
                        dict(
                            dispatched=dispatched - workload_settings["groups"],
                            completed=completed - workload_settings["groups"],
                            pending=len(pending),
                            elapsed_seconds=(now - begin) / 1e9,
                        ),
                    )
                    last_progress_ns = now
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    result = compute(run_directory, workload_settings)
    save(run_directory / "RESULT.json", result)
    print(
        json.dumps(
            {
                "status": "CELL_COMPLETE_PENDING_AUDIT",
                "offline_hit_percent": result["offline_hit_percent"],
                "TTFT_p95_s": result["metrics"]["TTFT_seconds"]["p95"],
                "output_tokens_s": result["metrics"]["output_tokens_per_second"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workload-settings", "--setup")
    p.add_argument("--comparison-settings", "--job")
    p.add_argument("--output")
    p.add_argument("--manifest")
    p.add_argument("--audit")
    a = p.parse_args()
    if a.audit:
        audit(Path(a.audit))
    else:
        run(
            read(a.workload_settings),
            read(a.comparison_settings),
            Path(a.output),
            Path(a.manifest),
        )
