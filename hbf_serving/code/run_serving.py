"""Run one serving benchmark and check its recorded requests and writes

run_serving_comparison sends requests and saves the engine logs
compute_serving_results counts hits and measures timings from those logs
"""

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from synthetic_requests import (
    build_workload_prompts,
    bind_features_to_request_ids,
    build_eligible_read_mask,
)
from run_files import resolve_model_path, write_run_json
from metrics import summarize_request_performance
from write_controller import PhysicalBudget


def read_json(json_path):
    """Read one JSON file."""
    return json.loads(Path(json_path).read_text())


def read_jsonl_records(jsonl_path):
    """Read the JSON records stored one per line."""
    return [json.loads(line) for line in Path(jsonl_path).read_text().splitlines()]


def sha256_file(file_path):
    """Calculate a file digest for later consistency checks."""
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def parse_request_row(request_id):
    """Extract the request’s position from its engine ID."""
    request_id_match = re.fullmatch(r"row(\d+)(?:-[0-9a-f]{8})?", request_id)
    if not request_id_match:
        raise ValueError("Unrecognized native request identity")
    return int(request_id_match[1])


def compute_serving_results(run_directory, workload_settings):
    """Calculate hits, timings and write charges from the saved engine events."""
    engine_settings = read_json(run_directory / "engine_settings.json")
    request_records = read_jsonl_records(run_directory / "requests.jsonl")
    first_cache_lookups = [
        lookup
        for lookup in read_jsonl_records(run_directory / "scheduler.jsonl")
        if lookup["event"] == "INITIAL_PREFIX_LOOKUP"
    ]
    request_count = workload_settings["groups"] + workload_settings["requests"]
    assert len(request_records) == len(first_cache_lookups) == request_count
    lookup_by_request_row = {
        parse_request_row(lookup["request_id"]): lookup for lookup in first_cache_lookups
    }
    assert set(lookup_by_request_row) == set(range(request_count))
    assert sorted(request_record["row"] for request_record in request_records) == list(
        range(request_count)
    ) and all(
        request_record["status"] == "COMPLETE" and request_record["generated_tokens"] == 64
        for request_record in request_records
    )
    # Count eligible reads served by the first cache lookup for each request.
    mask = build_eligible_read_mask(workload_settings)
    measured = [v for v in request_records if not v["warmup"]]
    assert len(measured) == workload_settings["requests"]
    measured_hits = total = measured_hbf_hits = measured_hbm_hits = 0
    allhit = alltotal = 0
    for request_index, bits in enumerate(mask):
        first_lookup = lookup_by_request_row[request_index]
        cached_prefix_blocks = first_lookup["cached_tokens"] // 16
        assert (
            first_lookup["cached_tokens"]
            == 16 * cached_prefix_blocks
            == 16
            * (first_lookup["cached_HBF_full_pages"] + first_lookup["cached_HBM_full_pages"])
        )
        assert 0 <= cached_prefix_blocks <= len(bits) and all(
            bits[:cached_prefix_blocks]
        ), "Cached prefix eligibility differs from declared synthetic history"
        credited_hits = sum(bits[:cached_prefix_blocks])
        eligible_blocks = sum(bits)
        allhit += credited_hits
        alltotal += eligible_blocks
        if request_index >= workload_settings["groups"]:
            measured_hits += credited_hits
            total += eligible_blocks
            measured_hbf_hits += first_lookup["cached_HBF_full_pages"]
            measured_hbm_hits += first_lookup["cached_HBM_full_pages"]
    assert measured_hits == measured_hbf_hits + measured_hbm_hits and 0 <= measured_hits <= total
    # Measure output rate and delays, including time spent draining the queue.
    measured_metrics = summarize_request_performance(measured)
    all_request_metrics = summarize_request_performance(request_records)
    begin = measured[0]["measurement_start_ns"]
    last_completion_ns = max(v["finish_ns"] for v in measured)
    interval = max(workload_settings["measurement_seconds"], (last_completion_ns - begin) / 1e9)
    measured_metrics.update(
        output_tokens_per_second=measured_metrics["output_tokens"] / interval,
        completed_requests_per_second=len(measured) / interval,
        arrival_window_seconds=workload_settings["measurement_seconds"],
        window_and_drain_seconds=interval,
        drain_seconds=max(
            0,
            (last_completion_ns - begin) / 1e9 - workload_settings["measurement_seconds"],
        ),
    )
    offers = (
        [
            offer
            for worker_event in read_jsonl_records(run_directory / "worker.jsonl")
            for offer in worker_event.get("offers", [])
        ]
        if engine_settings["HBF_pages"]
        else []
    )
    # Sum actual writes and copies from the worker's log.
    admission = dict(
        admitted=sum(bool(offer.get("decision", {}).get("admit")) for offer in offers),
        paid_pages=sum(offer.get("decision", {}).get("paid_cost", 0) for offer in offers),
        copied_bytes=sum(offer.get("native_bytes", 0) for offer in offers),
    )
    result = dict(
        status="COMPLETE",
        job=read_json(run_directory / "comparison_settings.json"),
        metrics=measured_metrics,
        cold_start_inclusive_metrics=all_request_metrics,
        offline_hit_percent=100 * measured_hits / total,
        eligible_accesses=total,
        eligible_hits=measured_hits,
        HBF_hit_pp=100 * measured_hbf_hits / total,
        HBM_hit_pp=100 * measured_hbm_hits / total,
        cold_start_inclusive_offline_hit_percent=100 * allhit / alltotal,
        hit_ratio=measured_hits / total,
        admission=admission,
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
            worker_event
            for worker_event in read_jsonl_records(run_directory / "worker.jsonl")
            if worker_event.get("event") == "PHYSICAL_CONTROLLER_CLOCK_START"
        ]
        assert len(physical_rows) == 1
        origin = physical_rows[0]["origin_ns"]
        ledger = PhysicalBudget(engine_settings["HBF_pages"], origin)
        for offer in offers:
            if "decision" not in offer:
                continue
            decision = offer["decision"]
            warm = parse_request_row(offer["request_id"]) < workload_settings["groups"]
            allowance = ledger.available_write_pages(
                decision["physical_decision_ns"], warm, None if warm else begin
            )
            assert (
                decision["physical_write_limit_pages"] == allowance
                and decision["physical_warmup"] == warm
            )
            assert decision["physical_origin_ns"] == origin
            assert (
                decision["controller_time_us"]
                == engine_settings["fit"]["cut_us"]
                + (decision["physical_decision_ns"] - origin) // 1000
            )
            ledger.charge_write_pages(decision["paid_cost"], warm, allowance)
        assert ledger.total == admission["paid_pages"]
        result["physical_writes"] = ledger.summarize_write_spending(
            last_completion_ns, begin, interval
        )
        assert (
            result["physical_writes"]["steady_rate_respected"]
            and result["physical_writes"]["daily_allowance_respected"]
        )
    return result


def run_serving_comparison(workload_settings, comparison_settings, run_directory):
    """Start a fresh engine, issue requests, drain its queue and save metrics."""
    run_directory.mkdir(exist_ok=False)
    # Prepare identical prompts and the selected cache allocation.
    hbm, hbf = comparison_settings["hbm"], comparison_settings["hbf"]
    payload = build_workload_prompts(workload_settings)
    request_count = len(payload)
    prompt_tokens = workload_settings["prefix_tokens"] + workload_settings["unique_suffix_tokens"]
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
        requests=bind_features_to_request_ids(workload_settings),
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
        baseline_prefix_retention=bool(comparison_settings.get("retain_prefixes", not bool(hbf))),
        shared_prefix_tokens=workload_settings["prefix_tokens"],
        shared_groups=workload_settings["groups"],
    )
    if hbf:
        engine_settings.update(
            controller_history=comparison_settings["controller"],
            controller_history_sha256=sha256_file(comparison_settings["controller"]),
            fit=comparison_settings["fit"],
            updated_rancho=True,
            repetition=comparison_settings["repetition"],
        )
    engine_settings_file = run_directory / "engine_settings.json"
    write_run_json(engine_settings_file, engine_settings)
    write_run_json(run_directory / "workload_settings.json", workload_settings)
    write_run_json(run_directory / "comparison_settings.json", comparison_settings)
    os.environ["RANCHO_HBF_ENGINE_SETTINGS"] = str(engine_settings_file)
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.sampling_params import RequestOutputKind

    model = resolve_model_path()
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
            kv_cache_memory_bytes=(1 + hbm + hbf) * workload_settings["native_page_bytes"],
            gpu_memory_utilization=0.15,
            distributed_executor_backend="mp",
            async_scheduling=False,
            scheduler_cls=(
                "direct_backend_hbf.HBFScheduler"
                if hbf
                else (
                    "direct_backend.RetainedPrefixScheduler"
                    if engine_settings["baseline_prefix_retention"]
                    else "direct_backend.DirectScheduler"
                )
            ),
            worker_extension_cls="native_pages.MemoryProbe",
            kv_transfer_config=KVTransferConfig(
                kv_connector="HBFConnector" if hbf else "DirectConnector",
                kv_connector_module_path="direct_backend_hbf" if hbf else "direct_backend",
                kv_role="kv_both",
            ),
        )
        geometry = llm.collective_rpc("rancho_memory_snapshot")
        write_run_json(run_directory / "GEOMETRY.json", geometry)
        assert geometry[0]["native_cache_blocks"] == 1 + hbm + hbf
        state = {}
        aliases = {}
        pending = set()
        completed = 0
        serial = 0

        # Submit one request and remember its scheduled arrival time.
        def submit(request_index, scheduled, begin):
            rid = f"row{request_index}"
            state[rid] = dict(
                row=request_index,
                expected_output_tokens=64,
                scheduled_ns=scheduled,
                dispatch_ns=time.monotonic_ns(),
                first_token_ns=None,
                last_token_ns=None,
                generated_tokens=0,
                finish_ns=None,
                status="RUNNING",
                cached_tokens=None,
                warmup=request_index < workload_settings["groups"],
                measurement_start_ns=begin,
            )
            internal = llm.llm_engine.add_request(
                rid,
                dict(prompt_token_ids=payload[request_index]),
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
        def advance_engine_and_record_outputs(handle):
            nonlocal completed
            outputs = llm.llm_engine.step()
            now = time.monotonic_ns()
            for request in outputs:
                rid = aliases[request.request_id]
                request_record = state[rid]
                count = sum(len(item.token_ids) for item in request.outputs)
                if count:
                    request_record["first_token_ns"] = request_record["first_token_ns"] or now
                    request_record["last_token_ns"] = now
                    request_record["generated_tokens"] += count
                request_record["cached_tokens"] = request.num_cached_tokens
                if request.finished:
                    request_record.update(finish_ns=now, status="COMPLETE")
                    pending.remove(rid)
                    completed += 1
                    handle.write(json.dumps(request_record) + "\n")

        # Fill each shared context, then start the timed arrival stream.
        with (run_directory / "requests.jsonl").open("x", buffering=1) as handle:
            for conversation_index in range(workload_settings["groups"]):
                now = time.monotonic_ns()
                submit(conversation_index, now, now)
                while pending:
                    advance_engine_and_record_outputs(handle)
            begin = time.monotonic_ns()
            write_run_json(
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
                    measured_request_index = dispatched - workload_settings["groups"]
                    scheduled = begin + round(
                        measured_request_index * 1e9 / comparison_settings["rate"]
                    )
                    if now < scheduled:
                        break
                    submit(dispatched, scheduled, begin)
                    dispatched += 1
                    now = time.monotonic_ns()
                if not pending:
                    measured_request_index = dispatched - workload_settings["groups"]
                    next_ns = begin + round(
                        measured_request_index * 1e9 / comparison_settings["rate"]
                    )
                    time.sleep(max(0, min(0.02, (next_ns - time.monotonic_ns()) / 1e9)))
                    continue
                advance_engine_and_record_outputs(handle)
                now = time.monotonic_ns()
                if now - last_progress_ns > 15_000_000_000 or completed == request_count:
                    write_run_json(
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
    result = compute_serving_results(run_directory, workload_settings)
    write_run_json(run_directory / "RESULT.json", result)
    print(
        json.dumps(
            {
                "status": "CELL_COMPLETE",
                "offline_hit_percent": result["offline_hit_percent"],
                "TTFT_p95_s": result["metrics"]["TTFT_seconds"]["p95"],
                "output_tokens_s": result["metrics"]["output_tokens_per_second"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-settings", type=Path, required=True)
    parser.add_argument("--comparison-settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_serving_comparison(
        read_json(arguments.workload_settings),
        read_json(arguments.comparison_settings),
        arguments.output,
    )
