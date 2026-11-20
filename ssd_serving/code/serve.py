"""Send requests to vLLM with the SSD cache enabled.

validate() checks capacities, arrival times and request inputs.
run() starts one engine, sends warm-up requests, then runs the measured requests.
It records outputs and calculates response times and output rate.

Assumes the supplied Qwen model layout and synchronous single-GPU execution.
Warm-up writes are counted, but warm-up requests are excluded from main timings."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from metrics import summarise
from model_location import model_path


def validate(serving_settings):
    """Check that requests are ordered and fit the configured GPU cache."""
    for key in (
        "hbm_pages",
        "ssd_pages",
        "write_budget_pages",
        "output_tokens",
        "max_model_len",
        "max_num_seqs",
        "warmups",
    ):
        if type(serving_settings[key]) != int or serving_settings[key] < 0:
            raise ValueError(key)
    if (
        not min(
            serving_settings["output_tokens"],
            serving_settings["hbm_pages"],
            serving_settings["max_num_seqs"],
        )
        > 0
    ):
        raise ValueError("Empty serving geometry")
    if serving_settings["clock"] not in ("monotonic", "request_logical"):
        raise ValueError("Declare policy clock")
    requests = serving_settings["requests"]
    if list(requests) != [
        f"row{i}" for i in range(len(requests))
    ] or not 0 <= serving_settings["warmups"] < len(requests):
        raise ValueError("Ordered request IDs required")
    measured = list(requests.values())[serving_settings["warmups"] :]
    arrival_times = [request["arrival_seconds"] for request in measured]
    if (
        any(
            not math.isfinite(arrival_time)
            for arrival_time in arrival_times
            + [serving_settings["arrival_window_seconds"]]
        )
        or arrival_times != sorted(arrival_times)
        or arrival_times[0] < 0
        or arrival_times[-1] > serving_settings["arrival_window_seconds"]
    ):
        raise ValueError("Invalid scheduled arrivals")
    for request in requests.values():
        if not request["tokens"] or len(request["tokens"]) + serving_settings[
            "output_tokens"
        ] > min(serving_settings["max_model_len"], 16 * serving_settings["hbm_pages"]):
            raise ValueError("Request cannot fit")
        if not isinstance(request["features"], dict):
            raise ValueError("Request feature mapping required")
        if serving_settings["clock"] == "request_logical" and not math.isfinite(
            request["policy_seconds"]
        ):
            raise ValueError("Invalid logical time")


def run(serving_inputs_file, output_directory):
    """Run a fresh engine and save request outputs, timings and storage counters."""
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("Use the qualified batch-invariant launcher")
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.sampling_params import RequestOutputKind

    serving_settings = json.loads(serving_inputs_file.read_text())
    validate(serving_settings)
    output_directory.mkdir()
    serving_settings["output"] = str(output_directory.resolve())
    serving_settings["batch_invariant"] = True
    serving_settings_file = output_directory / "serving_settings.json"
    serving_settings_file.write_text(json.dumps(serving_settings))
    os.environ["RANCHO_SSD_SERVING_SETTINGS"] = str(serving_settings_file.resolve())
    engine = LLM(
        model=model_path(),
        moe_backend="triton",
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        max_model_len=serving_settings["max_model_len"],
        max_num_seqs=serving_settings["max_num_seqs"],
        max_num_batched_tokens=max(256, serving_settings["max_model_len"]),
        block_size=16,
        kv_cache_memory_bytes=(1 + serving_settings["hbm_pages"]) * 1572864,
        gpu_memory_utilization=0.15,
        distributed_executor_backend="uni",
        async_scheduling=False,
        scheduler_cls="ssd_backend.SSDScheduler",
        kv_transfer_config=KVTransferConfig(
            kv_connector="Connector",
            kv_connector_module_path="ssd_backend",
            kv_role="kv_both",
        ),
    )
    output_token_count = serving_settings["output_tokens"]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=output_token_count,
        min_tokens=output_token_count,
        ignore_eos=True,
        detokenize=False,
        seed=17,
        output_kind=RequestOutputKind.DELTA,
    )
    requests = list(serving_settings["requests"].items())
    state = {}
    aliases = {}
    pending = set()

    def submit(i, scheduled_ns):
        key, request_record = requests[i]
        state[key] = dict(
            row=i,
            expected_output_tokens=output_token_count,
            generated_tokens=0,
            scheduled_ns=scheduled_ns,
            dispatch_ns=time.monotonic_ns(),
            first_token_ns=None,
            last_token_ns=None,
            output_ids=[],
            warmup=i < serving_settings["warmups"],
        )
        internal = engine.llm_engine.add_request(
            key, dict(prompt_token_ids=request_record["tokens"]), sampling
        )
        aliases[internal] = aliases[key] = key
        pending.add(key)

    def step():
        outputs = engine.llm_engine.step()
        now = time.monotonic_ns()
        for result in outputs:
            key = aliases[result.request_id]
            request_record = state[key]
            output_token_ids = [t for x in result.outputs for t in x.token_ids]
            if output_token_ids:
                request_record["first_token_ns"] = (
                    request_record["first_token_ns"] or now
                )
                request_record["last_token_ns"] = now
                request_record["output_ids"].extend(output_token_ids)
                request_record["generated_tokens"] += len(output_token_ids)
            if result.finished:
                assert request_record["generated_tokens"] == output_token_count
                request_record.update(
                    status="COMPLETE",
                    finish_ns=now,
                    native_cached_tokens=result.num_cached_tokens,
                )
                pending.remove(key)

    try:
        for i in range(serving_settings["warmups"]):
            submit(i, time.monotonic_ns())
            while pending:
                step()
        start = time.monotonic_ns()
        i = serving_settings["warmups"]
        while pending or i < len(requests):
            while i < len(requests):
                scheduled_ns = start + round(requests[i][1]["arrival_seconds"] * 1e9)
                if scheduled_ns > time.monotonic_ns():
                    break
                submit(i, scheduled_ns)
                i += 1
            if pending:
                step()
            elif i < len(requests):
                time.sleep(
                    min(0.02, max(0, (scheduled_ns - time.monotonic_ns()) / 1e9))
                )
    finally:
        engine.llm_engine.engine_core.shutdown()
    rows = list(state.values())
    assert len(rows) == len(requests)
    measured = rows[serving_settings["warmups"] :]
    metrics = summarise(measured)
    elapsed = max(
        serving_settings["arrival_window_seconds"],
        (max(request_record["finish_ns"] for request_record in measured) - start) / 1e9,
    )
    metrics.update(
        window_and_drain_seconds=elapsed,
        output_tokens_per_second=metrics["output_tokens"] / elapsed,
        completed_requests_per_second=len(measured) / elapsed,
        measurement_start_ns=start,
    )
    (output_directory / "requests.json").write_text(json.dumps(rows, indent=2))
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("settings", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    run(args.settings, args.output)
