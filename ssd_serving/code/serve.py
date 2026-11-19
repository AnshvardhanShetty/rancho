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


def validate(plan):
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
        if type(plan[key]) != int or plan[key] < 0:
            raise ValueError(key)
    if not min(plan["output_tokens"], plan["hbm_pages"], plan["max_num_seqs"]) > 0:
        raise ValueError("Empty serving geometry")
    if plan["clock"] not in ("monotonic", "request_logical"):
        raise ValueError("Declare policy clock")
    reqs = plan["requests"]
    if list(reqs) != [f"row{i}" for i in range(len(reqs))] or not 0 <= plan[
        "warmups"
    ] < len(reqs):
        raise ValueError("Ordered request IDs required")
    measured = list(reqs.values())[plan["warmups"] :]
    times = [r["arrival_seconds"] for r in measured]
    if (
        any(not math.isfinite(t) for t in times + [plan["arrival_window_seconds"]])
        or times != sorted(times)
        or times[0] < 0
        or times[-1] > plan["arrival_window_seconds"]
    ):
        raise ValueError("Invalid scheduled arrivals")
    for r in reqs.values():
        if not r["tokens"] or len(r["tokens"]) + plan["output_tokens"] > min(
            plan["max_model_len"], 16 * plan["hbm_pages"]
        ):
            raise ValueError("Request cannot fit")
        if not isinstance(r["features"], dict):
            raise ValueError("Request feature mapping required")
        if plan["clock"] == "request_logical" and not math.isfinite(
            r["policy_seconds"]
        ):
            raise ValueError("Invalid logical time")


def run(path, out):
    """Run a fresh engine and save request outputs, timings and storage counters."""
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("Use the qualified batch-invariant launcher")
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.sampling_params import RequestOutputKind

    p = json.loads(path.read_text())
    validate(p)
    out.mkdir()
    p["output"] = str(out.resolve())
    p["batch_invariant"] = True
    cfg = out / "plan.json"
    cfg.write_text(json.dumps(p))
    os.environ["RANCHO_SSD_PLAN"] = str(cfg.resolve())
    engine = LLM(
        model=model_path(),
        moe_backend="triton",
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        max_model_len=p["max_model_len"],
        max_num_seqs=p["max_num_seqs"],
        max_num_batched_tokens=max(256, p["max_model_len"]),
        block_size=16,
        kv_cache_memory_bytes=(1 + p["hbm_pages"]) * 1572864,
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
    count = p["output_tokens"]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=count,
        min_tokens=count,
        ignore_eos=True,
        detokenize=False,
        seed=17,
        output_kind=RequestOutputKind.DELTA,
    )
    requests = list(p["requests"].items())
    state = {}
    aliases = {}
    pending = set()

    def submit(i, at):
        key, r = requests[i]
        state[key] = dict(
            row=i,
            expected_output_tokens=count,
            generated_tokens=0,
            scheduled_ns=at,
            dispatch_ns=time.monotonic_ns(),
            first_token_ns=None,
            last_token_ns=None,
            output_ids=[],
            warmup=i < p["warmups"],
        )
        internal = engine.llm_engine.add_request(
            key, dict(prompt_token_ids=r["tokens"]), sampling
        )
        aliases[internal] = aliases[key] = key
        pending.add(key)

    def step():
        outputs = engine.llm_engine.step()
        now = time.monotonic_ns()
        for result in outputs:
            key = aliases[result.request_id]
            r = state[key]
            ids = [t for x in result.outputs for t in x.token_ids]
            if ids:
                r["first_token_ns"] = r["first_token_ns"] or now
                r["last_token_ns"] = now
                r["output_ids"].extend(ids)
                r["generated_tokens"] += len(ids)
            if result.finished:
                assert r["generated_tokens"] == count
                r.update(
                    status="COMPLETE",
                    finish_ns=now,
                    native_cached_tokens=result.num_cached_tokens,
                )
                pending.remove(key)

    try:
        for i in range(p["warmups"]):
            submit(i, time.monotonic_ns())
            while pending:
                step()
        start = time.monotonic_ns()
        i = p["warmups"]
        while pending or i < len(requests):
            while i < len(requests):
                at = start + round(requests[i][1]["arrival_seconds"] * 1e9)
                if at > time.monotonic_ns():
                    break
                submit(i, at)
                i += 1
            if pending:
                step()
            elif i < len(requests):
                time.sleep(min(0.02, max(0, (at - time.monotonic_ns()) / 1e9)))
    finally:
        engine.llm_engine.engine_core.shutdown()
    rows = list(state.values())
    assert len(rows) == len(requests)
    measured = rows[p["warmups"] :]
    m = summarise(measured)
    elapsed = max(
        p["arrival_window_seconds"],
        (max(r["finish_ns"] for r in measured) - start) / 1e9,
    )
    m.update(
        window_and_drain_seconds=elapsed,
        output_tokens_per_second=m["output_tokens"] / elapsed,
        completed_requests_per_second=len(measured) / elapsed,
        measurement_start_ns=start,
    )
    (out / "requests.json").write_text(json.dumps(rows, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    run(args.plan, args.output)
