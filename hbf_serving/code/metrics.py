"""Count output tokens and measure delays from the saved request times"""

import math
import statistics


def describe(values):
    """Summarise a list of times and report its percentiles"""
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, median=None, p95=None)

    def percentile(p):
        at = p * (len(values) - 1)
        low = math.floor(at)
        high = math.ceil(at)
        return values[low] + (values[high] - values[low]) * (at - low)

    return dict(
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p95=percentile(0.95),
    )


def summarise(records):
    """Compute throughput and delays for the measured requests"""
    if not records or any(r["status"] != "COMPLETE" for r in records):
        raise ValueError("Every measured GPU request must complete")
    if len({r["row"] for r in records}) != len(records):
        raise ValueError("Duplicate measured request")
    ttft, tpot, latency = [], [], []
    outputs = 0
    for r in records:
        n = r["expected_output_tokens"]
        if type(n) is not int or n < 0 or r["generated_tokens"] != n:
            raise ValueError("Recorded output length changed or incomplete")
        arrival, dispatch, finish = r["scheduled_ns"], r["dispatch_ns"], r["finish_ns"]
        if not 0 <= arrival <= dispatch <= finish:
            raise ValueError("Invalid arrival/dispatch/completion timing")
        if n:
            first, last = r["first_token_ns"], r["last_token_ns"]
            if first is None or last is None or not dispatch <= first <= last <= finish:
                raise ValueError("Missing or inconsistent real output timestamps")
            ttft.append((first - arrival) / 1e9)
            if n > 1:
                tpot.append((last - first) / (n - 1) / 1e9)
        elif r["first_token_ns"] is not None or r["last_token_ns"] is not None:
            raise ValueError(
                "Zero-output request must not have visible output timestamps"
            )
        latency.append((finish - arrival) / 1e9)
        outputs += n
    start = min(r["scheduled_ns"] for r in records)
    end = max(r["finish_ns"] for r in records)
    if end <= start:
        raise ValueError("Positive measured interval required")
    elapsed = (end - start) / 1e9
    return dict(
        requests=len(records),
        output_tokens=outputs,
        first_arrival_to_last_completion_seconds=elapsed,
        completed_requests_per_second=len(records) / elapsed,
        output_tokens_per_second=outputs / elapsed,
        TTFT_seconds=describe(ttft),
        TPOT_seconds=describe(tpot),
        request_latency_seconds=describe(latency),
        startup_and_teardown_excluded=True,
        zero_output_requests=sum(r["expected_output_tokens"] == 0 for r in records),
    )


def validate_pair(plan):
    """Check that the run configuration is complete and consistent."""
    if plan["execution"] != "real_GPU" or plan["simulation_only"] is not False:
        raise ValueError("This trial requires real GPU generation")
    if plan["model"] != "Qwen/Qwen3-4B-Instruct-2507" or plan["exchange_ratio"] != 8:
        raise ValueError("Wrong selected model or exchange comparison")
    if plan["warmup_requests"] != 0 or plan["initial_cache"] != "empty":
        raise ValueError("Both arms must use the selected empty-cache start")
    if (
        type(plan["measured_requests"]) is not int
        or not 100 <= plan["measured_requests"] <= 200
    ):
        raise ValueError("Measured cohort must contain 100 to 200 requests")
    fields = ["baseline_HBM_bytes", "RANCHO_HBM_bytes", "RANCHO_HBF_bytes"]
    if any(type(plan[f]) is not int or plan[f] <= 0 for f in fields):
        raise ValueError("Explicit positive capacities are required")
    if (
        8 * plan["baseline_HBM_bytes"]
        != 8 * plan["RANCHO_HBM_bytes"] + plan["RANCHO_HBF_bytes"]
    ):
        raise ValueError("Pair violates the selected 8:1 resource budget")
    if plan["HBM_accounting"] != "shared_active_and_retained":
        raise ValueError(
            "A separate fixed active-KV quota hides the intended capacity benefit"
        )
    for f in [
        "native_block_bytes",
        "cohort_sha256",
        "arrivals_sha256",
        "weights_accounting",
        "controller_initial_state",
        "policy_clock",
        "write_allowance",
        "write_channel",
        "GPU_backend_certificate",
    ]:
        if not plan.get(f):
            raise ValueError("Unspecified required field: " + f)
    cert = plan["GPU_backend_certificate"]
    if (
        cert.get("status") != "PASS_NATIVE_RANCHO_ENGINE"
        or cert.get("GPU_execution") is not True
    ):
        raise ValueError("Native GPU tier integration is not qualified")
    return True
