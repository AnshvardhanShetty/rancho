"""Count output tokens and measure delays from the saved request times"""

import math
import statistics


def summarize_latency_samples(values):
    """Summarise a list of times and report its percentiles"""
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, median=None, p95=None)

    def percentile(quantile):
        percentile_index = quantile * (len(values) - 1)
        lower_index = math.floor(percentile_index)
        upper_index = math.ceil(percentile_index)
        return values[lower_index] + (values[upper_index] - values[lower_index]) * (
            percentile_index - lower_index
        )

    return dict(
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p95=percentile(0.95),
    )


def summarize_request_performance(records):
    """Compute throughput and delays for the measured requests"""
    if not records or any(request_record["status"] != "COMPLETE" for request_record in records):
        raise ValueError("Every measured GPU request must complete")
    if len({request_record["row"] for request_record in records}) != len(records):
        raise ValueError("Duplicate measured request")
    first_token_delays, per_token_delays, request_delays = [], [], []
    generated_token_count = 0
    for request_record in records:
        expected_tokens = request_record["expected_output_tokens"]
        if (
            type(expected_tokens) is not int
            or expected_tokens < 0
            or request_record["generated_tokens"] != expected_tokens
        ):
            raise ValueError("Recorded output length changed or incomplete")
        arrival, dispatch, finish = (
            request_record["scheduled_ns"],
            request_record["dispatch_ns"],
            request_record["finish_ns"],
        )
        if not 0 <= arrival <= dispatch <= finish:
            raise ValueError("Invalid arrival/dispatch/completion timing")
        if expected_tokens:
            first, last = (
                request_record["first_token_ns"],
                request_record["last_token_ns"],
            )
            if first is None or last is None or not dispatch <= first <= last <= finish:
                raise ValueError("Missing or inconsistent real output timestamps")
            first_token_delays.append((first - arrival) / 1e9)
            if expected_tokens > 1:
                per_token_delays.append((last - first) / (expected_tokens - 1) / 1e9)
        elif (
            request_record["first_token_ns"] is not None
            or request_record["last_token_ns"] is not None
        ):
            raise ValueError("Zero-output request must not have visible output timestamps")
        request_delays.append((finish - arrival) / 1e9)
        generated_token_count += expected_tokens
    first_arrival_ns = min(request_record["scheduled_ns"] for request_record in records)
    last_completion_ns = max(request_record["finish_ns"] for request_record in records)
    if last_completion_ns <= first_arrival_ns:
        raise ValueError("Positive measured interval required")
    elapsed = (last_completion_ns - first_arrival_ns) / 1e9
    return dict(
        requests=len(records),
        output_tokens=generated_token_count,
        first_arrival_to_last_completion_seconds=elapsed,
        completed_requests_per_second=len(records) / elapsed,
        output_tokens_per_second=generated_token_count / elapsed,
        TTFT_seconds=summarize_latency_samples(first_token_delays),
        TPOT_seconds=summarize_latency_samples(per_token_delays),
        request_latency_seconds=summarize_latency_samples(request_delays),
        startup_and_teardown_excluded=True,
        zero_output_requests=sum(
            request_record["expected_output_tokens"] == 0 for request_record in records
        ),
    )
