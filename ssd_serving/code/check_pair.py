"""Compare the accept-all and reject-all SSD example runs.

Check that they used the same requests and produced the same generated output.
Each run's timing and I/O counts are checked separately."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(json_path):
    return json.loads(json_path.read_text())


def arm(root, accept):
    """Recompute one example’s timing and I/O checks."""
    cell = root / "cell"
    settings_file = cell / "serving_settings.json"
    if not settings_file.exists():
        settings_file = cell / "plan.json"
    serving_settings = read(settings_file)
    assert serving_settings["batch_invariant"]
    assert serving_settings["verify_payloads"] and serving_settings[
        "policy_factory"
    ] == ("ssd_policy:accept_fixture" if accept else "ssd_policy:reject_fixture")
    metrics = read(cell / "metrics.json")
    request_records = read(cell / "requests.json")
    measured = [
        request_record
        for request_record in request_records
        if not request_record["warmup"]
    ]
    assert (
        len(request_records) == len(serving_settings["requests"])
        and len(measured) == metrics["requests"]
    )
    begin = metrics["measurement_start_ns"]
    for request_record in measured:
        scheduled_request = serving_settings["requests"][
            "row" + str(request_record["row"])
        ]
        assert (
            request_record["status"] == "COMPLETE"
            and len(request_record["output_ids"]) == serving_settings["output_tokens"]
        )
        assert request_record["scheduled_ns"] == begin + round(
            1e9 * scheduled_request["arrival_seconds"]
        )
        assert (
            request_record["scheduled_ns"]
            <= request_record["dispatch_ns"]
            <= request_record["first_token_ns"]
            <= request_record["last_token_ns"]
            <= request_record["finish_ns"]
        )
    first_token_delays = sorted(
        (request_record["first_token_ns"] - request_record["scheduled_ns"]) / 1e9
        for request_record in measured
    )
    p95_index = 0.95 * (len(first_token_delays) - 1)
    p95 = (
        first_token_delays[math.floor(p95_index)] * (math.ceil(p95_index) - p95_index)
        + first_token_delays[math.ceil(p95_index)] * (p95_index - math.floor(p95_index))
        if p95_index % 1
        else first_token_delays[int(p95_index)]
    )
    assert math.isclose(p95, metrics["TTFT_seconds"]["p95"])
    window = max(
        serving_settings["arrival_window_seconds"],
        (max(request_record["finish_ns"] for request_record in measured) - begin) / 1e9,
    )
    assert math.isclose(
        sum(len(request_record["output_ids"]) for request_record in measured) / window,
        metrics["output_tokens_per_second"],
    )
    geometry = read(cell / "geometry.jsonl")
    assert (
        geometry["direct"] and geometry["dsync"] and geometry["page_bytes"] == 1572864
    )
    final_io_counters = json.loads((cell / "io.jsonl").read_text().splitlines()[-1])
    assert (
        final_io_counters["evictions"]
        == final_io_counters["absent_offers"] + final_io_counters["free_retentions"]
    )
    assert (
        final_io_counters["absent_offers"]
        == final_io_counters["written_pages"] + final_io_counters["unwritten_offers"]
    )
    assert final_io_counters["resident_ssd_pages"] <= serving_settings["ssd_pages"]
    if accept:
        assert (
            0 < final_io_counters["restored_pages"] <= final_io_counters["read_pages"]
        )
        assert (
            final_io_counters["written_pages"] == serving_settings["write_budget_pages"]
            and final_io_counters["unwritten_offers"] > 0
        )
        assert (
            final_io_counters["replacements"] > 0
            and final_io_counters["free_retentions"] > 0
        )
    else:
        assert (
            final_io_counters["written_pages"]
            == final_io_counters["read_pages"]
            == final_io_counters["restored_pages"]
            == 0
        )
        assert final_io_counters["unwritten_offers"] > 0
    comparable_settings = {
        p95_index: v
        for p95_index, v in serving_settings.items()
        if p95_index not in ("output", "policy_factory")
    }
    return (
        comparable_settings,
        [request_record["output_ids"] for request_record in request_records],
        dict(
            io=final_io_counters,
            geometry=geometry,
            complete_requests=len(request_records),
            measured_requests=len(measured),
            p95_ttft_formula_checked=True,
            throughput_window_and_drain_checked=True,
        ),
    )


def main(accept_directory, reject_directory, report_path):
    """Check matching outputs and save the paired correctness report."""
    accept_settings, accept_tokens, accept_report = arm(accept_directory, True)
    reject_settings, reject_tokens, reject_report = arm(reject_directory, False)
    assert accept_settings == reject_settings and accept_tokens == reject_tokens
    result = dict(
        status="PASS",
        scope="physical SSD single-GPU serving correctness fixture",
        qualification="single native Qwen group, synchronous executor, greedy output",
        batch_invariant=True,
        accept=accept_report,
        reject=reject_report,
        identical_output_tokens=sum(map(len, accept_tokens)),
        payload_hash_checked_on_each_read=True,
        restored_gpu_bytes_checked=True,
        performance_benchmark=False,
        source_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
    )
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("accept", "reject", "out"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    main(args.accept, args.reject, args.out)
