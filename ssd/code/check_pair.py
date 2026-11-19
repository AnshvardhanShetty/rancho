"""Compare the accept-all and reject-all SSD example runs.

Check that they used the same requests and produced the same generated output.
Each run's timing and I/O counts are checked separately."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def arm(root, accept):
    """Recompute one example’s timing and I/O checks."""
    cell = root / "cell"
    p = read(cell / "plan.json")
    assert p["batch_invariant"]
    assert p["verify_payloads"] and p["policy_factory"] == (
        "ssd_policy:accept_fixture" if accept else "ssd_policy:reject_fixture"
    )
    metrics = read(cell / "metrics.json")
    rows = read(cell / "requests.json")
    measured = [r for r in rows if not r["warmup"]]
    assert len(rows) == len(p["requests"]) and len(measured) == metrics["requests"]
    begin = metrics["measurement_start_ns"]
    for r in measured:
        info = p["requests"]["row" + str(r["row"])]
        assert r["status"] == "COMPLETE" and len(r["output_ids"]) == p["output_tokens"]
        assert r["scheduled_ns"] == begin + round(1e9 * info["arrival_seconds"])
        assert (
            r["scheduled_ns"]
            <= r["dispatch_ns"]
            <= r["first_token_ns"]
            <= r["last_token_ns"]
            <= r["finish_ns"]
        )
    values = sorted((r["first_token_ns"] - r["scheduled_ns"]) / 1e9 for r in measured)
    k = 0.95 * (len(values) - 1)
    p95 = (
        values[math.floor(k)] * (math.ceil(k) - k)
        + values[math.ceil(k)] * (k - math.floor(k))
        if k % 1
        else values[int(k)]
    )
    assert math.isclose(p95, metrics["TTFT_seconds"]["p95"])
    window = max(
        p["arrival_window_seconds"],
        (max(r["finish_ns"] for r in measured) - begin) / 1e9,
    )
    assert math.isclose(
        sum(len(r["output_ids"]) for r in measured) / window,
        metrics["output_tokens_per_second"],
    )
    geometry = read(cell / "geometry.jsonl")
    assert (
        geometry["direct"] and geometry["dsync"] and geometry["page_bytes"] == 1572864
    )
    last = json.loads((cell / "io.jsonl").read_text().splitlines()[-1])
    assert last["evictions"] == last["absent_offers"] + last["free_retentions"]
    assert last["absent_offers"] == last["written_pages"] + last["unwritten_offers"]
    assert last["resident_ssd_pages"] <= p["ssd_pages"]
    if accept:
        assert 0 < last["restored_pages"] <= last["read_pages"]
        assert (
            last["written_pages"] == p["write_budget_pages"]
            and last["unwritten_offers"] > 0
        )
        assert last["replacements"] > 0 and last["free_retentions"] > 0
    else:
        assert (
            last["written_pages"] == last["read_pages"] == last["restored_pages"] == 0
        )
        assert last["unwritten_offers"] > 0
    cfg = {k: v for k, v in p.items() if k not in ("output", "policy_factory")}
    return (
        cfg,
        [r["output_ids"] for r in rows],
        dict(
            io=last,
            geometry=geometry,
            complete_requests=len(rows),
            measured_requests=len(measured),
            p95_ttft_formula_checked=True,
            throughput_window_and_drain_checked=True,
        ),
    )


def main(a, b, out):
    """Check matching outputs and save the paired correctness report."""
    c1, t1, r1 = arm(a, True)
    c2, t2, r2 = arm(b, False)
    assert c1 == c2 and t1 == t2
    result = dict(
        status="PASS",
        scope="physical SSD single-GPU serving correctness fixture",
        qualification="single native Qwen group, synchronous executor, greedy output",
        batch_invariant=True,
        accept=r1,
        reject=r2,
        identical_output_tokens=sum(map(len, t1)),
        payload_hash_checked_on_each_read=True,
        restored_gpu_bytes_checked=True,
        performance_benchmark=False,
        source_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("accept", "reject", "out"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    main(args.accept, args.reject, args.out)
