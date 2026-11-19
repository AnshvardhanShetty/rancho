"""Check the recorded simulator policy on deterministic synthetic traces.

Golden hashes come from the simulator before the readability refactor.
Each case checks every saved output, including decisions, cache order and writes.
These small cases check behavior preservation; they do not reproduce the lab trace.
"""

import argparse
import array
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).with_name("simulator_expected.json")


def write_array(path, kind, values):
    array.array(kind, values).tofile(path.open("wb"))


def make_trace(folder, case):
    rng = random.Random(5147)
    requests = []
    # Earlier references, including duplicates, supply history without warming caches.
    for row in range(50):
        requests.append((row - 5, row % 4, [row % 8, row % 8, 8 + row % 12]))
    count = case["requests"]
    for row in range(count):
        if case["pattern"] == "tie":
            blocks = [32 + row % 8] * (1 + row % 3)
        elif case["pattern"] == "rewrite":
            blocks = [32 + row % 3, 33, 34, 35, 36, 33, 37, 38, 39]
        elif case["pattern"] == "empty":
            blocks = []
        else:
            session = row % 23
            length = 1 + rng.randrange(38)
            blocks = [32 + (session * 7 + j) % 223 for j in range(length)]
            if row % 5 == 0:
                blocks = [0, 1, 2, 3] + blocks
            if row % 7 == 0:
                blocks += blocks[:4]
            if row % 13 == 0:
                blocks = []
        # Irregular arrival spacing exercises both ahead/behind spending paths.
        time = 100 + row * 5 + (row // 4000) * 1000
        requests.append((time, row % 23, blocks))
    # Include a request beyond the end so the stopping position is also checked.
    requests.append((1_000_000, 1, [1, 2]))
    prefix = folder / "stream"
    write_array(Path(str(prefix) + "_nb.i32"), "i", [len(r[2]) for r in requests])
    write_array(Path(str(prefix) + "_sess.i32"), "i", [r[1] for r in requests])
    write_array(Path(str(prefix) + "_ts.i64"), "q", [r[0] for r in requests])
    write_array(
        Path(str(prefix) + "_ids.i32"), "i", [b for r in requests for b in r[2]]
    )
    write_array(folder / "zero", "B", [0] * len(requests))
    eligible = [
        int(r[0] >= 0 and (i % 11 != 0 or case["pattern"] == "tie"))
        for i, r in enumerate(requests)
    ]
    write_array(folder / "eligible", "B", eligible)
    predictions = [
        (
            1.0
            if case["pattern"] == "tie"
            else (0.0 if i % 17 == 0 else [0.01, 0.25, 1.0, 4.0, 13.0][i % 5])
        )
        for i in range(len(requests))
    ]
    write_array(folder / "scores", "d", predictions)
    end = 100 + count * 5 + (count // 4000) * 1000 + 500
    return prefix, end


def run_case(binary, case):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        stream, end = make_trace(folder, case)
        shadow = folder / "shadow"
        subprocess.run(
            [
                str(ROOT / "build/hbm_only_readable_prefix"),
                str(stream),
                "256",
                str(case["hbm"]),
                "100",
                str(end),
                str(shadow),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        out = folder / "out"
        command = [
            binary,
            stream,
            256,
            case["flash"],
            case["hbm"],
            folder / "zero",
            folder / "zero",
            case["budget"],
            100,
            end,
            "v1",
            out,
            "--live-rancho",
            case["seed"],
            folder / "eligible",
            folder / "scores",
        ]
        if case.get("debug"):
            command.append("--debug")
        command += [
            "--reuse-bonus",
            "--cost-bands",
            "--half-behind",
            "--shadow",
            "shadow-linear",
            shadow,
        ]
        subprocess.run(list(map(str, command)), check=True, stdout=subprocess.DEVNULL)
        outputs = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out.iterdir())
        }
        metrics = json.loads((out / "metrics.json").read_text())["evaluation"]
        diagnostic = json.loads((out / "diagnostic.json").read_text())
        assert metrics["writes"] <= case["budget"]
        assert metrics["credited_hits"] <= metrics["eligible_first_distinct"]
        return dict(outputs=outputs, metrics=metrics, diagnostic=diagnostic)


CASES = [
    dict(
        name="mixed_seed17",
        requests=24000,
        flash=23,
        hbm=11,
        budget=80000,
        seed=17,
        pattern="mixed",
    ),
    dict(
        name="mixed_seed29",
        requests=24000,
        flash=23,
        hbm=11,
        budget=80000,
        seed=29,
        pattern="mixed",
    ),
    dict(
        name="mixed_seed43",
        requests=24000,
        flash=23,
        hbm=11,
        budget=80000,
        seed=43,
        pattern="mixed",
    ),
    dict(
        name="flash_only",
        requests=2200,
        flash=50,
        hbm=0,
        budget=15000,
        seed=17,
        pattern="mixed",
    ),
    dict(
        name="budget_denied",
        requests=1100,
        flash=3,
        hbm=7,
        budget=9,
        seed=29,
        pattern="mixed",
        debug=True,
    ),
    dict(
        name="zero_budget",
        requests=35,
        flash=5,
        hbm=4,
        budget=0,
        seed=43,
        pattern="mixed",
        debug=True,
    ),
    dict(
        name="duplicate_rewrites",
        requests=2400,
        flash=4,
        hbm=3,
        budget=18000,
        seed=17,
        pattern="rewrite",
        debug=True,
    ),
    dict(
        name="equal_scores",
        requests=24000,
        flash=1,
        hbm=1,
        budget=18000,
        seed=29,
        pattern="tie",
    ),
    dict(
        name="empty_suffixes",
        requests=1500,
        flash=5,
        hbm=4,
        budget=100,
        seed=43,
        pattern="empty",
    ),
    dict(
        name="no_evaluation_requests",
        requests=0,
        flash=5,
        hbm=4,
        budget=100,
        seed=17,
        pattern="empty",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/rancho")
    parser.add_argument(
        "--record-original",
        action="store_true",
        help="Record expected hashes using the saved pre-refactor executable",
    )
    args = parser.parse_args()
    results = {case["name"]: run_case(args.binary.resolve(), case) for case in CASES}
    if args.record_original:
        if GOLDEN.exists():
            raise ValueError("Refusing to overwrite the expected results")
        GOLDEN.write_text(
            json.dumps(dict(cases=CASES, results=results), indent=2) + "\n"
        )
        print("Recorded original outputs for", len(CASES), "synthetic cases")
    else:
        expected = json.loads(GOLDEN.read_text())
        assert CASES == expected["cases"], "Case definitions changed"
        for name, actual in results.items():
            assert actual == expected["results"][name], f"Output mismatch: {name}"
        print("PASS:", len(CASES), "cases match all original output files")


if __name__ == "__main__":
    main()
