"""Check against synthetic outputs from the hash-matched recorded source.

These 96-hour fixtures activate spare-budget spending and hourly repricing.
They verify implementation parity, not the private workload result.
"""

from pathlib import Path
import hashlib, json, struct, subprocess, tempfile
import numpy as np

repo = Path(__file__).resolve().parents[3]
H = 3600000000
start = 72 * H
reports = []
with tempfile.TemporaryDirectory(prefix="ssd-policy-parity-") as tmp:
    root = Path(tmp)
    rows = []
    times = []
    for hour in range(96):
        for j in range(64):
            row = (
                [0, 1, 2, 3]
                if hour < 6
                else [4 + ((hour * 64 + j) * 2) % 500, 5 + ((hour * 64 + j) * 2) % 500]
            )
            rows.append(row)
            times.append(start + hour * H + j * H // 64)
    prefix = root / "stream"
    for suffix, kind, values in [
        ("_nb.i32", "<i4", list(map(len, rows))),
        ("_ids.i32", "<i4", sum(rows, [])),
        ("_ts.i64", "<i8", times),
    ]:
        np.asarray(values, dtype=kind).tofile(str(prefix) + suffix)
    tokens = root / "tokens"
    np.array([16 * len(r) + 1 for r in rows], dtype="<i8").tofile(tokens)
    cuts = root / "cuts"
    (np.arange(72, 168, dtype="<i8") * H).tofile(cuts)
    common = [
        "--prefix",
        prefix,
        "--tokens",
        tokens,
        "--nb",
        512,
        "--hbm",
        4,
        "--ssd",
        20,
        "--budget",
        10000,
        "--eval-start-us",
        start,
        "--stop-us",
        168 * H,
    ]
    subprocess.run(
        list(
            map(
                str,
                [
                    repo / "build/ssd/ssd_replay",
                    *common,
                    "--mode",
                    "all",
                    "--out-prefix",
                    root / "events",
                    "--evictions-out",
                    root / "evictions",
                ],
            )
        ),
        check=True,
        capture_output=True,
    )
    evictions = np.fromfile(root / "evictions", "<i4").reshape(-1, 6)
    raw = np.array(
        [256 * [0.0, 0.5, 2.0, 4.0, 16.0][i % 5] for i in range(len(evictions))],
        dtype="<f8",
    )
    scores = root / "test_raw.f64"
    raw.tofile(scores)
    for budget in [200, 2000, 10000]:
        outputs = []
        for binary, label in [(repo / "build/ssd/recorded_ssd_replay", "readable")]:
            out = root / f"{budget}_{label}"
            cmd = list(common)
            cmd[cmd.index("--budget") + 1] = budget
            cmd = [
                binary,
                *cmd,
                "--end-us",
                168 * H,
                "--mode",
                "quantile",
                "--seed",
                17,
                "--out-prefix",
                out,
                "--expected-evictions",
                root / "evictions",
                "--scores",
                scores,
                "--cuts-i64",
                cuts,
                "--reprice-stdio",
                "--reserve",
                0.15,
                "--release-us",
                120 * H,
                "--update-every",
                1000,
                "--history-size",
                20000,
                "--demand-multiplier",
                128,
                "--count-gain",
                32,
            ]
            count = 0
            with subprocess.Popen(
                list(map(str, cmd)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            ) as p:
                for line in p.stdout:
                    e = json.loads(line)
                    if e["event"] == "result":
                        result = e
                        continue
                    ids = e["ids"]
                    response = dict(
                        event="reprice_scores",
                        cut_us=e["cut_us"],
                        ids=ids,
                        scores=raw[np.asarray(ids, dtype=int)].tolist(),
                    )
                    p.stdin.write(json.dumps(response) + "\n")
                    p.stdin.flush()
                    count += 1
                assert p.wait() == 0 and count == 96
            outputs.append(
                (
                    result,
                    {
                        s: Path(str(out) + s).read_bytes()
                        for s in [
                            ".rows.i64",
                            ".programs.bits",
                            ".adaptive.txt",
                            ".hourly.tsv",
                            ".activation.txt",
                        ]
                    },
                )
            )

        result, files = outputs[0]
        reports.append(
            dict(
                budget=budget,
                requests=len(rows),
                evictions=len(evictions),
                hourly_updates=96,
                writes=result["writes"],
                supplemental_counters=files[".adaptive.txt"].decode().strip(),
                activation=files[".activation.txt"].decode().strip(),
                rows_sha256=hashlib.sha256(files[".rows.i64"]).hexdigest(),
                bits_sha256=hashlib.sha256(files[".programs.bits"]).hexdigest(),
            )
        )
expected = json.loads(Path(__file__).with_name("expected_policy.json").read_text())
assert reports == expected, "Recorded-source synthetic output changed"
print("PASS: recorded policy matches 3 golden cases, each with 96 hourly updates")
