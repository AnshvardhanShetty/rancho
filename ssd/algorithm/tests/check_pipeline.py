"""Exercise all 96 updates and accounting stages on generated test data.

Training labels and histories are synthetic plumbing fixtures, not a workload
validation or a measure of predictor quality. All files live in a temporary folder.
"""

from pathlib import Path
import hashlib, json, subprocess, tempfile
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ssd.algorithm.training_data import EDT
from ssd.algorithm.settings import HOUR
from ssd.algorithm.run import prepare, replay

repo = Path(__file__).resolve().parents[3]
with tempfile.TemporaryDirectory(prefix="ssd-full-pipeline-") as tmp:
    root = Path(tmp)
    ntrain = 900
    measured = 384
    times = np.concatenate(
        [
            np.arange(ntrain, dtype=np.int64) * 72 * HOUR // ntrain,
            72 * HOUR + np.arange(measured, dtype=np.int64) * HOUR // 4,
        ]
    )
    rows = [[] for _ in range(ntrain)] + [
        list((np.arange(896) + i * 896) % 90000) for i in range(measured)
    ]
    prefix = root / "stream"
    for suffix, dtype, values in [
        ("_ts.i64", "<i8", times),
        ("_nb.i32", "<i4", list(map(len, rows))),
        ("_ids.i32", "<i4", sum(rows, [])),
    ]:
        np.asarray(values, dtype=dtype).tofile(str(prefix) + suffix)
    tokens = root / "tokens"
    np.asarray([16 * len(r) + 1 for r in rows], dtype="<i8").tofile(tokens)
    common = [
        "--prefix",
        prefix,
        "--tokens",
        tokens,
        "--nb",
        90000,
        "--hbm",
        81380,
        "--ssd",
        5000,
        "--budget",
        10000,
        "--eval-start-us",
        72 * HOUR,
        "--stop-us",
        168 * HOUR,
    ]
    ref = root / "reference"
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
                    ref,
                    "--evictions-out",
                    root / "cold_evictions",
                ],
            )
        ),
        check=True,
        capture_output=True,
    )
    cold = np.fromfile(root / "cold_evictions", EDT)
    assert len(cold) > 0
    train = np.zeros(ntrain, dtype=EDT)
    train["row"] = np.arange(ntrain)
    train["last"] = np.maximum(0, train["row"] - 1)
    train["born"] = train["last"]
    train["block"] = np.arange(ntrain) % 90000
    train["freq"] = 1 + np.arange(ntrain) % 8
    train["nev"] = np.arange(ntrain) % 5
    events = root / "all_evictions"
    np.concatenate([train, cold]).tofile(events)
    np.save(root / "samples.npy", np.arange(ntrain, dtype=np.int64))
    features = {
        name: np.arange(len(times), dtype=float) % 7
        for name in [
            "user",
            "causal_request_position",
            "previous_session_gap_seconds",
            "session_age_seconds",
        ]
    }
    np.savez(root / "features.npz", **features)
    counts = np.zeros((169, ntrain), dtype="<u4")
    for hour in range(169):
        counts[hour] = np.maximum(
            0, np.floor((hour * HOUR - times[:ntrain]) / HOUR)
        ).astype(np.uint32) * (1 + np.arange(ntrain) % 3)
    counts.tofile(root / "observed")
    offsets = np.concatenate(
        [[0], np.cumsum(list(map(len, rows)), dtype=np.uint64)]
    ).astype("<u8")
    offsets.tofile(root / "offsets.i64")
    flags = []
    seen = set()
    for row in rows:
        flags.extend([int(b in seen) for b in row])
        seen.update(row)
    bits = np.zeros((len(flags) + 63) // 64 + 1, dtype="<u8")
    for i, f in enumerate(flags):
        if f:
            bits[i // 64] |= np.uint64(1) << np.uint64(i % 64)
    bits.tofile(root / "credit.bits")
    ranks = np.concatenate(
        [[0], np.cumsum([int(v).bit_count() for v in bits], dtype=np.uint64)]
    ).astype("<u8")
    ranks.tofile(root / "credit.rank")

    def record(p):
        return dict(
            path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()
        )

    manifest = dict(
        eviction_count=len(train) + len(cold),
        ntrain=len(train),
        feature_schema={
            k: dict(encoding="categorical" if k == "user" else "numeric")
            for k in [
                *features,
                "prior_touch_count",
                "prior_eviction_count",
                "seconds_in_HBM",
                "seconds_since_last_touch",
            ]
        },
        files=dict(
            timestamps=record(str(prefix) + "_ts.i64"),
            evictions=record(events),
            features=record(root / "features.npz"),
            sample_indices=record(root / "samples.npy"),
        ),
        cumulative_labels=record(root / "observed"),
        replay=dict(
            block_ids=record(str(prefix) + "_ids.i32"),
            block_counts=record(str(prefix) + "_nb.i32"),
            arrival_times=record(str(prefix) + "_ts.i64"),
            tokens=record(tokens),
            measured_evictions=record(root / "cold_evictions"),
            block_id_count=90000,
            credit=dict(
                offsets=record(root / "offsets.i64"),
                bits=record(root / "credit.bits"),
                ranks=record(root / "credit.rank"),
            ),
            reference_rows=record(str(ref) + ".rows.i64"),
        ),
    )
    path = root / "inputs.json"
    path.write_text(json.dumps(manifest))
    out = root / "run"
    out.mkdir()
    data, scores = prepare(path, out)
    result = replay(path, data, scores, out, 5000, 10000)
    assert result["writes"] <= 10000 and result["eligible_after_hbm"] > 0
    assert len(list((out / "models").iterdir())) == 96
    proof = dict(
        status="PASS_SYNTHETIC_COMPLETE_PIPELINE",
        hourly_models=96,
        measured_evictions=len(cold),
        writes=result["writes"],
        eligible_after_hbm=result["eligible_after_hbm"],
        saved_positions=result["saved_positions"],
        private_data_used=False,
    )
    print(json.dumps(proof))
