"""Build request features and run hourly model updates on supplied data.

build_request_features() calculates user ID, session position, age and gap.
run_hourly_models() fits on earlier observations and saves the next hour's
predictions. examples/serving_replay.py runs the synthetic CPU demo."""

import sys
from pathlib import Path

import numpy as np

from run_simulation import ROOT, read_json, verify_file_hash
from trace_reader import MODEL, loads

sys.path.insert(0, str(ROOT / "hbf_simulator/model"))
import model_training
from file_helpers import write_json

H = model_training.H


def build_request_features(source, prepared, work):
    """Compute each request’s features before updating its session history."""
    users, codes = {}, []
    with source.open("rb") as raw:
        for line in raw:
            row = loads(line)
            if row.get("model_id") == MODEL:
                user = row["user_id"]
                if not isinstance(user, str) or not user:
                    raise ValueError("Invalid user ID")
                codes.append(users.setdefault(user, len(users)))
    perm = np.fromfile(prepared / "perm_time.i64", "<i8")
    user = np.array(codes, dtype="<i8")[perm]
    sessions = np.fromfile(prepared / "dst_sess.i32", "<i4")
    times = np.fromfile(prepared / "dst_ts.i64", "<i8")
    count, age, gap = np.empty((3, len(times)), dtype=float)
    seen = {}
    for i, (sid, now) in enumerate(zip(sessions, times)):
        # On the first turn the count and age are zero. A gap of -1 means no previous turn.
        pos, first, last = seen.get(sid, (0, now, now))
        count[i] = pos
        age[i] = (now - first) / 1e6
        gap[i] = (now - last) / 1e6 if pos else -1
        seen[sid] = pos + 1, first, now
    names = read_json(ROOT / "hbf_simulator/model/model_settings.json")["features"]
    np.savez(work / "features.npz", **dict(zip(names, (user, count, age, gap))))
    write_json(
        work / "registry.json",
        dict(
            features=[
                dict(
                    name=n,
                    status="ALLOWED",
                    encoding="categorical" if i == 0 else "numeric",
                )
                for i, n in enumerate(names)
            ]
        ),
    )
    return times


def run_hourly_models(work, label_dir, first, last, dest, expected):
    """Run the hourly fitting loop and combine predictions into one file."""
    data = model_training.RequestData(label_dir)
    scores = np.zeros(data.n, dtype="<f8")
    for hour in range(first, last):
        predictions, first_row, stop_row = model_training.fit_hourly_model(
            data, hour * H, work / "fits" / dest.name / str(hour)
        )
        scores[first_row:stop_row] = predictions
    scores.tofile(dest)
    verify_file_hash(dest, expected)
