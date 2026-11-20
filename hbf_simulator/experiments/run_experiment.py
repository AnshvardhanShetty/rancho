"""Build the predictor's inputs and fit the requested hours

build_request_features makes the four inputs in arrival order
run_hourly_models collects one prediction for each request in those hours
"""

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
    # Give each user a number then put those numbers in replay order
    user_codes, users_in_file_order = {}, []
    with source.open("rb") as trace_file:
        for line in trace_file:
            row = loads(line)
            if row.get("model_id") == MODEL:
                user_id = row["user_id"]
                if not isinstance(user_id, str) or not user_id:
                    raise ValueError("Invalid user ID")
                users_in_file_order.append(
                    user_codes.setdefault(user_id, len(user_codes))
                )
    arrival_order = np.fromfile(prepared / "perm_time.i64", "<i8")
    user_id = np.array(users_in_file_order, dtype="<i8")[arrival_order]
    session_codes = np.fromfile(prepared / "dst_sess.i32", "<i4")
    arrival_times = np.fromfile(prepared / "dst_ts.i64", "<i8")
    turn_number, session_age, previous_gap = np.empty(
        (3, len(arrival_times)), dtype=float
    )
    # Read each chat's earlier state before adding the current request
    session_history = {}
    for row_index, (session, now) in enumerate(zip(session_codes, arrival_times)):
        # On the first turn the count and age are zero. A gap of -1 means no previous turn.
        earlier_turns, first_arrival, previous_arrival = session_history.get(
            session, (0, now, now)
        )
        turn_number[row_index] = earlier_turns
        session_age[row_index] = (now - first_arrival) / 1e6
        previous_gap[row_index] = (
            (now - previous_arrival) / 1e6 if earlier_turns else -1
        )
        session_history[session] = earlier_turns + 1, first_arrival, now
    feature_names = read_json(ROOT / "hbf_simulator/model/model_settings.json")[
        "features"
    ]
    np.savez(
        work / "features.npz",
        **dict(zip(feature_names, (user_id, turn_number, session_age, previous_gap)))
    )
    write_json(
        work / "registry.json",
        dict(
            features=[
                dict(
                    name=feature_name,
                    status="ALLOWED",
                    encoding="categorical" if row_index == 0 else "numeric",
                )
                for row_index, feature_name in enumerate(feature_names)
            ]
        ),
    )
    return arrival_times


def run_hourly_models(work, label_dir, first, last, dest, expected):
    """Run the hourly fitting loop and combine predictions into one file."""
    data = model_training.RequestData(label_dir)
    predictions_by_request = np.zeros(data.n, dtype="<f8")
    for hour in range(first, last):
        predictions, first_row, stop_row = model_training.fit_hourly_model(
            data, hour * H, work / "fits" / dest.name / str(hour)
        )
        predictions_by_request[first_row:stop_row] = predictions
    predictions_by_request.tofile(dest)
    verify_file_hash(dest, expected)
