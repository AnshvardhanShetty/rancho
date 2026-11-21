"""Turn prepared requests and earlier reuse counts into a prediction file.

1. build_request_features reads the trace and saves four inputs per request.
2. run_hourly_models loads those inputs and the saved reuse counts. It fits on
   reads seen before each hour and predicts the requests arriving in that hour.

The trace reader supplies the replay order. The cache simulator uses the
finished predictions to make write decisions.
"""

import sys

import numpy as np

from run_simulation import ROOT, read_json, verify_file_hash
from trace_reader import MODEL, loads

sys.path.insert(0, str(ROOT / "hbf_simulator/model"))
import model_training
from file_helpers import write_json

H = model_training.H


def build_request_features(trace_path, prepared_directory, feature_output_directory):
    """Compute each request’s features before updating its session history."""
    # Number users in the order they first appear in the trace.
    user_codes = {}
    users_in_file_order = []
    with trace_path.open("rb") as trace_file:
        for line in trace_file:
            row = loads(line)
            if row.get("model_id") == MODEL:
                user_id = row["user_id"]
                if not isinstance(user_id, str) or not user_id:
                    raise ValueError("Invalid user ID")
                users_in_file_order.append(user_codes.setdefault(user_id, len(user_codes)))

    # Match these codes to the sessions and times already sorted by the trace reader.
    arrival_order = np.fromfile(prepared_directory / "perm_time.i64", "<i8")
    user_codes_in_arrival_order = np.array(users_in_file_order, dtype="<i8")[arrival_order]
    session_codes = np.fromfile(prepared_directory / "dst_sess.i32", "<i4")
    arrival_times = np.fromfile(prepared_directory / "dst_ts.i64", "<i8")

    # Work out what was known about each session when the request arrived.
    turn_number, session_age, previous_gap = np.empty((3, len(arrival_times)), dtype=float)
    session_history = {}
    for row_index, (session, arrival_us) in enumerate(zip(session_codes, arrival_times)):
        # On the first turn the count and age are zero. A gap of -1 means no previous turn.
        earlier_turns, first_arrival, previous_arrival = session_history.get(
            session, (0, arrival_us, arrival_us)
        )
        turn_number[row_index] = earlier_turns
        session_age[row_index] = (arrival_us - first_arrival) / 1e6
        previous_gap[row_index] = (arrival_us - previous_arrival) / 1e6 if earlier_turns else -1
        session_history[session] = earlier_turns + 1, first_arrival, arrival_us

    # Save one array per feature and tell the model which column holds user IDs.
    feature_names = read_json(ROOT / "hbf_simulator/model/model_settings.json")["features"]
    feature_values = (
        user_codes_in_arrival_order,
        turn_number,
        session_age,
        previous_gap,
    )
    np.savez(
        feature_output_directory / "features.npz", **dict(zip(feature_names, feature_values))
    )
    write_json(
        feature_output_directory / "registry.json",
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


def run_hourly_models(
    output_directory,
    label_directory,
    first_hour,
    end_hour,
    predictions_path,
    expected_sha256,
):
    """Save predictions in replay order, leaving unscored requests at zero."""
    data = model_training.RequestData(label_directory)
    predictions_by_request = np.zeros(data.request_count, dtype="<f8")

    # Each fit returns its predictions and where they belong in the full trace.
    for hour in range(first_hour, end_hour):
        predictions, first_row, stop_row = model_training.fit_and_predict_one_hour(
            data,
            hour * H,
            output_directory / "fits" / predictions_path.name / str(hour),
        )
        predictions_by_request[first_row:stop_row] = predictions

    predictions_by_request.tofile(predictions_path)
    verify_file_hash(predictions_path, expected_sha256)
