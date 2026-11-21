"""Train on a synthetic history and replay three later requests with a two-block budget."""

import array
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Use the shared trace reader and simulator helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hbf_simulator/experiments"))
from trace_reader import prepare_trace
from run_simulation import ROOT, read_json, run_cache_simulation, run_program
from run_experiment import build_request_features

sys.path.insert(0, str(ROOT / "hbf_simulator/model"))
import model_training


def prepare_example():
    """Fit the bundled synthetic predictor and prepare the three-request fixture."""
    # Each independent training session requests its single block twice.
    history = []
    for i in range(240):
        for turn in range(2):
            history.append(dict(time=i * 100 + turn, session=i, user=i % 8, blocks=[i]))
    requests = history + read_json(ROOT / "hbf_simulator/examples/demo_requests.json")
    split = requests[len(history)]["time"]
    features, sessions, offered, labels = [], {}, [], []
    last = {}
    for i, request in enumerate(requests):
        session, now = request["session"], request["time"]
        count, start, previous = sessions.get(session, (0, now, now))
        features.append(
            [
                request["user"],
                count,
                (now - start) / 1e6,
                (now - previous) / 1e6 if count else -1,
            ]
        )
        sessions[session] = count + 1, start, now
        if now < split:
            # The first request offers the block. Its second request supplies one label.
            for block in set(request["blocks"]):
                key = session, block
                if key in last:
                    labels[last[key]] += 1
                else:
                    last[key] = len(labels)
                    offered.append(i)
                    labels.append(0)
    feature_table = np.array(features, float)
    train_rows = np.array(offered)
    arrival_times_us = np.array([request["time"] for request in requests])

    def get_training_rows_and_reuse(cutoff_us):
        # Second requests arrive exactly one microsecond after the first.
        keep = arrival_times_us[train_rows] < cutoff_us
        selected_rows = train_rows[keep]
        observed_reads = np.array(labels)[keep] * (
            arrival_times_us[selected_rows] + 1 < cutoff_us
        )
        return selected_rows, observed_reads

    data = SimpleNamespace(
        get_training_rows_and_reuse=get_training_rows_and_reuse,
        arrival_times_us=arrival_times_us,
        feature_table=lambda prediction_rows, names: feature_table[prediction_rows],
        categorical_columns=lambda names: [0],
    )
    model, _ = model_training.fit_tree_model(
        data, split, read_json(ROOT / "hbf_simulator/model/model_settings.json")
    )
    if model["kind"] != "expanded_tree":
        raise RuntimeError("Expected a fitted tree in the demo")
    prediction_rows = np.arange(len(history), len(requests))
    scores = [0.0] * len(history) + model_training.predict_reuse(
        model, feature_table[prediction_rows]
    ).tolist()
    return requests, split, feature_table, scores, model


def run_demo():
    """Fit on synthetic sessions, then check the results of three later requests."""
    requests, start_us, feature_table, predictions, _ = prepare_example()
    end_us = start_us + 100
    block_id_count = 256
    hbm_capacity = 1

    with tempfile.TemporaryDirectory() as temp_directory:
        work = Path(temp_directory)
        trace_path = work / "requests.jsonl"
        prepared_directory = work / "prepared"

        # Save the synthetic requests in the same format as a trace.
        with trace_path.open("w") as trace_file:
            for request in requests:
                row = dict(
                    model_id="deepseek-v4-flash",
                    session_id=str(request["session"]),
                    user_id=str(request["user"]),
                    timestamp_arrival=request["time"] / 1e6,
                    hash_ids=request["blocks"],
                )
                trace_file.write(json.dumps(row) + "\n")

        # Check that reading the trace produces the features we trained on.
        prepare_trace(trace_path, prepared_directory)
        build_request_features(trace_path, prepared_directory, work)
        feature_names = read_json(ROOT / "hbf_simulator/model/model_settings.json")["features"]
        with np.load(work / "features.npz") as saved:
            rebuilt_features = np.column_stack([saved[name] for name in feature_names])
        if not np.array_equal(rebuilt_features, feature_table):
            raise RuntimeError("Feature preparation differs from the synthetic examples")

        # Save the request flags and one reuse prediction per request.
        stream_prefix = prepared_directory / "dst"
        zero_flags_path = work / "streamzero"
        eligible_requests_path = work / "streameligible"
        predictions_path = work / "streamscores"
        zero_flags_path.write_bytes(bytes(len(requests)))
        eligible_requests_path.write_bytes(bytes([1]) * len(requests))
        predictions_path.write_bytes(array.array("d", predictions).tobytes())

        # Compare what HBM alone can serve, then replay RANCHO.
        hbm_only_prefix_path = work / "shadow"
        run_program(
            "hbm_only_readable_prefix",
            stream_prefix,
            block_id_count,
            hbm_capacity,
            start_us,
            end_us,
            hbm_only_prefix_path,
        )
        result = run_cache_simulation(
            request_stream_prefix=stream_prefix,
            block_id_count=block_id_count,
            hbf_capacity=2,
            hbm_capacity=hbm_capacity,
            zero_flags_path=zero_flags_path,
            eligible_requests_path=eligible_requests_path,
            predictions_path=predictions_path,
            seed=17,
            write_budget=2,
            start_us=start_us,
            end_us=end_us,
            hbm_only_prefix_path=hbm_only_prefix_path,
            output_directory=work / "out",
        )

        print({name: result[name] for name in ("requests", "hbf_hits", "hbm_hits", "writes")})
        if (result["requests"], result["hbf_hits"], result["writes"]) != (3, 4, 2):
            raise RuntimeError("Unexpected demo result")


if __name__ == "__main__":
    run_demo()
