"""Train on a synthetic history and replay three later requests with a two-block budget."""

import array
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Use the shared trace reader and simulator helpers.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "hbf_simulator/experiments")
)
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

    def training_examples(cutoff_us):
        # Second requests arrive exactly one microsecond after the first.
        keep = arrival_times_us[train_rows] < cutoff_us
        selected_rows = train_rows[keep]
        observed_reads = np.array(labels)[keep] * (
            arrival_times_us[selected_rows] + 1 < cutoff_us
        )
        return selected_rows, observed_reads

    data = SimpleNamespace(
        training_examples=training_examples,
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
    """Fit on synthetic sessions, then check flash hits and writes on three later requests."""
    requests, split, feature_table, scores, model = prepare_example()
    with tempfile.TemporaryDirectory() as temp_directory:
        work = Path(temp_directory)
        source = work / "requests.jsonl"
        with source.open("w") as f:
            for request in requests:
                row = dict(
                    model_id="deepseek-v4-flash",
                    session_id=str(request["session"]),
                    user_id=str(request["user"]),
                    timestamp_arrival=request["time"] / 1e6,
                    hash_ids=request["blocks"],
                )
                f.write(json.dumps(row) + "\n")
        prepare_trace(source, work / "prepared")
        # Check that feature building from the JSONL agrees with the synthetic inputs.
        build_request_features(source, work / "prepared", work)
        names = read_json(ROOT / "hbf_simulator/model/model_settings.json")["features"]
        with np.load(work / "features.npz") as saved:
            rebuilt_features = np.column_stack([saved[name] for name in names])
        if not np.array_equal(rebuilt_features, feature_table):
            raise RuntimeError(
                "Feature preparation differs from the synthetic examples"
            )
        stream = work / "prepared/dst"
        for name, array_type, values in [
            ("zero", "B", [0] * len(requests)),
            ("eligible", "B", [1] * len(requests)),
            ("scores", "d", scores),
        ]:
            (work / ("stream" + name)).write_bytes(
                array.array(array_type, values).tobytes()
            )
        run_program(
            "hbm_only_readable_prefix",
            stream,
            256,
            1,
            split,
            split + 100,
            work / "shadow",
        )
        result = run_cache_simulation(
            stream,
            256,
            2,
            1,
            work / "streamzero",
            work / "streameligible",
            work / "streamscores",
            17,
            2,
            split,
            split + 100,
            work / "shadow",
            work / "out",
        )
        print({k: result[k] for k in ("requests", "hbf_hits", "hbm_hits", "writes")})
        if (result["requests"], result["hbf_hits"], result["writes"]) != (3, 4, 2):
            raise RuntimeError("Unexpected demo result")


if __name__ == "__main__":
    run_demo()
