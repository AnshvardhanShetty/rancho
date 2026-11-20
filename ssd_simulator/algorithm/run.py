"""Run the recorded SSD experiment from prepared files

Check the inputs and fit one model for each measured hour
Start the C++ replay and update earlier offer scores as each model changes
Check the write log and count the prefill blocks SSD saved
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np
from .training_data import TrainingData, resolve_and_verify_input_file
from .training import fit_eviction_reuse_model, predict_reuse_for_evictions
from .settings import SETTINGS, HOUR
from .accounting import count_eligible_reads_saved_and_compare

HERE = Path(__file__).resolve().parent
BUILD = HERE.parents[1] / "build/ssd"


def write_json(path, value):
    """Save one JSON record, rejecting duplicate stage outputs."""
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def sha256_file(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def train_hourly_eviction_models(manifest_path, output):
    """Verify supplied arrays and train all 96 hourly raw-target models."""
    training_data = TrainingData(manifest_path)
    measured_data = TrainingData(manifest_path, cold=True)
    models = output / "models"
    models.mkdir()
    predictions_path = output / "test_raw.f64"
    with predictions_path.open("xb") as stream:
        for hour in range(72, 168):
            cutoff_us = hour * HOUR
            model, record = fit_eviction_reuse_model(training_data, cutoff_us, SETTINGS)
            if model["kind"] != "tree" or record["max_train_decision_us"] >= cutoff_us:
                raise ValueError("Expected a fitted causal tree")
            first_eviction, stop_eviction = (
                measured_data.find_first_eviction_at_or_after(cutoff_us),
                measured_data.find_first_eviction_at_or_after(cutoff_us + HOUR),
            )
            hour_output = models / str(hour)
            hour_output.mkdir()
            with (hour_output / "model.pkl").open("xb") as file:
                pickle.dump(model, file, protocol=5)
            for start in range(first_eviction, stop_eviction, 200000):
                rows = np.arange(start, min(start + 200000, stop_eviction))
                values, _ = predict_reuse_for_evictions(measured_data, model, rows)
                values.astype("<f8").tofile(stream)
            record.update(
                first=first_eviction,
                stop=stop_eviction,
                model_sha256=sha256_file(hour_output / "model.pkl"),
                training_target="historical marginal credited prefix unlock",
            )
            write_json(hour_output / "model_fit_summary.json", record)
    if predictions_path.stat().st_size != measured_data.eviction_count * 8:
        raise ValueError("One score per measured eviction is required")
    expected_sha256 = training_data.manifest.get("expected_scores_sha256")
    if expected_sha256 and sha256_file(predictions_path) != expected_sha256:
        raise ValueError("Predictions differ from the recorded experiment")
    return (measured_data, predictions_path)


def replay_with_hourly_models(
    manifest_path, measured_data, predictions_path, output, ssd_blocks, budget
):
    """Run the verified policy and answer its hourly requests for updated scores."""
    manifest = measured_data.manifest
    parent = Path(manifest_path).resolve().parent
    inputs = manifest["replay"]
    block_ids_path = resolve_and_verify_input_file(inputs["block_ids"], parent)
    prefix = str(block_ids_path).removesuffix("_ids.i32")
    for key, suffix in [("block_counts", "_nb.i32"), ("arrival_times", "_ts.i64")]:
        if resolve_and_verify_input_file(inputs[key], parent) != Path(prefix + suffix):
            raise ValueError("Stream filenames must share one prefix")
    tokens = resolve_and_verify_input_file(inputs["tokens"], parent)
    expected_evictions = resolve_and_verify_input_file(
        inputs["measured_evictions"], parent
    )
    if expected_evictions.stat().st_size != measured_data.eviction_count * 24:
        raise ValueError("Measured eviction stream differs from model rows")
    cutoff_times_path = output / "cuts.i64"
    (np.arange(72, 168, dtype="<i8") * HOUR).tofile(cutoff_times_path)
    shared_replay_arguments = [
        "--prefix",
        prefix,
        "--tokens",
        tokens,
        "--nb",
        inputs["block_id_count"],
        "--hbm",
        81380,
        "--ssd",
        ssd_blocks,
        "--budget",
        budget,
        "--eval-start-us",
        72 * HOUR,
        "--stop-us",
        168 * HOUR,
    ]
    command = [
        BUILD / "recorded_ssd_replay",
        *shared_replay_arguments,
        "--end-us",
        168 * HOUR,
        "--mode",
        "quantile",
        "--seed",
        17,
        "--out-prefix",
        output / "native",
        "--expected-evictions",
        expected_evictions,
        "--scores",
        predictions_path,
        "--cuts-i64",
        cutoff_times_path,
        "--reprice-stdio",
        "--reserve",
        0.15,
        "--release-us",
        120 * HOUR,
        "--update-every",
        1000,
        "--history-size",
        20000,
        "--demand-multiplier",
        128,
        "--count-gain",
        32,
    ]
    hourly_updates = 0
    result = None
    with subprocess.Popen(
        list(map(str, command)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            for line in process.stdout:
                event = json.loads(line)
                if event["event"] == "result":
                    result = event
                    continue
                cutoff_us = (72 + hourly_updates) * HOUR
                if event["event"] != "reprice" or event["cut_us"] != cutoff_us:
                    raise ValueError("Unexpected hourly update")
                rows = np.asarray(event["ids"], dtype=np.int64)
                if np.any(rows < 0) or np.any(
                    rows >= measured_data.find_first_eviction_at_or_after(cutoff_us)
                ):
                    raise ValueError("History contains a future eviction")
                hour_output = output / "models" / str(72 + hourly_updates)
                fit_summary_file = hour_output / "model_fit_summary.json"
                if not fit_summary_file.exists():
                    fit_summary_file = hour_output / "FIT.json"
                record = json.loads(fit_summary_file.read_text())
                if sha256_file(hour_output / "model.pkl") != record["model_sha256"]:
                    raise ValueError("Saved hourly model changed")
                with (hour_output / "model.pkl").open("rb") as file:
                    model = pickle.load(file)
                values, _ = predict_reuse_for_evictions(measured_data, model, rows)
                response = dict(
                    event="reprice_scores",
                    cut_us=cutoff_us,
                    ids=event["ids"],
                    scores=values.tolist(),
                )
                process.stdin.write(json.dumps(response, allow_nan=False) + "\n")
                process.stdin.flush()
                hourly_updates += 1
            if process.wait() or hourly_updates != 96 or result is None:
                raise ValueError("Replay did not finish all 96 model updates")
        except BaseException:
            process.kill()
            process.wait()
            raise
    write_json(output / "native.json", result)
    checker_output = subprocess.check_output(
        list(
            map(
                str,
                [
                    BUILD / "check_replay",
                    *shared_replay_arguments,
                    "--program-bits",
                    output / "native.programs.bits",
                    "--out-prefix",
                    output / "audit",
                ],
            )
        )
    )
    audit = json.loads(checker_output)
    for key, value in audit.items():
        if key != "schema" and result[key] != value:
            raise ValueError("Independent replay mismatch: " + key)
    if sha256_file(output / "native.rows.i64") != sha256_file(
        output / "audit.rows.i64"
    ):
        raise ValueError("Per-request accounting differs")
    write_json(output / "audit.json", audit)
    credit_records = inputs["credit"]
    paths = {
        name: resolve_and_verify_input_file(record, parent)
        for name, record in credit_records.items()
    }
    credit_directory = paths["offsets"].parent
    for name, filename in [
        ("offsets", "offsets.i64"),
        ("bits", "credit.bits"),
        ("ranks", "credit.rank"),
    ]:
        if paths[name] != credit_directory / filename:
            raise ValueError("Credit arrays must use the documented filenames")
    reference = resolve_and_verify_input_file(inputs["reference_rows"], parent)
    denominator, saved, gained, lost = count_eligible_reads_saved_and_compare(
        output / "native.rows.i64", reference, credit_directory
    )
    summary = dict(
        eligible_after_hbm=denominator,
        saved_positions=saved,
        saved_percent=100 * saved / denominator,
        writes=result["writes"],
        budget=budget,
        gained_vs_reference=gained,
        lost_vs_reference=lost,
    )
    for key, expected in manifest.get("expected_result", {}).items():
        if summary[key] != expected:
            raise ValueError("Recorded result differs: " + key)
    write_json(output / "RESULT.json", summary)
    return summary


def main():
    """Run one supplied fixed SSD configuration; do not search for settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssd-blocks", type=int, required=True)
    parser.add_argument("--budget-blocks", type=int, required=True)
    args = parser.parse_args()
    if args.ssd_blocks <= 0 or args.budget_blocks < 0:
        parser.error("Positive capacity and nonnegative allowance required")
    args.output.mkdir(parents=True, exist_ok=False)
    measured_data, predictions_path = train_hourly_eviction_models(
        args.manifest, args.output
    )
    print(
        json.dumps(
            replay_with_hourly_models(
                args.manifest,
                measured_data,
                predictions_path,
                args.output,
                args.ssd_blocks,
                args.budget_blocks,
            )
        )
    )


if __name__ == "__main__":
    main()
