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
from .training_data import TrainingData, pinned
from .training import fit_model, predict_events
from .settings import SETTINGS, HOUR
from .accounting import credit_and_overlap

HERE = Path(__file__).resolve().parent
BUILD = HERE.parents[1] / "build/ssd"


def save(path, value):
    """Save one JSON record, rejecting duplicate stage outputs."""
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare(manifest_path, output):
    """Verify supplied arrays and train all 96 hourly raw-target models."""
    training = TrainingData(manifest_path)
    cold = TrainingData(manifest_path, cold=True)
    models = output / "models"
    models.mkdir()
    scores = output / "test_raw.f64"
    with scores.open("xb") as stream:
        for hour in range(72, 168):
            cut = hour * HOUR
            model, record = fit_model(training, cut, SETTINGS)
            if model["kind"] != "tree" or record["max_train_decision_us"] >= cut:
                raise ValueError("Expected a fitted causal tree")
            first, stop = (
                cold.first_eviction_at(cut),
                cold.first_eviction_at(cut + HOUR),
            )
            folder = models / str(hour)
            folder.mkdir()
            with (folder / "model.pkl").open("xb") as file:
                pickle.dump(model, file, protocol=5)
            for start in range(first, stop, 200000):
                rows = np.arange(start, min(start + 200000, stop))
                values, _ = predict_events(cold, model, rows)
                values.astype("<f8").tofile(stream)
            record.update(
                first=first,
                stop=stop,
                model_sha256=sha(folder / "model.pkl"),
                training_target="historical marginal credited prefix unlock",
            )
            save(folder / "FIT.json", record)
    if scores.stat().st_size != cold.n * 8:
        raise ValueError("One score per measured eviction is required")
    expected = training.manifest.get("expected_scores_sha256")
    if expected and sha(scores) != expected:
        raise ValueError("Predictions differ from the recorded experiment")
    return (cold, scores)


def replay(manifest_path, cold, scores, output, ssd_blocks, budget):
    """Run the verified policy and answer its hourly requests for updated scores."""
    manifest = cold.manifest
    parent = Path(manifest_path).resolve().parent
    inputs = manifest["replay"]
    blocks = pinned(inputs["block_ids"], parent)
    prefix = str(blocks).removesuffix("_ids.i32")
    for key, suffix in [("block_counts", "_nb.i32"), ("arrival_times", "_ts.i64")]:
        if pinned(inputs[key], parent) != Path(prefix + suffix):
            raise ValueError("Stream filenames must share one prefix")
    tokens = pinned(inputs["tokens"], parent)
    expected_evictions = pinned(inputs["measured_evictions"], parent)
    if expected_evictions.stat().st_size != cold.n * 24:
        raise ValueError("Measured eviction stream differs from model rows")
    cuts = output / "cuts.i64"
    (np.arange(72, 168, dtype="<i8") * HOUR).tofile(cuts)
    common = [
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
        *common,
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
        scores,
        "--cuts-i64",
        cuts,
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
    count = 0
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
                cut = (72 + count) * HOUR
                if event["event"] != "reprice" or event["cut_us"] != cut:
                    raise ValueError("Unexpected hourly update")
                rows = np.asarray(event["ids"], dtype=np.int64)
                if np.any(rows < 0) or np.any(rows >= cold.first_eviction_at(cut)):
                    raise ValueError("History contains a future eviction")
                folder = output / "models" / str(72 + count)
                record = json.loads((folder / "FIT.json").read_text())
                if sha(folder / "model.pkl") != record["model_sha256"]:
                    raise ValueError("Saved hourly model changed")
                with (folder / "model.pkl").open("rb") as file:
                    model = pickle.load(file)
                values, _ = predict_events(cold, model, rows)
                response = dict(
                    event="reprice_scores",
                    cut_us=cut,
                    ids=event["ids"],
                    scores=values.tolist(),
                )
                process.stdin.write(json.dumps(response, allow_nan=False) + "\n")
                process.stdin.flush()
                count += 1
            if process.wait() or count != 96 or result is None:
                raise ValueError("Replay did not finish all 96 model updates")
        except BaseException:
            process.kill()
            process.wait()
            raise
    save(output / "native.json", result)
    checked = subprocess.check_output(
        list(
            map(
                str,
                [
                    BUILD / "check_replay",
                    *common,
                    "--program-bits",
                    output / "native.programs.bits",
                    "--out-prefix",
                    output / "audit",
                ],
            )
        )
    )
    audit = json.loads(checked)
    for key, value in audit.items():
        if key != "schema" and result[key] != value:
            raise ValueError("Independent replay mismatch: " + key)
    if sha(output / "native.rows.i64") != sha(output / "audit.rows.i64"):
        raise ValueError("Per-request accounting differs")
    save(output / "audit.json", audit)
    credit_records = inputs["credit"]
    paths = {name: pinned(record, parent) for name, record in credit_records.items()}
    credit_directory = paths["offsets"].parent
    for name, filename in [
        ("offsets", "offsets.i64"),
        ("bits", "credit.bits"),
        ("ranks", "credit.rank"),
    ]:
        if paths[name] != credit_directory / filename:
            raise ValueError("Credit arrays must use the documented filenames")
    reference = pinned(inputs["reference_rows"], parent)
    denominator, saved, gained, lost = credit_and_overlap(
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
    save(output / "RESULT.json", summary)
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
    cold, scores = prepare(args.manifest, args.output)
    print(
        json.dumps(
            replay(
                args.manifest,
                cold,
                scores,
                args.output,
                args.ssd_blocks,
                args.budget_blocks,
            )
        )
    )


if __name__ == "__main__":
    main()
