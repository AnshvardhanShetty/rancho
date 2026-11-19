"""Train RANCHO and replay the synthetic serving prompts on the CPU.

Uses the serving generator's earlier requests, warm-up and cache capacities.
Requests are processed in arrival order with a total one-device-day write
allowance. GPU scheduling, decoding and elapsed-time credits belong to the
separate serving runner."""

import argparse
import array
from collections import OrderedDict
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hbf_serving/code"))
from controlled_workload import block_keys, eligibility, prompts, records, tokens
from features import RequestHistory
from synthetic_workload import FIT

sys.path.insert(0, str(ROOT / "hbf_simulator/experiments"))
from run_simulation import run_cache_simulation, run_program
from synthetic_training import (
    SyntheticTrainingData, initial_predictions, fit_research_model, model_training,
)


def write_array(path, kind, values):
    """Write the little-endian arrays expected by the existing C++ replay."""
    data = array.array(kind, values)
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def make_stream(setup, folder):
    """Use shared content hashes to identify blocks, including their prefix."""
    payload = prompts(setup)
    # These references establish eligibility; they do not populate either cache.
    history = [tokens(setup, i) for i in range(setup["max_reference_requests"])]
    history += [tokens(setup, i, calibration=True)
                for i in range(setup["calibration_requests"])]
    vocabulary = {}
    block_rows = []
    for prompt in history + payload:
        block_rows.append([
            vocabulary.setdefault(key, len(vocabulary)) for key in block_keys(prompt)
        ])
    history_count = len(history)
    split = 1_000_000
    warmup = setup["groups"]
    measured_begin = split + warmup
    times = list(range(-history_count, 0))
    times += [split + i for i in range(warmup)]
    times += [measured_begin + round(i * 1_000_000 / 20)
              for i in range(setup["requests"])]
    sessions = [i % warmup for i in range(setup["max_reference_requests"])]
    sessions += [10_000 + i for i in range(setup["calibration_requests"])]
    sessions += [r["session"] for r in records(setup)[1]]
    names = json.loads((ROOT / "hbf_simulator/model/model_settings.json").read_text())["features"]
    history_clock = RequestHistory()
    gap = setup["logical_gap_seconds"] * 1_000_000
    policy_times = [FIT["cut_us"] - (history_count - i) * gap
                    for i in range(history_count)]
    policy_times += [r["policy_us"] for r in records(setup)[1]]
    feature_rows = []
    for i, now in enumerate(policy_times):
        info = history_clock.arrival(i, now, sessions[i], setup["synthetic_user"])
        feature_rows.append([info.shared()[name] for name in names])
    features = np.array(feature_rows, dtype=float)
    expected_features = np.array([[r["features"][name] for name in names]
                                 for r in records(setup)[1]], dtype=float)
    if not np.array_equal(features[history_count:], expected_features):
        raise RuntimeError("Features differ from the serving workload generator")
    stream = folder / "requests"
    write_array(Path(str(stream) + "_nb.i32"), "i", map(len, block_rows))
    write_array(Path(str(stream) + "_sess.i32"), "i", sessions)
    write_array(Path(str(stream) + "_ts.i64"), "q", times)
    write_array(Path(str(stream) + "_ids.i32"), "i",
                (block for row in block_rows for block in row))
    write_array(folder / "zero", "B", [0] * len(times))
    write_array(folder / "eligible", "B", [1] * len(times))
    return dict(stream=stream, block_count=len(vocabulary), start=split,
                end=times[-1] + 1, history_count=history_count,
                rows=block_rows[history_count:], eligibility=eligibility(setup),
                history_rows=block_rows[:history_count],
                history_times=policy_times[:history_count],
                history_sessions=sessions[:history_count], features=features)


def fit_predictor(setup, data, folder, hbm, hbf):
    """Collect suffix offers using the arm's capacities and one-capacity allowance."""
    settings = json.loads((ROOT / "hbf_simulator/model/model_settings.json").read_text())
    n = data["history_count"]
    names = settings["features"]
    initial = SyntheticTrainingData(data["history_rows"], data["history_times"],
                                    data["history_sessions"], data["features"][:n], names)
    predicted = initial_predictions(initial, settings)
    stream = folder / "training"
    write_array(Path(str(stream) + "_nb.i32"), "i", map(len, data["history_rows"]))
    write_array(Path(str(stream) + "_sess.i32"), "i", data["history_sessions"])
    write_array(Path(str(stream) + "_ts.i64"), "q", data["history_times"])
    write_array(Path(str(stream) + "_ids.i32"), "i",
                (block for row in data["history_rows"] for block in row))
    write_array(folder / "training_zero", "B", [0] * n)
    write_array(folder / "training_eligible", "B", [1] * n)
    write_array(folder / "training_predictions", "d", predicted)
    shadow = folder / "training_shadow"
    start = data["history_times"][0]
    run_program("hbm_only_readable_prefix", stream, data["block_count"], hbm,
                start, FIT["cut_us"], shadow)
    output = folder / "training_replay"
    run_cache_simulation(stream, data["block_count"], hbf, hbm,
                         folder / "training_zero", folder / "training_eligible",
                         folder / "training_predictions", setup["repetition"], hbf,
                         start, FIT["cut_us"], shadow, output)
    observed = np.fromfile(output / "rows.i64", dtype="<i8").reshape(-1, 10)
    if len(observed) != n:
        raise RuntimeError("Training replay omitted earlier requests")
    offers = [blocks[int(row[3]):]
              for blocks, row in zip(data["history_rows"], observed)]
    final = SyntheticTrainingData(data["history_rows"], data["history_times"],
                                  data["history_sessions"], data["features"][:n],
                                  names, offered=offers)
    model, record = fit_research_model(final, FIT["cut_us"], settings)
    scores = [0.0] * n + model_training.predict_reuse(model, data["features"][n:]).tolist()
    write_array(folder / "predictions", "d", scores)
    _, labels = final.training_examples(FIT["cut_us"])
    return dict(training_examples=len(labels), unique_labels=len(np.unique(labels)),
                model_kind=model["kind"], tree_count=record["tree_count"])


def retained_baseline(setup, data):
    """Retain computed shared prefixes as the serving HBM-only baseline does."""
    pinned = set()
    transient = OrderedDict()
    hits = fetched = positions = eligible = 0
    shared = setup["prefix_tokens"] // 16
    for i, blocks in enumerate(data["rows"]):
        first = 0
        for block in blocks:
            if block not in pinned and block not in transient:
                break
            if block in transient:
                transient.move_to_end(block)
            first += 1
        pinned.update(blocks[:shared])
        for block in pinned:
            transient.pop(block, None)
        for block in blocks[first:]:
            if block not in pinned:
                transient[block] = True
                transient.move_to_end(block)
        if len(pinned) > setup["baseline_pages"]:
            raise ValueError("Shared prefixes exceed baseline HBM capacity")
        while len(pinned) + len(transient) > setup["baseline_pages"]:
            transient.popitem(last=False)
        if i >= setup["groups"]:
            positions += len(blocks)
            fetched += first
            eligible += sum(data["eligibility"][i])
            hits += sum(data["eligibility"][i][:first])
    return dict(policy="HBM only", hbm_blocks=setup["baseline_pages"],
                hbf_blocks=0, hbm_replaced_percent=0.0, eligible_accesses=eligible,
                cache_hits=hits, hit_ratio_percent=100 * hits / eligible,
                prefill_blocks_recomputed=positions - fetched,
                flash_blocks_written=0, write_allowance_blocks=0)


def replay_arm(setup, data, folder, hbm, ratio):
    """Run the real C++ admission controller and count measured request outputs."""
    hbf = ratio * (setup["baseline_pages"] - hbm)
    training = fit_predictor(setup, data, folder, hbm, hbf)
    shadow = folder / "shadow"
    run_program("hbm_only_readable_prefix", data["stream"], data["block_count"],
                hbm, data["start"], data["end"], shadow)
    out = folder / "replay"
    counters = run_cache_simulation(
        data["stream"], data["block_count"], hbf, hbm,
        folder.parent / "zero", folder.parent / "eligible",
        folder / "predictions", setup["repetition"], hbf,
        data["start"], data["end"], shadow, out,
    )
    rows = np.fromfile(out / "rows.i64", dtype="<i8").reshape(-1, 10)
    rows = rows[data["history_count"] + setup["groups"]:]
    if len(rows) != setup["requests"]:
        raise RuntimeError("Measured request count differs from the workload")
    hits = eligible = 0
    for i, row in enumerate(rows):
        mask = data["eligibility"][i + setup["groups"]]
        eligible += sum(mask)
        hits += sum(mask[:int(row[3])])
        if int(row[4] + row[5]) != sum(mask[:int(row[3])]):
            raise RuntimeError("Independent prefix recount differs from C++ hits")
    writes = counters["writes"]
    if writes > hbf:
        raise RuntimeError("Flash write allowance exceeded")
    return dict(policy=f"RANCHO {ratio}:1", hbm_blocks=hbm, hbf_blocks=hbf,
                hbm_replaced_percent=100 * (1 - hbm / setup["baseline_pages"]),
                eligible_accesses=eligible, cache_hits=hits,
                hit_ratio_percent=100 * hits / eligible,
                prefill_blocks_recomputed=int(sum(row[2] - row[3] for row in rows)),
                flash_blocks_written=writes, write_allowance_blocks=hbf,
                training=training)


def run_demo():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the computed summary as JSON")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("--output must name a new file")
    setup = json.loads((ROOT / "hbf_serving/synthetic_setup.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        data = make_stream(setup, folder)
        results = [retained_baseline(setup, data)]
        for hbm, ratio in [(320, 8), (144, 16)]:
            arm = folder / f"rancho{ratio}"
            arm.mkdir()
            results.append(replay_arm(setup, data, arm, hbm, ratio))
    print("CPU cache replay of the bundled synthetic serving prompts")
    print(f"{setup['groups']} warm-up requests; {setup['requests']} measured requests")
    print("Research training pipeline fitted on earlier requests from the same generator.")
    print(f"{'Policy':<16} {'HBM blocks':>11} {'HBM saved':>11} {'Cache hits':>12} "
          f"{'Recomputed':>12} {'Flash writes':>13}")
    for row in results:
        print(f"{row['policy']:<16} {row['hbm_blocks']:>11} "
              f"{row['hbm_replaced_percent']:>10.1f}% "
              f"{row['hit_ratio_percent']:>11.2f}% "
              f"{row['prefill_blocks_recomputed']:>12} "
              f"{row['flash_blocks_written']:>13}")
    print("Writes include warm-up. Hit ratios and recomputation exclude warm-up.")
    print("Prompt-cache replay only; GPU scheduling and write timing are not simulated.")
    if args.output:
        summary = dict(workload="bundled synthetic serving prompts",
                       predictor="research pipeline fitted on earlier synthetic requests",
                       mode="sequential prompt-cache replay", results=results)
        with args.output.open("x") as handle:
            handle.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    run_demo()
