"""Replay the bundled serving prompts through the CPU cache simulator.

Uses the existing prompt generator, warm-up, capacities and reference eligibility.
The example predictor is fitted on the existing small synthetic training history;
the private serving predictor is not needed or reconstructed. Requests are replayed
in arrival order, with no GPU scheduler or decode execution. The replay enforces
a total one-device-day write allowance, not wall-clock GPU write credits. It
measures cache behavior and cannot measure throughput or latency.
"""

import argparse
import array
from collections import OrderedDict
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving/code"))
from controlled_workload import block_keys, eligibility, prompts, records, tokens

from demo import prepare_example, model_training, run_cache_simulation, run_program


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
    _, _, _, _, model = prepare_example()
    names = json.loads((ROOT / "model/model_settings.json").read_text())["features"]
    features = np.array([[r["features"][name] for name in names]
                         for r in records(setup)[1]], dtype=float)
    predictions = [0.0] * history_count
    predictions += model_training.predict_reuse(model, features).tolist()
    stream = folder / "requests"
    write_array(Path(str(stream) + "_nb.i32"), "i", map(len, block_rows))
    write_array(Path(str(stream) + "_sess.i32"), "i", sessions)
    write_array(Path(str(stream) + "_ts.i64"), "q", times)
    write_array(Path(str(stream) + "_ids.i32"), "i",
                (block for row in block_rows for block in row))
    write_array(folder / "zero", "B", [0] * len(times))
    write_array(folder / "eligible", "B", [1] * len(times))
    write_array(folder / "predictions", "d", predictions)
    return dict(stream=stream, block_count=len(vocabulary), start=split,
                end=times[-1] + 1, history_count=history_count,
                rows=block_rows[history_count:], eligibility=eligibility(setup))


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
    shadow = folder / "shadow"
    run_program("hbm_only_readable_prefix", data["stream"], data["block_count"],
                hbm, data["start"], data["end"], shadow)
    out = folder / "replay"
    counters = run_cache_simulation(
        data["stream"], data["block_count"], hbf, hbm,
        folder.parent / "zero", folder.parent / "eligible",
        folder.parent / "predictions", setup["repetition"], hbf,
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
                flash_blocks_written=writes, write_allowance_blocks=hbf)


def run_demo():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the computed summary as JSON")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("--output must name a new file")
    setup = json.loads((ROOT / "serving/synthetic_setup.json").read_text())
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
    print("Predictor fitted on the bundled synthetic example history.")
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
                       predictor="bundled synthetic example fit",
                       mode="sequential prompt-cache replay", results=results)
        with args.output.open("x") as handle:
            handle.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    run_demo()
