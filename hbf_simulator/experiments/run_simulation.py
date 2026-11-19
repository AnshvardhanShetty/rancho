"""Start the simulator with caller-supplied files, capacities, and write allowance.

No trace data or lab locations are bundled. run_cache_simulation() returns the
hit/write counters saved by the C++ program; it does not train models.
"""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text())


def verify_file_hash(path, digest):
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != digest:
            raise ValueError(f"Input checksum mismatch: {path}")


def run_program(name, *args):
    subprocess.run(
        [str(ROOT / "build" / name), *map(str, args)],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def run_cache_simulation(
    stream, n, flash, ram, zero, eligible, scores, seed, budget, start, end, shadow, out
):
    """Run one seed with its saved predictions and read the simulator’s counters."""
    run_program(
        "rancho",
        stream,
        n,
        flash,
        ram,
        zero,
        zero,
        budget,
        start,
        end,
        "v1",
        out,
        "--live-rancho",
        seed,
        eligible,
        scores,
        "--reuse-bonus",
        "--cost-bands",
        "--half-behind",
        "--shadow",
        "shadow-linear",
        shadow,
    )
    return read_json(out / "metrics.json")["evaluation"]
