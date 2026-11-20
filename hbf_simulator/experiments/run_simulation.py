"""Pass prepared inputs to the C++ cache simulator

The caller supplies the request files and predictions
The simulator saves the hit and write counts and this file reads them back
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
    request_stream_prefix,
    block_id_count,
    hbf_capacity,
    hbm_capacity,
    zero_flags_path,
    eligible_requests_path,
    predictions_path,
    seed,
    write_budget,
    start_us,
    end_us,
    hbm_only_prefix_path,
    output_directory,
):
    """Run one seed with its saved predictions and read the simulator’s counters."""
    run_program(
        "rancho",
        request_stream_prefix,
        block_id_count,
        hbf_capacity,
        hbm_capacity,
        zero_flags_path,
        zero_flags_path,
        write_budget,
        start_us,
        end_us,
        "v1",
        output_directory,
        "--live-rancho",
        seed,
        eligible_requests_path,
        predictions_path,
        "--reuse-bonus",
        "--cost-bands",
        "--half-behind",
        "--shadow",
        "shadow-linear",
        hbm_only_prefix_path,
    )
    return read_json(output_directory / "metrics.json")["evaluation"]
