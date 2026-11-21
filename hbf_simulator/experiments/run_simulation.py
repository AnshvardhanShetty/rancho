"""Run the C++ cache replay and read back its counts.

The caller supplies the prepared request files, predictions, cache sizes and
write budget. run_cache_simulation passes them to the simulator for one seed,
then reads metrics.json after it exits.
"""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text())


def verify_file_hash(path, expected_sha256):
    with path.open("rb") as input_file:
        if hashlib.file_digest(input_file, "sha256").hexdigest() != expected_sha256:
            raise ValueError(f"Input checksum mismatch: {path}")


def run_program(program_name, *args):
    subprocess.run(
        [str(ROOT / "build" / program_name), *map(str, args)],
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
    # Keep the argument order expected by parse_simulation_settings in C++.
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
