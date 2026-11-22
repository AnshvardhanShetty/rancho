"""Run the C++ cache replay and read back its counts.

The caller supplies the prepared request files, predictions, cache sizes and
write budget. run_cache_simulation passes them to the simulator for one seed,
then reads metrics.json after it exits.
"""

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json_file(file_path):
    """Read a JSON file into Python."""
    return json.loads(file_path.read_text())


def run_program(program_name, *arguments):
    subprocess.run(
        [str(REPO_ROOT / "build" / program_name), *map(str, arguments)],
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

    return load_json_file(output_directory / "metrics.json")["evaluation"]
