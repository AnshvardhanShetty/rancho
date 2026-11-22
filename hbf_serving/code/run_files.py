"""Read and write the settings and records saved by a serving run"""

import json
from pathlib import Path


def read_run_json(output_directory, filename, earlier_filename):
    """Read new or earlier serving output folders"""
    path = output_directory / filename
    if not path.exists():
        path = output_directory / earlier_filename
    return json.loads(path.read_text())


def write_run_json(path, record):
    """Write JSON to a temporary file, then replace the destination."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
