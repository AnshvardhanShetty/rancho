"""Read and write the settings and records saved by a serving run"""

import json
import os
from pathlib import Path


def write_run_json(path, record):
    """Write JSON to a temporary file, then replace the destination."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_model_path():
    """Return the local model directory from RANCHO_MODEL_PATH."""
    path = Path(os.environ["RANCHO_MODEL_PATH"]).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("RANCHO_MODEL_PATH must name an existing model directory")
    return str(path)
