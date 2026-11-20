"""Use model weights supplied by the caller; no download is performed."""

import os
from pathlib import Path


def resolve_model_path():
    """Return the local model directory from RANCHO_MODEL_PATH."""
    path = Path(os.environ["RANCHO_MODEL_PATH"]).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("RANCHO_MODEL_PATH must name an existing model directory")
    return str(path)
