"""File helpers used by the hourly predictor in the artifact."""

import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


# Load a JSON settings or record file.
def read_json(path):
    return json.loads(Path(path).read_text())


# Save a JSON record; exclusive mode refuses to overwrite a file.
def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(value, handle)


# Hash the file contents so later stages can check they have the same input.
def file_fingerprint(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
