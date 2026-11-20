"""Read model records and check which files a run used"""

import hashlib
import json
from pathlib import Path

MODEL_DIRECTORY = Path(__file__).resolve().parent
SIMULATOR_DIRECTORY = MODEL_DIRECTORY.parent


# Load a JSON settings or record file.
def read_json(path):
    return json.loads(Path(path).read_text())


def read_training_json(directory, filename, earlier_filename):
    """Use the descriptive filename if present or read an older training folder"""
    path = directory / filename
    if not path.exists():
        path = directory / earlier_filename
    return read_json(path)


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
