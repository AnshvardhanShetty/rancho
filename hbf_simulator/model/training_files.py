"""Read model records and check which files a run used"""

import hashlib
import json
from pathlib import Path


# Load a JSON settings or record file.
def read_json(path):
    return json.loads(Path(path).read_text())


def find_training_file(directory, filename, archived_filename):
    """Find a training file in a new run or an archived run."""
    path = directory / filename
    return path if path.exists() else directory / archived_filename


# Save a JSON record; exclusive mode refuses to overwrite a file.
def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as json_file:
        json.dump(value, json_file)


# Hash the file contents so later stages can check they have the same input.
def file_fingerprint(path):
    with Path(path).open("rb") as input_file:
        return hashlib.file_digest(input_file, "sha256").hexdigest()
