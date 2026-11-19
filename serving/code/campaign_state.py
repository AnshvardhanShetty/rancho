"""Save a JSON record without exposing a partly written file."""
import json
from pathlib import Path

def save(path, obj):
    """Write JSON to a temporary file, then replace the destination."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
