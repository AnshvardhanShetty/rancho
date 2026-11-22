"""Run the SSD demo on a caller-selected GPU and SSD.

Assumes a compatible Linux vLLM/CUDA environment and local model weights.
Both modes use the same requests.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHARED_SERVING_CODE = HERE.parent / "hbf_serving/code"


def main():
    """Supply the local SSD path and launch a fresh engine process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-writes", action="store_true", help="Run with SSD writes off")
    parser.add_argument("--ssd-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    serving_settings = json.loads((HERE / "demo_workload.json").read_text())
    serving_settings["save_to_ssd"] = not args.skip_writes
    serving_settings["ssd_directory"] = str(args.ssd_directory.resolve(strict=True))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    serving_inputs_file = output / "serving_inputs.json"
    serving_inputs_file.write_text(json.dumps(serving_settings, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE / "code"), str(SHARED_SERVING_CODE), env.get("PYTHONPATH", "")]
    )
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "code/serve.py"),
            str(serving_inputs_file),
            str(output / "cell"),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
