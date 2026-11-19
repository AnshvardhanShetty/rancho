"""Run one SSD correctness example on a caller-selected GPU and SSD.

Assumes a compatible Linux vLLM/CUDA environment and local model weights.
This launcher does not reserve hardware or reproduce a performance benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "serving/code"


def main():
    """Supply the local SSD path and launch a fresh engine process."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", type=Path, default=HERE / "examples/fixture_accept.json"
    )
    parser.add_argument("--ssd-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    plan["ssd_directory"] = str(args.ssd_directory.resolve(strict=True))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configured = output / "input.json"
    configured.write_text(json.dumps(plan, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE / "code"), str(SHARED), env.get("PYTHONPATH", "")]
    )
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "code/serve.py"),
            str(configured),
            str(output / "cell"),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
