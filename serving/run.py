"""Run one synthetic serving comparison arm on a user-selected GPU.

Assumes compatible vLLM/CUDA and local model weights. RANCHO also needs a
four-feature saved predictor. This script does not select or reserve a GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from campaign_state import save
from updated_model import SuffixGBT, CUT


def main():
    """Prepare one arm's inputs, run its engine, then check the saved result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["hbm", "rancho8", "rancho16"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup", type=Path, default=HERE / "synthetic_setup.json")
    parser.add_argument("--predictor", type=Path)
    args = parser.parse_args()
    setup = json.loads(args.setup.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hbm, ratio = {"hbm": (960, 0), "rancho8": (320, 8), "rancho16": (144, 16)}[args.arm]
    hbf = ratio * (setup["baseline_pages"] - hbm)
    job = dict(arm=args.arm, hbm=hbm, hbf=hbf, ratio=ratio,
               retain_prefixes=not bool(hbf), rate=20, pilot=False,
               repetition=setup["repetition"], controller=None)
    if hbf:
        if args.predictor is None:
            parser.error("RANCHO needs --predictor pointing to COMPLETE.json and model.pkl")
        predictor = args.predictor.resolve()
        spec = dict(cut_us=CUT, directory=str(predictor),
                    receipt_sha256=hashlib.sha256((predictor / "COMPLETE.json").read_bytes()).hexdigest())
        model = SuffixGBT(spec)
        controller = output / "controller.json"
        save(controller, dict(cold=True, observations=0, paid_pages=0, fit=spec,
                              model_sha256=model.model_sha256))
        job.update(fit=spec, controller=str(controller))
    save(output / "setup.json", setup)
    save(output / "job.json", job)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE / "code") + os.pathsep + env.get("PYTHONPATH", "")
    runner = HERE / "code/run_cell.py"
    subprocess.run([sys.executable, str(runner), "--setup", str(output / "setup.json"),
                    "--job", str(output / "job.json"), "--output", str(output / "cell")],
                   env=env, check=True)
    subprocess.run([sys.executable, str(runner), "--audit", str(output / "cell")], env=env, check=True)


if __name__ == "__main__":
    main()
