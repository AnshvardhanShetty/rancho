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
from campaign_state import write_run_json
from updated_model import CUT, SuffixGBT, model_fit_summary_path


def main():
    """Prepare one arm's inputs, run its engine, then check the saved result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["hbm", "rancho8", "rancho16"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload-settings",
        "--setup",
        type=Path,
        default=HERE / "synthetic_setup.json",
    )
    parser.add_argument("--predictor", type=Path)
    args = parser.parse_args()
    workload_settings = json.loads(args.workload_settings.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hbm_pages, ratio = {"hbm": (960, 0), "rancho8": (320, 8), "rancho16": (144, 16)}[
        args.arm
    ]
    hbf_pages = ratio * (workload_settings["baseline_pages"] - hbm_pages)
    comparison_settings = dict(
        arm=args.arm,
        hbm=hbm_pages,
        hbf=hbf_pages,
        ratio=ratio,
        retain_prefixes=not bool(hbf_pages),
        rate=20,
        pilot=False,
        repetition=workload_settings["repetition"],
        controller=None,
    )
    if hbf_pages:
        if args.predictor is None:
            parser.error(
                "RANCHO needs --predictor pointing to a folder with model.pkl and its fit summary"
            )
        predictor = args.predictor.resolve()
        predictor_settings = dict(
            cut_us=CUT,
            directory=str(predictor),
            receipt_sha256=hashlib.sha256(
                model_fit_summary_path(predictor).read_bytes()
            ).hexdigest(),
        )
        model = SuffixGBT(predictor_settings)
        controller_start_state = output / "controller_start_state.json"
        write_run_json(
            controller_start_state,
            dict(
                cold=True,
                observations=0,
                paid_pages=0,
                fit=predictor_settings,
                model_sha256=model.model_sha256,
            ),
        )
        comparison_settings.update(
            fit=predictor_settings, controller=str(controller_start_state)
        )
    workload_settings_file = output / "workload_settings.json"
    comparison_settings_file = output / "comparison_settings.json"
    write_run_json(workload_settings_file, workload_settings)
    write_run_json(comparison_settings_file, comparison_settings)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE / "code") + os.pathsep + env.get("PYTHONPATH", "")
    runner = HERE / "code/run_cell.py"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--workload-settings",
            str(workload_settings_file),
            "--comparison-settings",
            str(comparison_settings_file),
            "--output",
            str(output / "cell"),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(runner), "--audit", str(output / "cell")],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
