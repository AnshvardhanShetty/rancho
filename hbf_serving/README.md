# HBF in vLLM

This is my HBF integration with vLLM, the engine that runs the language model. RANCHO decides which KV to move out of ordinary GPU memory, called **HBM**, so the engine has more space for active requests.

vLLM finds the cached beginning of a prompt and computes the KV it is missing. RANCHO scores that new KV and checks the write budget. When it accepts a write, the worker copies the KV to HBF, updates its cache location and releases the original HBM space.

This prototype uses a separate region of GPU memory to stand in for HBF. Reads run at GPU-memory speed. Writes are limited to **8 GB/s**. The serving integration is under development.

## Run the demo

Use one CUDA GPU and local Qwen3-30B-A3B-Instruct-2507 BF16 weights. The recorded
run used vLLM 0.27.1 and PyTorch 2.13.0 with 48 layers and 16-token KV blocks.
Install the Python dependencies in that environment with
`pip install -r hbf_serving/requirements.txt`.

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python hbf_serving/run.py --arm hbm --output /path/to/new/baseline-output
python hbf_serving/run.py --arm rancho16 --predictor /path/to/saved/predictor --output /path/to/new/rancho-output
```

The HBF run takes the saved four-feature predictor fitted at hour 72. Its folder
contains `model.pkl` and `model_fit_summary.json` (or `COMPLETE.json` for an older
fit). Set `synthetic_user` in [synthetic_setup.json](synthetic_setup.json) to the
user value the predictor expects. For a self-contained CPU example that trains
its own predictor, run `make demo-serving`.

The demo runs 1,200 requests across eight chats at 20 requests per second. Each
run writes hit rate, throughput, latency and flash writes to `cell/RESULT.json`.

## Code

| File | What it does |
| --- | --- |
| [run.py](run.py) | Pick a cache setup and start the demo |
| [run_serving.py](code/run_serving.py) | Start vLLM, send requests and report results |
| [direct_backend.py](code/direct_backend.py) | Connect the scheduler to the worker and set up the HBM-only baseline |
| [direct_backend_hbf.py](code/direct_backend_hbf.py) | Apply the write limit and copy accepted KV to HBF |
| [hbf_admission.py](code/hbf_admission.py) | Decide what to write using the reuse model and write budget |

[synthetic_requests.py](code/synthetic_requests.py) builds the prompts and model
inputs. [reuse_predictor.py](code/reuse_predictor.py) loads the model.
[write_controller.py](code/write_controller.py) sets cutoffs and tracks writes.
[tier_pool.py](code/tier_pool.py) manages page ownership, and
[native_pages.py](code/native_pages.py) reads and copies vLLM's pages.
[metrics.py](code/metrics.py) and [run_files.py](code/run_files.py) handle timing
and file I/O.
