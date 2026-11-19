# Running with vLLM

The GPU runner uses one GPU, local Qwen3-30B-A3B-Instruct-2507 BF16 weights and a compatible CUDA environment. The recorded setup used vLLM 0.27.1 and PyTorch 2.13.0. The integration uses vLLM internals, so other versions may need changes. This public snapshot has been checked on CPU; it has not been rerun on a GPU.

## GPU commands

From the repository root, set the model location and choose a new output directory for each run:

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python serving/run.py --arm hbm --output /path/to/new/baseline-output
python serving/run.py --arm rancho16 --predictor /path/to/saved/predictor --output /path/to/new/rancho-output
```

The RANCHO predictor folder must contain `model.pkl` and `COMPLETE.json` from the four-feature offered-suffix fit at hour 72. [updated_model.py](code/updated_model.py) checks the feature names, cutoff, tree count and file hashes before loading the native trees. Supply the user encoding expected by that predictor in `synthetic_setup.json`. The GPU runner keeps the predictor fixed while updating reuse history and the controller.

For a self-contained local run that trains its own predictor, use `make demo-serving` from the repository root.

## Benchmark setup

The synthetic workload has eight conversations. Each prompt contains 1,872 shared tokens and 96 unique tokens; the model generates 64 output tokens. Eight warm-up requests fill the shared prefixes. Then 1,200 measured requests arrive at 20 requests per second over 60 seconds.

The HBM-only baseline retains the shared prefixes. They occupy 936 of its 960 KV pages. RANCHO uses 144 HBM pages and 13,056 HBF pages at 16:1. At 8:1, it uses 320 HBM pages and 5,120 HBF pages.

HBF is emulated in GPU memory. Writes are synchronous and limited to 8 GB/s. The run starts with a one-device-day write allowance; measured writes also need credits earned since measurement began. Warm-up writes count towards the total allowance.

Throughput includes the time needed to finish queued requests. First-token latency includes queueing. Warm-up is excluded from these timings. The [saved measurements](synthetic_results.json) contain one repetition on one RTX PRO 6000 Blackwell Max-Q GPU.

## CPU checks

```bash
PYTHONPATH=serving/code python -m unittest discover -s serving/tests
```

These check block ownership, write budgets, request accounting and timing calculations. The native vLLM queue check is skipped when vLLM is unavailable.
