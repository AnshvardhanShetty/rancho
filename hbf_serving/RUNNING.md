# Run the HBF serving example

The runner uses one GPU with CUDA and local Qwen3-30B-A3B-Instruct-2507 BF16 weights. The recorded setup used vLLM 0.27.1 and PyTorch 2.13.0. The code uses vLLM's internal cache interfaces, which can change between versions. This public copy has been checked on CPU.

## Run on a GPU

From the repo root, set the model path and choose a new output directory for each run:

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python hbf_serving/run.py --arm hbm --output /path/to/new/baseline-output
python hbf_serving/run.py --arm rancho16 --predictor /path/to/saved/predictor --output /path/to/new/rancho-output
```

The RANCHO run needs a predictor folder with `model.pkl` and `COMPLETE.json` from the four-input offered-suffix fit at hour 72. [updated_model.py](code/updated_model.py) checks the inputs, cutoff, tree count and checksum. Set the user value in [synthetic_setup.json](synthetic_setup.json) to the value your predictor expects. The predictor stays fixed during the run. Reuse history and the cutoff continue to update.

For a self-contained CPU run that trains its own predictor, use `make demo-serving`.

## What the benchmark does

Eight warm-up requests compute the shared prefixes. Then 1,200 measured requests arrive over 60 seconds. Each has 1,872 shared tokens and 96 unique input tokens. The model generates 64 output tokens per request.

The baseline keeps 936 pages of shared prefixes in its 960-page HBM cache. At 16:1, RANCHO uses 144 HBM pages and 13,056 HBF pages. At 8:1, it uses 320 HBM pages and 5,120 HBF pages.

HBF is emulated in GPU memory. Writes wait for the copy to finish and are limited to 8 GB/s. Warm-up writes use part of the full allowance. During measurement, writes also need credits earned from elapsed time.

Throughput includes the time to finish queued requests. Time to first token includes queueing. Both timings exclude warm-up. The [saved results](synthetic_results.json) are from one repetition on one RTX PRO 6000 Blackwell Max-Q GPU.

## CPU checks

```bash
PYTHONPATH=hbf_serving/code python -m unittest discover -s hbf_serving/tests
```

These check page ownership, spending and request timing. The native queue check needs vLLM.
