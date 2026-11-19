# HBF serving with vLLM

The serving code lets vLLM keep KV in two regions: HBM and an emulated HBF tier. RANCHO decides which newly computed KV to save in HBF.

## What happens to a request

1. **Find cached KV.** The scheduler reads the available prefix from either tier. vLLM computes the missing KV in GPU memory.
2. **Decide what to save.** Once that computation finishes, RANCHO predicts reuse from the four request features. It scores the remaining suffix using that prediction, earlier reuse and write cost. The controller checks its cutoff and remaining write allowance.
3. **Copy admitted KV.** The worker copies accepted full pages to the HBF region. Copies share an 8 GB/s write limit. The scheduler publishes a copy only after it finishes, so later requests cannot read unfinished data.
4. **Serve later requests.** Cached HBF pages participate in vLLM's native prefix lookup and attention. Pages already present need no new write. Both tiers track ownership so pages in use cannot be recycled.

HBF is **emulated in GPU memory**. Reads therefore use GPU-memory speed; this does not measure physical HBF hardware. The integration uses vLLM scheduler and connector extensions, without editing the installed vLLM package.

## Synthetic workload result

At **20 requests/s**, we used eight shared contexts, 1,872 shared input tokens and 96 unique input tokens per request, with 64 output tokens. After eight serial warm-up requests, we sent 1,200 requests over 60 seconds and waited for the queue to finish.

| Configuration | HBM replaced | Output tokens/s | p95 time to first token | KV hit rate |
| --- | ---: | ---: | ---: | ---: |
| HBM only | — | 67.6 | 1,020.2 s | 95.12% |
| RANCHO, 8:1 | 66.7% | 923.1 | 20.0 s | 95.12% |
| RANCHO, 16:1 | 85% | 429.9 | 109.6 s | 95.12% |

These are **one repetition on a constructed workload**, using Qwen3-30B-A3B-Instruct-2507 in BF16 on one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition. They do not establish performance across production workloads. The HBM-only baseline retains the declared shared prefixes after computing them; it is not stock vLLM eviction. Those prefixes occupy 936 of its 960 usable KV pages. The 8:1 arm has 320 HBM + 5,120 HBF pages; the 16:1 arm has 144 HBM + 13,056 HBF pages.

Throughput includes the time needed to drain the queue. Time to first token includes queueing from scheduled arrival. Warm-up is excluded from these timings. Hit rate uses eligible within-session reads, with the same synthetic reference history across arms. The saved headline values are in [synthetic_results.json](synthetic_results.json).

Warm-up writes count against the same device-day allowance. Measured writes additionally require credits earned from real elapsed time, starting at zero. The startup burst is not a sustainable 1-DWPD write rate. This short test does not establish flash endurance.

**This integration and the research are still under development.** The public source differs from the private working version. Traces, trained production predictors, model weights and private run logs are not included.

## Read the code

| File | Purpose |
| --- | --- |
| [run.py](run.py) | Prepare a synthetic comparison arm, run it, then audit its output. |
| [code/run_cell.py](code/run_cell.py) | Start vLLM, send requests, count native cache hits and measure timing. |
| [code/controlled_workload.py](code/controlled_workload.py) | Generate synthetic token IDs and request features. |
| [code/direct_backend.py](code/direct_backend.py) | Connect native vLLM scheduling and KV storage to the two tiers. |
| [code/direct_backend_hbf.py](code/direct_backend_hbf.py) | Offer completed KV to RANCHO and copy admitted pages. |
| [code/updated_policy.py](code/updated_policy.py) | Combine tree predictions, earlier reuse and the write controller. |
| [code/updated_model.py](code/updated_model.py) | Read a supplied four-feature tree and predict reuse. |
| [code/physical_hbf.py](code/physical_hbf.py) | Bind admission to the real clock and available write credit. |
| [code/tier_pool.py](code/tier_pool.py) | Track page ownership and the free-page queues. |

The remaining modules provide page layout checks, baseline prefix retention, metrics and independent decision checks. They use vLLM APIs; no vLLM source is vendored here. vLLM is a separate dependency under its upstream license.

## Run CPU checks

From the repository root, after installing its Python requirements:

```bash
PYTHONPATH=serving/code python -m unittest discover -s serving/tests
```

These checks cover page ownership, write budgets, metrics and synthetic request accounting. Native vLLM queue checks are skipped when vLLM is unavailable. They do not run a GPU benchmark.

## Run a serving arm

Use a compatible CUDA/vLLM environment and enough GPU memory for the BF16 model and both KV regions. The recorded environment used vLLM 0.27.1 and PyTorch 2.13.0. These extensions depend on vLLM internals and may need changes for other versions. The public packaging has been checked on CPU; it has not been rerun on a GPU.

Choose an available GPU yourself. Set `RANCHO_MODEL_PATH` to local Qwen model weights. The runner does not download weights, reserve a device or stop other jobs.

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python serving/run.py --arm hbm --output /path/to/new/baseline-output
python serving/run.py --arm rancho16 --predictor /path/to/saved/predictor --output /path/to/new/rancho-output
```

Each output folder must be new. A RANCHO predictor folder contains `model.pkl` and `COMPLETE.json` from an hourly four-feature offered-suffix fit at the day-four cutoff. Its receipt must include the feature list, label cutoff, tree count and model digest checked in `updated_model.py`. That loader reads native tree text and numeric category values without executing the saved pickle. Use your own trusted files; an adjacent digest detects inconsistency, not who produced a file. The production predictors behind the table are not distributed. A different predictor is a different experiment; the synthetic setup alone cannot recreate the recorded scores.

The public synthetic user value is a placeholder. Supply a correctly encoded user value for your predictor when preparing a new experiment. Predictor feature timestamps follow the constructed session history; write credits and controller observations use real elapsed time. The serving benchmark uses a frozen predictor; it does not retrain while handling requests.
