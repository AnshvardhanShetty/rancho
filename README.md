# RANCHO: write-budget-aware KV-cache admission

**Research in progress · Targeting OSDI '27**

RANCHO is an **online cache admission algorithm** for storing LLM key-value (KV) caches in **high-bandwidth flash (HBF)** under a write-endurance constraint. It combines a learned reuse predictor with an adaptive write-budget controller to decide which KV to admit to flash.

LLM servers cache attention keys and values to avoid recomputing earlier input tokens. This KV occupies scarce GPU **high-bandwidth memory (HBM)**. HBF could provide more capacity and reduce the HBM needed for KV storage, but repeated writes wear out flash. The problem is to preserve cache hits while limiting the volume written.

## Algorithm

1. **Estimate reuse.** A gradient-boosted tree predictor estimates later reads per offered KV block from four request features: user ID, turn number, session age and time since the previous request. Training uses only reuse observed before each model update.
2. **Score the admission candidate.** When a request misses in the cache, the server recomputes the missing suffix. RANCHO scores that suffix using predicted reuse, observed reuse history and its write cost.
3. **Enforce the write budget.** An adaptive controller estimates future write demand from recent offers and sets a score cutoff based on the remaining allowance. Accepted KV is written to HBF; rejected candidates consume no flash-write budget.

## Implementation

- **Offline simulator:** replay timestamped requests through HBM and HBF caches, apply admission and eviction, and count cache hits and charged writes.
- **vLLM integration:** run the algorithm during actual model serving. vLLM handles inference and scheduling; RANCHO decides whether to retain newly computed KV in HBF. Later requests reuse admitted KV through the cache lookup path.

The vLLM prototype **emulates HBF in a separate GPU-memory region**, with writes capped at 8 GB/s. Reads use GPU-memory speed, so these measurements do not characterize physical HBF hardware. The [serving code](serving/README.md) explains the integration.

This public version includes the predictor, simulator, serving integration and synthetic examples. It differs from the private working version as development continues. Private traces, trained production models and raw run logs are not included.

## Results so far

In offline replay, the following HBM replacement points kept the combined HBM + HBF hit rate above 94%, within the write allowance:

| HBF capacity per unit of HBM removed | Write limit (DWPD) | HBM replaced | Hit rate |
| --- | ---: | ---: | ---: |
| 8:1 | 0.3 | 25% | 94.47% |
| 8:1 | 1 | 66⅔% | 94.19% |
| 8:1 | 3 | 95% | 96.52% |
| 16:1 | 0.3 | 50% | 94.37% |
| 16:1 | 1 | 85% | 96.75% |
| 16:1 | 3 | 100% | 98.74% |

DWPD means drive writes per day. Hit rate is the fraction of eligible within-session KV reads served by HBM or HBF. These results are pooled over three seeds on one FreeInference workload used during development; each row uses its recorded study configuration. They measure cache hits in offline replay, rather than serving throughput.

This public repository is a source snapshot and differs from our private working version, where the research and experiments are still being developed. The synthetic demo explains the pipeline; it does not reproduce the full evaluation above. Restricted trace data, trained production models, private experiment settings, and detailed run records are not distributed here.

## vLLM serving

We also connect RANCHO to [vLLM](https://github.com/vllm-project/vllm), which runs the language model and schedules requests. When vLLM computes missing KV, RANCHO decides which of it to keep in the flash tier so later requests can reuse it. This prototype emulates HBF in GPU memory, with a limit on write bandwidth.

On a synthetic workload at **20 requests/s**, RANCHO replaced **85% of KV-cache HBM** and served **429.9 output tokens/s**, compared with **67.6** for the HBM-only baseline. Both served **95.12%** of eligible KV reads from cache. This was one repetition with a baseline that retains shared prefixes and emulated HBF reads at GPU-memory speed. The [serving README](serving/README.md) explains how it works and shows the comparison. **The integration is still under development.**

## Read the code

| Order | File | What it does |
| --- | --- | --- |
| 1 | [experiments/trace_reader.py](experiments/trace_reader.py) | Read JSONL requests and save their block lists, sessions, and times in arrival order. |
| 2 | [experiments/run_experiment.py](experiments/run_experiment.py) | Build the four features and provide the loop that runs successive hourly models. Inputs are supplied by the caller. |
| 3 | [model/model_training.py](model/model_training.py) | Load checked reuse counts, prepare earlier examples, fit trees, predict, and save one hourly update. |
| 4 | [model/tree_encoding.py](model/tree_encoding.py) | Give user IDs consistent category codes during fitting and prediction. |
| 5 | [experiments/run_simulation.py](experiments/run_simulation.py) | Start the compiled simulator and read its output counters. |
| 6 | [simulator/rancho.cpp](simulator/rancho.cpp) | Read the cached prefix, price and score the suffix, decide where to store it, and count hits and writes. |

The predictor uses user ID, turn number, session age, and time since the previous request. At each training cutoff, labels include only reuse observed before that cutoff. Its output is one predicted average reads-per-offered-block value per request. The simulator determines the actual suffix using its own cache contents.

Supporting simulator code:

- [write_controller.hpp](simulator/write_controller.hpp): estimate affordable write demand from recent offers and choose score cutoffs.
- [cache.hpp](simulator/cache.hpp): maintain least-recently-used storage order and calculate exact trial-write cost.
- [hbm_only_readable_prefix.cpp](simulator/hbm_only_readable_prefix.cpp): run a separate HBM-only comparison to filter the history bonus.
- [simulation_io.hpp](simulator/simulation_io.hpp): load supplied settings and arrays, then save decisions, cache snapshots, and counters.

`count_session_hits()` counts eligible within-session reads served by the readable prefix. `add_metrics()` adds those counts to the totals. The hit rate is (HBF hits + HBM hits) divided by eligible within-session reads.

## Run the synthetic demo

Requires Python 3.11+, Make, and a C++17 compiler on little-endian Linux or macOS. On macOS, LightGBM also requires `libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make demo
```

The demo trains on earlier synthetic sessions, checks feature preparation through the JSONL reader, and replays [three synthetic requests](examples/demo_requests.json). The first writes two blocks. The next two each read both from flash. Expected: **4 flash hits, 0 HBM hits, and 2 blocks written**.

This checks the pipeline, not predictor quality or performance on a real workload. The demo constructs its small synthetic training labels directly; the model code also supports the checked hourly count format used by the offline label-building tools.

## Other source

- [model/](model/README.md): predictor, four-feature settings, encoding, and file helpers.
- [reproduction_support/](reproduction_support/README.md): offline source used to construct and independently check offered-suffix reuse counts. Restricted inputs and archived files are not included.
- [tests/](tests/README.md): deterministic synthetic regression cases. Run `make test-simulator` to check them.

The demo does not fetch trace data or trained models and requires no lab-server access. The real-data experiment launchers, private paths, and detailed result files are not part of this public snapshot.
