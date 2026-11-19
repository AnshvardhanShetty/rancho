# RANCHO: KV caching with high-bandwidth flash

**Research in progress · Targeting OSDI '27**

When a language model processes a conversation, it saves intermediate results called **key-value data (KV)**. The next message can reuse that data instead of computing the whole conversation again. KV takes up the GPU's limited **high-bandwidth memory (HBM)**, leaving less room to serve other requests.

We study whether **high-bandwidth flash (HBF)** can replace some of that KV-cache memory. Flash adds storage capacity, but writing everything to it would wear it out. **RANCHO decides which KV is worth saving, so we can reuse it while staying within a write budget.**

## How it works

1. **Predict reuse.** A small tree model learns from earlier requests. It uses user ID, turn number, conversation age and time since the previous request to estimate how often the KV will be read again.
2. **Choose what to write.** RANCHO combines that prediction with earlier reuse and the amount of data to write. Its controller adjusts the admission cutoff using recent write demand and the remaining budget.
3. **Reuse the saved KV.** In our [vLLM integration](serving/README.md), vLLM first uses any cached beginning of the input and computes what is missing. RANCHO then decides which newly computed KV to copy to HBF. Later requests can use those copies instead of recomputing them.

The repository includes two ways to study this: an **offline simulator** that replays request traces and counts hits and writes, and a **vLLM prototype** that runs an actual language model and measures throughput and latency. The prototype emulates HBF in a separate GPU-memory region, limits write bandwidth, and uses GPU-memory speed for reads. It does not measure physical HBF hardware.

This public version differs from our private working version as development continues. It includes source code and synthetic examples, without private traces, trained production models or raw run logs.

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
