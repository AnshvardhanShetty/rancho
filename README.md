# RANCHO

**Research in progress · Targeting OSDI '27**

RANCHO is an **algorithm that decides which parts of an LLM's KV cache to save in flash, while limiting how much it writes**.

When a language model processes a conversation, it saves intermediate results called **key-value data (KV)**. Later requests can reuse that data instead of repeating the computation. This cache occupies the GPU's **high-bandwidth memory (HBM)**, which has limited capacity. Keeping old conversations in memory leaves less room to process new requests.

We study storing KV in **high-bandwidth flash (HBF)** and **solid-state drives (SSDs)**. Both provide additional storage, but flash wears out as data is written. RANCHO aims to save the KV that will be reused enough to justify the write.

## How the algorithm works

1. **Predict whether the KV will be reused.** The HBF version learns from user ID, turn number, conversation age and time since the previous request. The SSD version also uses how often a block has been read or removed from GPU memory, how long it has been there, and when it was last used.
2. **Decide whether it is worth saving.** It combines that prediction with reuse already observed and the amount of data to write.
3. **Stay within the write allowance.** It adjusts how selective it is based on recent requests and the budget left. Only selected KV is written to flash, where later requests can reuse it.

## What is in this repository

- **A simulator:** read a recorded sequence of requests, simulate the caches, and count how much computation they avoid and how much data they write.
- **An HBF vLLM prototype:** run RANCHO alongside [vLLM](https://github.com/vllm-project/vllm), the software that runs the language model and schedules requests. When vLLM computes missing KV, RANCHO decides what to save for later requests. This lets us measure the effect on serving speed and response time.

- **An SSD engine integration:** when GPU memory needs space, decide whether to save the KV being removed on a solid-state drive (SSD). If a later request needs it, load it back instead of recomputing it. The [SSD code](ssd/README.md) includes small examples that check saving and restoring data. **This engine integration is also under development.**

The HBF prototype represents HBF using a separate region of GPU memory. Writes are limited to 8 GB/s, but reads use GPU-memory speed. **It does not test physical HBF hardware.**

This public version differs from our private working version as development continues. Private traces, trained production models and raw run logs are not included.

## HBF cache results

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

## SSD recomputation results

Our recorded-trace evaluation also shows that SSD offloading can avoid computing KV again after it is no longer in GPU memory. In one tested setting, RANCHO avoided **88% of the eligible KV recomputation left after the GPU cache lookup**, by restoring saved KV from SSD.

This result uses a 48 GB GPU cache, an SSD cache 72 times that size, and a 3-DWPD write allowance. It counts repeated KV reads within the same conversation on one recorded workload; it is not an 88% reduction in response time. The [SSD overview](ssd/README.md) gives the results at different write allowances. **The SSD engine integration is still under development.**

## HBF vLLM serving results

On a synthetic workload at **20 requests/s under high HBM capacity pressure**, RANCHO replaced **85% of the HBM used for KV caching**, with **536% higher output throughput** and **89% lower p95 time to first token** than the HBM-only baseline. Both served **95.12%** of eligible KV reads from cache.

Higher throughput means generating more output in the same time. The p95 time is how long 95% of requests waited for their first output token, including queueing. Here, the baseline keeps shared conversation data in most of its KV-cache memory, leaving little room to process concurrent requests.

These are one-repetition results on a constructed workload, with the HBF emulation described above. They are not a general speedup across workloads. The [serving README](serving/README.md) explains the code and gives the comparison. **The integration is still under development.**

## Read the code

| Order | File | What it does |
| --- | --- | --- |
| 1 | [experiments/trace_reader.py](experiments/trace_reader.py) | Read recorded requests and save their KV block IDs, conversation IDs and arrival times in time order. |
| 2 | [experiments/run_experiment.py](experiments/run_experiment.py) | Build the four predictor inputs and run model updates one hour at a time. |
| 3 | [model/model_training.py](model/model_training.py) | Use earlier requests and their observed reuse to train the predictor and save its predictions. |
| 4 | [model/tree_encoding.py](model/tree_encoding.py) | Represent each user ID consistently during training and prediction. |
| 5 | [experiments/run_simulation.py](experiments/run_simulation.py) | Start the compiled simulator and read its output counters. |
| 6 | [simulator/rancho.cpp](simulator/rancho.cpp) | Check what is cached, decide whether to save the missing KV, and count reads served and data written. |

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
