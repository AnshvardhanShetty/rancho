# RANCHO: HBF admission simulator and reuse predictor

RANCHO decides which KV blocks to write to flash under a limited write allowance. A four-feature tree predicts later reuse. The simulator combines that prediction with observed reuse, write cost, and the remaining budget to make admission decisions.

This private sharing copy contains source code and synthetic examples. It contains no production trace, saved production models, archived predictions, or measured research results. Lab-specific reproduction settings and the original repository history are excluded.

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

The demo does not fetch trace data or trained models and requires no lab-server access. The real-data experiment launchers, private paths, and measured results are not part of this sharing copy.
