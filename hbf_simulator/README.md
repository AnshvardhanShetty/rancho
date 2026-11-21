# HBF cache simulator

The simulator uses block IDs to track the cache. It runs on a CPU and does not run the language model.

From the repo root, run `make demo` for three requests or `make demo-serving` for the larger synthetic example. Both are explained in [examples](examples/README.md).

## Follow one request

1. Python prepares the requests in arrival order and predicts how often their KV will be read again.
2. The simulator checks how much of each request it can read from the caches. It calls the part starting at the first missing block the **suffix**.
3. RANCHO scores the suffix and checks whether the write is worth making and fits the budget.
4. The simulator updates the caches and records hits and writes.

| Start here | What to look for |
| --- | --- |
| [experiments](experiments/README.md) | How Python reads requests and starts the replay |
| [model](model/README.md) | What the model learns and what it predicts |
| [rancho.cpp](rancho.cpp) | `process_request()` puts the request steps together |
| [write_controller.hpp](write_controller.hpp) | How recent offers and remaining budget set the cutoff |
| [cache.hpp](cache.hpp) | How a cache stores block IDs and removes the least recently used block |
| [hbm_only_readable_prefix.cpp](hbm_only_readable_prefix.cpp) | What GPU memory alone could serve for each request |
| [simulation_io.hpp](simulation_io.hpp) | What files the simulator reads and saves |

[Label tools](reproduction_support/README.md) show how to build the training counts from a trace. [Tests](tests/README.md) show what the public examples check.
