# Running RANCHO in vLLM

This is my HBF integration with vLLM, the engine that runs the language model. RANCHO decides which KV to move out of ordinary GPU memory, called **HBM**, so the engine has more space for active requests.

vLLM finds the cached beginning of a prompt and computes the KV it is missing. RANCHO scores that new KV and checks the write budget. When it accepts a write, the worker copies the KV to HBF, updates its cache location and releases the original HBM space.

This prototype uses a separate region of GPU memory to stand in for HBF. Reads run at GPU-memory speed. Writes are limited to **8 GB/s**. The serving integration is under development.

## Follow the code

| File | What it does |
| --- | --- |
| [run.py](run.py) | Choose the HBM-only or HBF run and start it |
| [code/run_cell.py](code/run_cell.py) | Send requests and measure hits, throughput and latency |
| [code/direct_backend.py](code/direct_backend.py) | Connect the scheduler to the cache and the GPU worker |
| [code/direct_backend_hbf.py](code/direct_backend_hbf.py) | Ask RANCHO what to write and move accepted KV |
| [code/updated_policy.py](code/updated_policy.py) | Use the predictor, reuse history and write budget to decide |
| [code/physical_hbf.py](code/physical_hbf.py) | Limit writes using time and the allowance |

## Synthetic run

Eight conversations share prefixes that use most of the baseline's GPU cache. At 20 requests per second, the smaller HBM cache with HBF leaves more room for requests to run together.

| Configuration | HBM replaced | Output tokens/s | p95 time to first token | KV hit rate |
| --- | ---: | ---: | ---: | ---: |
| HBM only | — | 67.6 | 1,020.2 s | 95.12% |
| RANCHO at 16:1 | 85% | 429.9 | 109.6 s | 95.12% |

Here p95 means the time by which 95% of requests got their first token. These are measurements from one run on one GPU. See [commands and setup](RUNNING.md) or try the [same prompts on CPU](../hbf_simulator/examples/README.md).
