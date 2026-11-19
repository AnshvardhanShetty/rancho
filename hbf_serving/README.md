# RANCHO with vLLM

This is my integration of RANCHO into vLLM. It connects the reuse predictor and write-budget controller to the engine's KV cache, so accepted KV is copied to HBF and can be reused by later requests.

The serving engine is still under development.

## How it works

When a request arrives, vLLM finds its cached prefix and computes the missing KV during prefill. RANCHO predicts later reuse and decides whether to save the computed KV in HBF. After an accepted copy, the engine updates the request's cache locations and releases the original HBM space.

The prototype emulates HBF in a separate region of GPU memory. Reads use GPU-memory speed, and writes share an **8 GB/s limit**. The predictor runs on the CPU.

## Experiment

We use eight synthetic conversations with shared prefixes. The baseline keeps these prefixes in HBM, occupying most of its KV-cache memory and leaving little space to process requests together. Moving the prefixes to HBF lets the engine process more requests at once.

Requests arrive at **20 per second** for 60 seconds. Each has 1,872 shared input tokens, 96 unique input tokens and 64 output tokens. We run eight warm-up requests before the 1,200 measured requests.

| Configuration | KV-cache HBM replaced | Output tokens/s | Time to first token, p95 | Cache-hit ratio |
| --- | ---: | ---: | ---: | ---: |
| HBM only | — | 67.6 | 1,020.2 s | 95.12% |
| RANCHO, 16:1 | 85% | 429.9 | 109.6 s | 95.12% |

This gives **6.4× throughput and 89% lower p95 time to first token**. Throughput counts generated tokens over the time needed to finish all requests. Time to first token includes queueing; p95 is the time by which 95% of requests have received their first token. The benchmark uses Qwen3-30B-A3B-Instruct-2507 on one GPU, with one repetition. [Commands and settings](RUNNING.md).

## Code

| File | What it does |
| --- | --- |
| [run.py](run.py) | Start a serving run. |
| [code/run_cell.py](code/run_cell.py) | Send requests and measure cache hits, throughput and latency. |
| [code/direct_backend.py](code/direct_backend.py) | Connect vLLM's scheduler to the two KV storage regions. |
| [code/direct_backend_hbf.py](code/direct_backend_hbf.py) | Ask RANCHO what to save, copy accepted KV and update cache locations. |
| [code/updated_policy.py](code/updated_policy.py) | Predict reuse and make the admission decision. |
| [code/physical_hbf.py](code/physical_hbf.py) | Track write credits as time passes. |

For setup and run commands, see [RUNNING.md](RUNNING.md). The [CPU demo](../hbf_simulator/examples/README.md) runs the synthetic workload through the cache simulator.
