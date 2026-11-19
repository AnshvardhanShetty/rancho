# RANCHO

I’m leading the development of RANCHO as first author supervised by Prof. Juncheng Yang at Harvard SEAS. RANCHO is an algorithm for reducing GPU memory use and prefill recomputation in LLM serving to increase throughput and lower time to first token (TTFT).

I carried out the system design, implementation, and experiments, including the reuse predictor, write-budget controller, cache simulator, and vLLM integration. The manuscript is in preparation, targeting OSDI '27.

The research is ongoing. This repository provides a limited public demo so those interested can explore my implementation and run selected components on synthetic workloads. The full research implementation and evaluation remain private and under development.

## How it works

Language models use GPU memory to process requests and keep the **KV cache**, which stores information already computed about earlier tokens. Later requests can reuse that KV to skip prompt processing, called **prefill**. Keeping more KV saves computation but leaves less memory for active requests.

HBF (High-bandwidth flash) stacks NAND flash near the processor for high-bandwidth reads. Sandisk and SK hynix target 8–16 times HBM’s capacity at comparable read bandwidth and similar cost. 

A solid-state drive (SSD) uses NAND flash in a separate device accessed through the computer’s storage connection.

Both forms of flash storage can store prefix-KV, freeing space in HBM to serve more active requests concurrently and improving TTFT and throughput, but write-allowance limits how much KV can be written to these systems. 

RANCHO is an online KV admission algorithm that predicts which cached state will be most valuable to preserve and decides what to write to flash under a limited endurance budget. It combines a gradient-boosted tree that predicts future KV reuse with an adaptive write controller that adjusts admission thresholds based on expected write demand and the remaining budget.

- **High-bandwidth flash (HBF)** stores selected KV after prefill, freeing GPU memory for active requests.
- **Solid-state drives (SSDs)** save selected KV as it gets evicted from memory (HBM or DRAM), so later requests can load it back instead of computing it again.

```mermaid
flowchart LR
    A[Computed KV] --> B[Predict later reuse]
    B --> C[Decide what is worth writing]
    C --> D[Save selected KV in flash]
    D --> E[Reuse it on later requests]
```

## Results

The simulator results use a private FreeInference trace with **1.49 million requests over seven days**. The trace is not included in this repository.

### High-bandwidth flash

In simulator experiments, RANCHO replaced **85% of the GPU memory used for prefix caching** while meeting a 95% cache-hit target (±1 percentage point).

| Policy | KV-cache GPU memory replaced |
| --- | ---: |
| **RANCHO** | **85%** |
| Random: save each request with 50% probability | 20% |
| Access Count: save after three accesses | 40% |

Access Count waits until the first missing block has been accessed three times before saving the recomputed KV. Each GB of GPU memory removed adds 16 GB of flash. All policies have a write allowance averaging one full flash-capacity write per day.

I also integrated RANCHO into **vLLM**, an engine for serving language models. The prototype uses a separate region of GPU memory to emulate HBF and limits writes to **8 GB/s**. We test it with synthetic conversations whose shared prefixes occupy most of the baseline's KV-cache memory, leaving little room for active requests.

With **85% less KV-cache HBM**, RANCHO achieves **6.4× throughput** at the same 95.12% cache-hit ratio. The time by which 95% of requests receive their first token (TTFT) falls **89%**. [vLLM code and setup](hbf_serving/README.md).

### SSD

SSD caching avoids **79.5% of the repeated prefill left after memory caching** within conversations in our simulator experiments, using 48 GB of memory and a 2.304 TB SSD at one drive write per day. The memory capacity stays fixed. [SSD simulator](ssd_simulator/README.md).

The HBF serving engine is still under development. We are also developing an SSD serving engine; its initial code is in [ssd_serving](ssd_serving/README.md).

## Try it without a GPU

Run the synthetic conversations from the vLLM example through the cache simulator on your laptop. The code trains RANCHO on earlier requests, then shows how much KV later requests reuse and how much it writes to flash.

Requires Python 3.11+, Make and a C++17 compiler. On macOS, install `libomp` for LightGBM.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make demo-serving
```

The demo processes 1,200 requests across eight conversations. It prints cache-hit ratios, GPU memory savings, recomputation and flash writes. In this example, RANCHO maintains a **95.12% cache-hit ratio with 85% less KV-cache HBM**.

For a three-request walkthrough, run `make demo`. [Example code](hbf_simulator/examples/README.md) · [Run with vLLM on a GPU](hbf_serving/RUNNING.md).

## Read the code

| Folder | What it contains |
| --- | --- |
| [hbf_simulator](hbf_simulator/README.md) | HBF cache simulation, reuse predictor and CPU examples. |
| [ssd_simulator](ssd_simulator/README.md) | SSD cache simulation, model training and recomputation counts. |
| [hbf_serving](hbf_serving/README.md) | HBF integration with vLLM. Under development. |
| [ssd_serving](ssd_serving/README.md) | SSD integration with vLLM. Under development. |

Run `make test-simulator` for the cache tests and `make test-demo` for the CPU example checks.
