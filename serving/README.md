# RANCHO with vLLM

When an LLM answers a request, it computes information about the input tokens. This information is called **key-value data (KV)**. Saving it lets the server reuse the computation when a later request starts with the same tokens—for example, the next message in a conversation.

KV normally occupies the GPU's **high-bandwidth memory (HBM)**. That memory is limited: storing old conversations leaves less room for requests being processed now. We study whether **high-bandwidth flash (HBF)** can hold some of that KV instead. Flash adds capacity, but repeated writes wear it out. We therefore need to choose what is worth saving.

**RANCHO makes that choice.** It predicts how often KV will be reused and decides whether the expected benefit is worth the write, given the remaining write allowance.

## How it works

[vLLM](https://github.com/vllm-project/vllm) runs the language model and schedules requests. This code adds RANCHO and a second place to store KV:

1. **Reuse what is already saved.** When a request arrives, vLLM looks for the beginning of its input in the cache. It computes the part that is missing.
2. **Decide whether to keep the newly computed KV.** RANCHO uses user ID, turn number, conversation age and time since the previous request to predict reuse. It combines that prediction with earlier accesses, the amount to write and the remaining allowance.
3. **Save the selected KV in HBF.** The worker copies the accepted data. A later request can use that copy instead of recomputing it. KV that is not saved in HBF stays subject to the ordinary HBM capacity and eviction limits.

The predictor estimates reuse; the write controller decides how selective to be. As requests arrive, the controller uses recent write offers and spending to adjust its admission cutoff. This serving version uses a frozen trained predictor while updating the controller and reuse history during execution.

**The prototype emulates HBF in a separate region of GPU memory.** Writes are limited to 8 GB/s, but reads run at GPU-memory speed. It lets us exercise the integration with real model execution; it is not a test of physical HBF hardware.

## Results so far

On a synthetic workload with eight shared contexts at **20 requests/s**:

| Configuration | HBM replaced | Output tokens/s | p95 time to first token | KV hit rate |
| --- | ---: | ---: | ---: | ---: |
| HBM only | — | 67.6 | 1,020.2 s | 95.12% |
| RANCHO, 8:1 | 66.7% | 923.1 | 20.0 s | 95.12% |
| RANCHO, 16:1 | 85% | 429.9 | 109.6 s | 95.12% |

“HBM replaced” is the reduction in KV-cache HBM relative to the baseline. At 16:1, each unit removed is replaced with sixteen units of HBF capacity. Output tokens/s measures how quickly the server finishes generating answers. Time to first token includes time spent waiting in the queue; p95 means 95% of requests received their first token within that time.

These are one-repetition measurements on a constructed workload. The baseline keeps the shared prefixes in HBM, where they occupy most of its cache; it does not use stock vLLM eviction. HBF reads are emulated as described above. See [run instructions and benchmark conditions](RUNNING.md) for details.

**The integration and research are still under development.** This public version differs from the private working version. Private traces, trained production predictors, model weights and raw run logs are not included.

## Follow the code

| Start here | What it does |
| --- | --- |
| [run.py](run.py) | Start a synthetic serving test and check its output. |
| [code/run_cell.py](code/run_cell.py) | Send requests to vLLM and measure cache hits, throughput and latency. |
| [code/direct_backend.py](code/direct_backend.py) | Connect vLLM's scheduler and KV storage to the two memory regions. |
| [code/direct_backend_hbf.py](code/direct_backend_hbf.py) | Pass newly computed KV to RANCHO and copy accepted data. |
| [code/updated_policy.py](code/updated_policy.py) | Predict reuse, add earlier reuse history and make the write decision. |
| [code/physical_hbf.py](code/physical_hbf.py) | Enforce write credits earned from real elapsed time. |

The other modules handle storage bookkeeping, synthetic inputs and result checks. vLLM is installed separately; its source is not copied into this repository.

For setup, commands and checks, see [RUNNING.md](RUNNING.md).
