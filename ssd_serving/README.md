# SSD in vLLM

This code saves selected KV when it leaves GPU memory. If a later request needs that KV, vLLM can read it from SSD instead of computing it again.

1. When the engine needs to reuse a GPU cache slot, the connector checks whether the old block can be saved. Active requests keep their blocks.
2. The policy uses the block's history, the last request's information and the remaining write allowance to decide. An existing SSD copy can be reused without another write.
3. The worker writes accepted blocks to SSD. A later request can copy the saved blocks back into GPU memory.

The integration is under development. The included examples accept or reject writes so we can check that restoring KV gives the same output. The learned SSD policy is in the [CPU simulator](../ssd_simulator/README.md).

| File | What it does |
| --- | --- |
| [run.py](run.py) | Start one example using the model and SSD directory you supply |
| [code/serve.py](code/serve.py) | Send the requests and save their outputs and response times |
| [code/ssd_backend.py](code/ssd_backend.py) | Catch blocks before eviction and restore SSD hits before use |
| [code/ssd_policy.py](code/ssd_policy.py) | Define what the decision function receives |
| [code/ssd_storage.py](code/ssd_storage.py) | Store blocks in a local SSD file and read them back |
| [code/check_pair.py](code/check_pair.py) | Compare the output of the accept and reject runs |

See [RUNNING.md](RUNNING.md) for setup, commands and how to supply a policy.
