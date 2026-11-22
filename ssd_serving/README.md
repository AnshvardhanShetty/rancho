# SSD in vLLM

The demo saves KV blocks to SSD when vLLM needs to reuse their GPU slots.
When a later request matches a saved block, the worker loads it back into
GPU memory. Writes stop when the budget runs out.

## Run the demo

Use Linux with CUDA, a compatible vLLM install, local Qwen3-30B-A3B-Instruct-2507
BF16 weights, and a local SSD with an ext4 filesystem. The demo expects the
model's 48-layer cache layout with 16 tokens per block.

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python ssd_serving/run.py --ssd-directory /path/to/ssd --output /path/to/new/ssd-output
```

To run the same requests with SSD writes off, add `--skip-writes` and pick a
new output directory. Look in `cell/io.jsonl` for writes and restores and
`cell/requests.json` for the generated tokens. The SSD file is temporary.

The demo uses a simple budget check. The learned policy is in the
[SSD simulator](../ssd_simulator/README.md).

## Code

| File | What it does |
| --- | --- |
| [run.py](run.py) | Set up the demo and start vLLM in a fresh process |
| [code/serve.py](code/serve.py) | Send the requests and save their outputs and response times |
| [code/ssd_backend.py](code/ssd_backend.py) | Catch blocks before eviction and restore SSD hits before use |
| [code/ssd_storage.py](code/ssd_storage.py) | Store blocks in a local SSD file and read them back |
| [demo_workload.json](demo_workload.json) | Set the cache sizes, write budget, and request inputs |
