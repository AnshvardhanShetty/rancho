# Running the SSD examples

The GPU examples require Linux, compatible vLLM/CUDA, local Qwen3-30B-A3B-Instruct-2507 BF16 weights and a local SSD with an ext4 filesystem. The code uses direct I/O and synchronous writes. It expects the same 48-layer, 16-token KV layout as the HBF integration. See the [vLLM setup](../hbf_serving/RUNNING.md). This public snapshot has been checked on CPU; it has not been rerun on a GPU.

## Save and restore KV

Run these sequentially on the same GPU and model, choosing a new output directory for each:

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python ssd_serving/run.py --plan ssd_serving/examples/fixture_accept.json --ssd-directory /path/to/local/ssd --output /path/to/new/accept-output
python ssd_serving/run.py --plan ssd_serving/examples/fixture_reject.json --ssd-directory /path/to/local/ssd --output /path/to/new/reject-output
python ssd_serving/code/check_pair.py /path/to/new/accept-output /path/to/new/reject-output /path/to/check-report.json
```

One example saves every eligible block within the budget; the other rejects every write. The comparison checks that restored KV produces the same output. Byte verification is enabled. The launcher uses vLLM's batch-invariant mode for the output comparison.

The engine creates a temporary SSD file and removes it when closed. Only complete input-KV blocks are saved. Transfers pass through a CPU buffer, and writes count against the allowance, including warm-up.

## Connect an admission policy

Set `policy_factory` in the plan to `module:function`. The factory receives `policy_settings` and returns a function that accepts an [Offer](code/ssd_policy.py). Return `True` to save the block or `False` to skip it.

The offer contains prior accesses and evictions, time in memory, time since the last access, features from the last request and the remaining write budget. Supply request features and user encoding consistently with the predictor. The included accept/reject policies check the storage interface; the research SSD predictor is supplied separately.

## CPU checks

```bash
PYTHONPATH=ssd_serving/code:hbf_serving/code python -m unittest discover -s ssd_serving/tests
```

These check the example plans, policy interface, write limits and protection of active KV using stand-ins for GPU interfaces. For a CPU cache replay, see [ssd/replay](../ssd_simulator/replay/README.md).
