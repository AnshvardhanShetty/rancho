# Running the SSD examples

The GPU path requires Linux, compatible vLLM/CUDA, local Qwen3-30B-A3B-Instruct-2507 BF16 weights, and enough GPU memory for the model. It expects the same 48-layer, 16-token KV-block layout as the HBF serving path. See the [HBF environment notes](../serving/RUNNING.md). This public packaging has CPU checks; it has not been rerun on a GPU.

Choose an available GPU and a directory on a local SSD. The storage code currently requires an ext4 filesystem backed by a local block device. It bypasses the operating system's file cache and waits for each write to complete. The directory is supplied at launch; no lab paths or machine names are built in. The runner does not reserve a GPU or stop other jobs.

## Run the two storage examples

From the repository root:

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python ssd/run.py --plan ssd/examples/fixture_accept.json --ssd-directory /path/to/local/ssd --output /path/to/new/accept-output
python ssd/run.py --plan ssd/examples/fixture_reject.json --ssd-directory /path/to/local/ssd --output /path/to/new/reject-output
python ssd/code/check_pair.py /path/to/new/accept-output /path/to/new/reject-output /path/to/check-report.json
```

Each output directory must be new. The engine uses a temporary SSD file that it removes when closed. Run the examples sequentially on the same device and model. They check the storage path using synthetic token IDs, not a real conversation trace. Byte verification adds overhead and is enabled for these correctness examples.

The launcher enables vLLM's batch-invariant mode so the output comparison is meaningful. The engine uses one synchronous process for scheduling and GPU execution. Other engine configurations need their own validation.

## Supply a decision function

The plan's `policy_factory` has the form `module:function`. The factory receives `policy_settings` and returns a function taking an `Offer` from [ssd_policy.py](code/ssd_policy.py). Return `True` to save a block or `False` to skip it.

An offer includes the block's past accesses and evictions, time in GPU memory, time since its last access, and the inputs from the last request that used it. Request features and user-ID encoding must be supplied consistently with the chosen predictor. The engine does not train a model or reconstruct those request features itself.

The included accept/reject functions check the interface. The production SSD predictor and its model/controller binding are not distributed in these examples. Do not interpret the examples as an evaluation of the learned SSD policy.

Writes consume the declared block allowance, including warm-up writes. Host write counts do not measure additional writes inside the SSD. Only full input-KV blocks are saved. Transfers use a CPU staging buffer and are synchronous. The code depends on vLLM's internal interfaces and may need changes for other versions.

## CPU checks

```bash
PYTHONPATH=ssd/code:serving/code python -m unittest discover -s ssd/tests
```

These checks cover the synthetic plans, decision interface, hard write limit and protection of active or unfinished KV. The eviction checks use stand-ins for GPU interfaces. They do not test real SSD transfers or GPU execution.
