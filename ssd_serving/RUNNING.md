# SSD serving setup

Use Linux, a compatible CUDA and vLLM setup, local Qwen3-30B-A3B-Instruct-2507 BF16 weights and a local SSD formatted as ext4. The storage code writes directly to SSD and waits for each write. It expects the 48-layer, 16-token KV block layout described in the [HBF setup](../hbf_serving/RUNNING.md). This public copy has been checked on CPU.

## Save and restore

Run both examples from the repo root, one at a time on the same GPU. Give each its own new output directory.

```bash
export RANCHO_MODEL_PATH=/path/to/local/model
python ssd_serving/run.py \
  --settings ssd_serving/examples/fixture_accept.json \
  --ssd-directory /path/to/local/ssd --output /path/to/new/accept-output
python ssd_serving/run.py \
  --settings ssd_serving/examples/fixture_reject.json \
  --ssd-directory /path/to/local/ssd --output /path/to/new/reject-output
python ssd_serving/code/check_pair.py \
  /path/to/new/accept-output /path/to/new/reject-output /path/to/check-report.json
```

One run saves eligible blocks up to the budget. The other skips every write. The comparison checks that bringing KV back from SSD still gives the same model output. It checks the bytes copied and uses vLLM's batch-invariant mode to keep the output comparable.

The engine saves full input blocks through a CPU buffer. The SSD file is temporary and is removed when the store closes. Warm-up writes count against the allowance.

The launcher saves the inputs to `serving_inputs.json`. `cell/serving_settings.json` is the file the SSD connector reads. The comparison script also reads older runs saved as `cell/plan.json`.

## Custom policy

The settings file names a `policy_factory` as `module:function`. The factory takes `policy_settings` and returns a function that takes an [Offer](code/ssd_policy.py). Return `True` to save the block or `False` to skip it.

The offer includes block history, the most recent request's features and the write budget. A trained predictor needs the same feature order and user encoding it was trained with. The bundled accept and reject rules exercise this interface.

## CPU checks

```bash
PYTHONPATH=ssd_serving/code:hbf_serving/code python -m unittest discover -s ssd_serving/tests
```

These check example inputs, policy decisions and the protection of blocks in use. For the offline algorithm, see the [SSD simulator](../ssd_simulator/README.md).
