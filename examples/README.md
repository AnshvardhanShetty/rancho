# Synthetic examples

## Serving prompts on CPU

Run `make demo-serving` from the repository root after installing the Python requirements. `serving_replay.py` uses the prompt generator and cache allocations in `serving/` to replay eight warm-up requests and 1,200 measured requests without a GPU.

It trains the existing example predictor on synthetic history, then calls the C++ RANCHO simulator. The HBM-only comparison retains the shared beginnings as the serving baseline does. Output includes cache hits, HBM replacement, recomputation and flash writes. Writes include warm-up; measured cache metrics exclude it. The allowance is one flash-capacity equivalent across the replay.

The serving experiment's trained predictor is private. This example uses its own synthetic fit. It processes prompt-cache operations in arrival order, without GPU scheduling, output-token execution or the serving engine's elapsed-time write-credit guard. Matching hit ratios does not establish identical admission decisions or serving performance.

To save a computed summary:

```bash
python examples/serving_replay.py --output demo-summary.json
```

The output must be a new file. Temporary request streams, predictions and replay logs are removed when the process exits.

## Three-request walkthrough

Run `make demo`. `demo.py` trains on generated earlier sessions, checks the request reader and feature builder, then evaluates the three requests in `demo_requests.json`.

Expected synthetic totals are 4 flash hits, 0 HBM hits, and 2 blocks written.
