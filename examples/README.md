# Synthetic examples

## Serving prompts on CPU

Run `make demo-serving` from the repository root after installing the Python requirements. `serving_replay.py` uses the prompt generator and cache allocations in `serving/` to replay eight warm-up requests and 1,200 measured requests without a GPU.

It uses the research training code to build an initial predictor, replay earlier requests to collect actual suffix offers, and fit the final predictor from the observed reuse of those suffixes. The fitted model supplies scores to the C++ RANCHO simulator. The HBM-only comparison retains the shared beginnings as the serving baseline does. Output includes cache hits, HBM replacement, recomputation and flash writes. Writes include warm-up; measured cache metrics exclude it. The allowance is one flash-capacity equivalent across the replay.

The training recipe, label-selection code and admission controller are the research implementations. The model is fitted locally on the generator's earlier requests; it is not the saved model from the GPU benchmark. It processes prompt-cache operations in arrival order, without GPU scheduling, output-token execution or the serving engine's elapsed-time write-credit guard. Matching hit ratios does not establish identical admission decisions or serving performance.

To save a computed summary:

```bash
python examples/serving_replay.py --output demo-summary.json
```

The output must be a new file. Temporary request streams, predictions and replay logs are removed when the process exits.

## Three-request walkthrough

Run `make demo`. `demo.py` trains on generated earlier sessions, checks the request reader and feature builder, then evaluates the three requests in `demo_requests.json`.

Expected synthetic totals are 4 flash hits, 0 HBM hits, and 2 blocks written.
