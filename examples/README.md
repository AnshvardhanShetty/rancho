# Synthetic examples

## Serving workload on CPU

Run `make demo-serving` from the repository root after installing the Python requirements.

The demo uses the serving experiment's synthetic prompt generator: eight conversations, eight warm-up requests and 1,200 measured requests. It trains RANCHO on earlier requests, then replays the measured requests through the C++ simulator. The output shows cache hits, GPU memory savings, recomputation and flash writes.

Training uses the research pipeline. An initial predictor supplies scores for a replay that records which KV each request offers. The final predictor learns from the later reuse of that KV. Labels include only accesses observed before each training cutoff.

The baseline keeps shared prefixes in HBM. RANCHO uses HBF and a smaller HBM cache. Both start empty before warm-up; cache-hit ratios and recomputation count only measured requests. Writes include warm-up and must fit one flash-capacity allowance.

The model is trained locally on these synthetic requests. The CPU replay processes prompt caching in arrival order; the GPU runner also executes the model, schedules concurrent requests and applies elapsed-time write credits. [Serving code](../serving/README.md).

To save the computed summary in a new file:

```bash
python examples/serving_replay.py --output demo-summary.json
```

## Three-request walkthrough

Run `make demo` for a smaller example. It trains a predictor, checks the request reader and features, then processes the three requests in `demo_requests.json`.

The first request writes two KV blocks. Two later requests reuse both. The result is **4 flash hits and 2 block writes**.
