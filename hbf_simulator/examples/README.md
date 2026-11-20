# Try RANCHO on a CPU

Run these commands from the repo root after [setting up Python](../../README.md#try-it-without-a-gpu).

## Three requests

```bash
make demo
```

The first of the three requests writes two blocks to flash. The next two each read those blocks back. You should see four flash hits and two writes. Start at [demo.py](demo.py) to see the whole run.

## Eight conversations

```bash
make demo-serving
```

[serving_replay.py](serving_replay.py) uses the same synthetic prompts as the [vLLM example](../../hbf_serving/README.md). It warms the cache with eight requests, then measures 1,200 more. It trains on earlier synthetic requests. A first replay records the KV offered for writing, and the final model learns from later reads of those blocks.

The output shows cache hits, blocks recomputed and writes. Hits and recomputation exclude warm-up. The write allowance covers both warm-up and measurement. This run measures cache behavior. The GPU example runs the model and measures latency and throughput.

Save the counts in a new file with:

```bash
python hbf_simulator/examples/serving_replay.py --output demo-summary.json
```
