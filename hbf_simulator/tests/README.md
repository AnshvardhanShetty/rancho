# HBF simulator tests

Run these from the repo root:

```bash
make test-simulator
make test-demo
```

[simulator_regression.py](simulator_regression.py) makes ten small traces. It checks decisions, cache contents and totals against the saved expected outputs. The cases include empty requests, duplicate blocks, rewrites, score ties and cutoff updates.

[test_serving_replay.py](test_serving_replay.py) checks the larger CPU demo's hits, writes and training labels. One small example checks which reads are visible at a training cutoff.

See [HBF serving checks](../../hbf_serving/RUNNING.md#cpu-checks) or [SSD replay tests](../../ssd_simulator/algorithm/RUNNING.md#synthetic-tests) for the other parts of the repo.
