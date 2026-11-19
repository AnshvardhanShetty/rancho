# Synthetic regression checks

Run `make test-simulator` from the repository root. `simulator_regression.py` generates ten deterministic synthetic traces and compares the saved outputs against `simulator_expected.json`.

The cases cover empty requests, duplicate blocks, rewrites, score ties, and controller updates. All inputs are generated; the expected hashes do not come from a production workload.
