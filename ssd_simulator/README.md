# SSD cache simulator

This simulator tracks KV by block ID and runs without a GPU. It decides which blocks to save to SSD as they leave GPU memory. It then counts how much later prompt computation those writes save.

- [algorithm](algorithm/README.md) has RANCHO's model training, write decisions and recomputation counts. Start with [algorithm/run.py](algorithm/run.py) to see the steps together.
- [replay](replay/README.md) has the baseline policies, the independent cache checker and the small synthetic cases.

See [run commands](algorithm/RUNNING.md) for the SSD algorithm or [SSD serving](../ssd_serving/README.md) for the vLLM integration that moves actual KV.
