# SSD simulator

The simulator replays requests and decides which blocks to save on SSD when they leave memory. It counts how much repeated prefill those blocks avoid.

| Folder | What it contains |
| --- | --- |
| [algorithm](algorithm/README.md) | The reuse predictor, hourly training and write-budget controller. |
| [replay](replay/README.md) | The baseline cache replay and small synthetic tests. |

Start with [algorithm/run.py](algorithm/run.py) to follow training and replay. See [algorithm/RUNNING.md](algorithm/RUNNING.md) for commands and input formats.

For saving and restoring actual KV with vLLM, see the [SSD serving engine](../ssd_serving/README.md), which is under development.
