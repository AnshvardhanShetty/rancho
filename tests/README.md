# Tests

From the repository root:

```bash
make test-simulator
make test-demo
```

The simulator tests cover ten synthetic cases, including duplicate blocks, empty requests, repeated writes, score ties and controller updates. They compare decisions, cache contents and counts against the saved expected outputs.

The demo tests check the serving workload's cache counts, write allowance and training cutoffs. A small hand-counted example checks the reuse labels, including blocks shared between conversations.

The serving and SSD folders contain their own checks. See [serving/RUNNING.md](../serving/RUNNING.md) and [ssd/replay/README.md](../ssd/replay/README.md).
