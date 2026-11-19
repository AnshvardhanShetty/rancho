# SSD cache replay

This simulator reads requests in arrival order and models which KV is kept in GPU memory or on SSD. It stores block IDs rather than actual model data, so it can study cache decisions without running an LLM.

For each request it:

1. Finds how much of the input is already cached, stopping at the first missing block.
2. Works out which GPU blocks must leave to make room.
3. Decides whether to save those blocks on SSD, within the write allowance.
4. Updates both caches and records the reads served and blocks written.

A block already on SSD does not need another write. When the SSD cache is full, the least recently used block is replaced.

## Files

| File | What it does |
| --- | --- |
| [ssd_replay.cpp](ssd_replay.cpp) | Cache replay and write-budget controller. Start with the short overview at the top. |
| [check_replay.cpp](check_replay.cpp) | Independently reconstruct caches and counts from the replay's write log. |
| [test_replay.py](test_replay.py) | Compare 112 small synthetic cases with a separate Python cache model and the C++ checker. |
| [test_rules.json](test_rules.json) | Admission rules exercised by those synthetic cases. |

The replay can accept supplied reuse scores, admit everything within the budget, admit randomly, or use access/eviction-count rules. In scored mode, one score corresponds to one GPU eviction. With hourly model updates, it can ask a Python driver to update the scores of earlier offers before recalculating the cutoff.

This is the baseline replay. The [SSD algorithm](../algorithm/README.md) adds the predictor, hourly training and within-session reuse accounting.


## Build and check

Requires a C++17 compiler, Python 3.11+, OpenSSL development headers and `pkg-config`. From the repository root:

```bash
make -C ssd/replay test
```

The command builds two programs under `build/ssd` and checks generated requests against a separate Python cache model. Cases include duplicates, prefix gaps, repeated writes, budget exhaustion, replacement, oversized requests and a corrupted write log.

If OpenSSL is not found automatically, supply `OPENSSL_CFLAGS` and `OPENSSL_LIBS` to Make for your installation.

## Inputs and outputs

The replay reads binary arrays supplied by the caller:

- `<prefix>_nb.i32`: number of blocks in each request.
- `<prefix>_ids.i32`: the requests' block IDs joined together, preserving order and duplicates.
- `<prefix>_ts.i64`: arrival times in microseconds.
- `--tokens`: input token counts, one per request.
- `--scores`: predicted reuse scores, indexed by eviction, when using scored admission.

It writes counters to standard output, plus a per-request `.rows.i64` file and a `.programs.bits` write-decision log. The independent checker uses that log to reconstruct what happened.

Requests are processed one at a time. The counters report cached prompt positions. The research evaluation additionally filters accesses by conversation before calculating recomputation reduction. The [SSD integration](../README.md) handles concurrent model execution.
