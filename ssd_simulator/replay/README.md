# SSD baseline replay

The replay reads requests in arrival order. It finds the cached part of each prompt, starting at the first block and stopping at the first miss. It then makes room for new KV in GPU memory. A policy decides which evicted blocks to save to SSD. An existing SSD copy costs no new write. When SSD is full, it removes the least recently used block.

This replay has simple policies such as random and access count. It can also use supplied scores, one per GPU eviction. See the [recorded SSD algorithm](../algorithm/README.md) for RANCHO's model training and final reuse counts.

## Read the code

- [ssd_replay.cpp](ssd_replay.cpp) runs the caches and handles write decisions
- [check_replay.cpp](check_replay.cpp) rebuilds both caches from the write log and checks the counts
- [test_replay.py](test_replay.py) compares 112 small cases against a separate Python cache
- [test_rules.json](test_rules.json) lists the policies used by the cases

## Tests

From the repo root, with Python 3.11+, a C++17 compiler, OpenSSL headers and `pkg-config`:

```bash
make -C ssd_simulator/replay test
```

The command builds the two programs under `build/ssd` and makes its own small traces. If Make cannot find OpenSSL, pass `OPENSSL_CFLAGS` and `OPENSSL_LIBS` for your installation.

## Input files

- `<prefix>_nb.i32` has the block count for each request
- `<prefix>_ids.i32` has all block IDs in request order, including duplicates
- `<prefix>_ts.i64` has arrival times in microseconds
- `--tokens` points to the input token counts
- `--scores` points to the scores when replaying a scored policy

It prints totals and saves one row per request in `.rows.i64`. The separate checker reads `.programs.bits` to see which writes were made. These counts describe the cached prefix. The [paper accounting](../algorithm/accounting.py) also checks which reads are eligible within each conversation.
