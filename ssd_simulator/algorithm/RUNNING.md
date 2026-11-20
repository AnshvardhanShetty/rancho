# Run the SSD algorithm

Install the repo's Python requirements. The C++ tools also need a C++17 compiler, OpenSSL headers and `pkg-config`.

## Check it with generated data

From the repo root:

```bash
make -C ssd_simulator/replay recorded-support
python -m unittest discover -s ssd_simulator/algorithm/tests
python ssd_simulator/algorithm/tests/check_policy.py
python ssd_simulator/algorithm/tests/check_pipeline.py
```

The policy check compares write decisions and cache counts against the recorded source on small inputs. The pipeline check generates examples and runs training, replay and accounting together.

## Run with prepared inputs

The runner needs a manifest that points to a prepared trace and its training records. Use a new output directory. Capacity and write allowance are measured in blocks.

```bash
python -m ssd_simulator.algorithm.run \
  --manifest /path/to/prepared-inputs.json --output /path/to/new/output \
  --ssd-blocks BLOCK_COUNT --budget-blocks WRITE_ALLOWANCE
```

The recorded run uses 81,380 GPU cache blocks. It fits each hour, replays days 4–7 with seed 17, and asks Python to refresh saved offer scores when the model changes. Each hourly folder contains the model and a `model_fit_summary.json` with its checksum. The runner then checks the write log and counts which reads SSD saved.

The manifest has:

| Key | What it supplies |
| --- | --- |
| `files` | `timestamps`, `evictions`, `features` and `sample_indices` |
| `eviction_count` and `ntrain` | Total evictions and the number before the measured replay |
| `feature_schema` | Feature names and which values are categories |
| `cumulative_labels` | Little-endian unsigned 32-bit counts with 169 rows and one column per sampled eviction. Row h counts gains observed before hour h |
| `replay` | `block_ids`, `block_counts`, `arrival_times`, `tokens`, `measured_evictions`, `reference_rows`, `block_id_count` and a `credit` group with `offsets`, `bits` and `ranks` |
| `expected_scores_sha256` and `expected_result` | Optional saved checksum and counts for an exact comparison |

Each file record has a `path` and `sha256`. Paths can be relative to the manifest. See the [request file formats](../replay/README.md#files-the-replay-uses). The three credit files are `offsets.i64`, `credit.bits` and `credit.rank` in one directory. The original prediction checksum is in [SOURCE_VERIFICATION.json](SOURCE_VERIFICATION.json).

## Build the training counts

1. Use [session_accounting.cpp](support/session_accounting.cpp) to build and check the same-session read index for the prepared request sequence. Keep the request order, session IDs, block order and duplicates from the trace.
2. Run the reference policy with cold caches over days 1–3 and again over days 4–7. Check the writes and the request rows.
3. [label_replay.cpp](support/label_replay.cpp) takes the reference writes and measures what saving each sampled eviction would have added. Use `--label-samples`, `--label-credit` and `--label-event-offset`. Use offset zero for training and the earlier eviction count for measurement.
4. [labels.py](labels.py) combines the two gain files into one label row per cutoff. A read at the cutoff belongs to the following fit.

The public repo includes the code and generated examples. The full trace and prepared production inputs are private.
