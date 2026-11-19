# Offline reuse-count source

These tools build the inputs consumed by `RequestData` in the model code. Their source is included to show how labels are formed; no production inputs, private paths, saved models, or measured results are included.

1. `training/initial_reuse_counts.c`: build initial trace-derived reuse counts, grouped by the supplied time interval.
2. `training/record_write_offers.cpp`: replay an initial predictor and record each actual suffix offer, including declined offers.
3. `training/reuse_index.cpp`: index eligible within-session reads by block and request. Its audit mode checks the index independently.
4. `training/suffix_reuse_counts.cpp`: count each offered suffix's later eligible reads by hour, with an independent audit.

The final label files are `c.i32` (distinct blocks per offer), `Y.u64` (observed reads by hour and offer), and `AUDIT.json`. Hourly training uses only completed hours before its cutoff.

The offer collector preserves its original collection modes. The current evaluation implementation is [simulator/rancho.cpp](../simulator/rancho.cpp). The synthetic demo constructs simple labels directly and does not require these tools.
