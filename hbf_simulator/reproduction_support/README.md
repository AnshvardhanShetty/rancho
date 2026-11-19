# Building reuse labels

These tools build training counts from request sequences and recorded cache decisions.

1. `training/initial_reuse_counts.c` counts later accesses to newly introduced KV.
2. `training/record_write_offers.cpp` records the recomputed suffix considered for each write, including rejected writes.
3. `training/reuse_index.cpp` indexes repeated accesses by block and request.
4. `training/suffix_reuse_counts.cpp` counts later accesses to each offered suffix and checks the counts with a separate calculation.

The output records the number of blocks per offer in `c.i32` and later accesses by hour in `Y.u64`. Training uses the hours completed before each fit.

For the small CPU demo, [synthetic_training.py](../experiments/synthetic_training.py) builds these counts in memory. The cache decisions come from [rancho.cpp](../rancho.cpp).
