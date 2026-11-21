# Reuse labels

A training label says how often the blocks in one write offer were read again. These tools produce those counts from a prepared trace.

1. [initial_reuse_counts.c](training/initial_reuse_counts.c) counts later reads of blocks introduced by each request. These counts fit the first predictor.
2. [record_write_offers.cpp](training/record_write_offers.cpp) replays the first predictor and saves which suffix the cache offered to write on each request. It saves declined offers too.
3. [reuse_index.cpp](training/reuse_index.cpp) records where each block is read again in the same chat.
4. [suffix_reuse_counts.cpp](training/suffix_reuse_counts.cpp) uses the index to count later reads of each offered suffix. It checks those counts with a separate calculation.

`c.i32` has the number of distinct blocks in each offer. `Y.u64` has the number of later reads in each hour. The trainer adds only the hours that ended before the fit starts.

The [CPU demo](../examples/README.md) builds the same kind of counts in memory from its synthetic requests.
