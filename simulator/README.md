# Cache replay

Start at `Simulator::process_request()` in `rancho.cpp`: read the prefix, identify reuse, count hits, price the suffix, decide, write, and update history for later requests.

- `write_controller.hpp`: use recent scores and quoted write costs to choose affordable score cutoffs.
- `cache.hpp`: store block IDs, maintain least-recently-used order, and trial writes to obtain their exact charge.
- `hbm_only_readable_prefix.cpp`: replay a separate HBM-only cache for the history filter.
- `simulation_io.hpp`: load caller-supplied input arrays and save decisions, counters, and cache snapshots.

Python supplies predictions already fitted from past observations. The C++ replay makes cache decisions and counts outcomes; it does not train the model. The synthetic demo shows how the pieces connect.
