# HBF cache simulator

The simulator processes requests in arrival order. For each request, it finds the cached prefix, scores the remaining KV, checks the write budget and updates the caches.

Start with `Simulator::process_request()` in [rancho.cpp](rancho.cpp).

| File | Purpose |
| --- | --- |
| [write_controller.hpp](write_controller.hpp) | Use recent demand and remaining writes to update admission thresholds. |
| [cache.hpp](cache.hpp) | Store block IDs, replace old blocks and count the writes needed for a copy. |
| [hbm_only_readable_prefix.cpp](hbm_only_readable_prefix.cpp) | Track which accesses an HBM-only cache could serve. |
| [simulation_io.hpp](simulation_io.hpp) | Read inputs and save decisions and counts. |

Python supplies reuse predictions. C++ uses those predictions to make cache decisions. Run `make demo-serving` from the repository root to see them together.
