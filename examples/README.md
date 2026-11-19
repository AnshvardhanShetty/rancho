# Synthetic example

Run `make demo` from the repository root. `demo.py` trains on generated earlier sessions, checks the trace reader and feature builder, then evaluates the three requests in `demo_requests.json`.

Expected synthetic totals: 4 flash hits, 0 HBM hits, and 2 blocks written. These are a small correctness check, not a research result.
