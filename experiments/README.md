# Reusable experiment helpers

- `trace_reader.py`: parse one request, collect requests, sort them, and save simulator arrays.
- `run_experiment.py`: build request features and run the hourly model loop on supplied inputs.
- `run_simulation.py`: start the compiled simulator with supplied paths, capacities, predictions, and budget, then read its counters.

The runnable entry point is [examples/demo.py](../examples/demo.py). The fixed real-data launchers, archive manifests, and detailed experiment outputs are excluded from this copy.
