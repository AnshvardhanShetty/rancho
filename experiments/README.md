# Experiment helpers

The CPU demo starts at [examples/serving_replay.py](../examples/serving_replay.py).

| File | Purpose |
| --- | --- |
| [synthetic_training.py](synthetic_training.py) | Build reuse labels from earlier synthetic requests and fit the predictor. |
| [trace_reader.py](trace_reader.py) | Read requests, sort by arrival time and save arrays for the simulator. |
| [run_experiment.py](run_experiment.py) | Calculate request features and train hourly predictors on supplied data. |
| [run_simulation.py](run_simulation.py) | Run the C++ simulator and read its counts. |
