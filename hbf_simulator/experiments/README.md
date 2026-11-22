# Trace replay

Start at the [three-request demo](../examples/demo.py) to see these files work together.

1. [trace_reader.py](trace_reader.py) reads one request per JSONL line, selects the model we want and saves the requests in arrival order.
2. [request_features.py](request_features.py) builds the four model inputs. It stores the same user ID as a number and keeps the features in the same order as the requests.
3. [run_simulation.py](run_simulation.py) passes the prepared requests, predictions, cache sizes and write allowance to the C++ simulator. It reads back the counts the simulator saves.

The [larger CPU demo](../examples/serving_replay.py) makes its own requests. [synthetic_training.py](synthetic_training.py) builds reuse examples from those earlier requests and uses the same tree trainer.
