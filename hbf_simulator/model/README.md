# Predicting KV reuse

The model gives each request one estimate: the average number of later reads per block offered for writing. Its four inputs are user ID, turn number, chat age and time since the previous request in the chat.

Start in [model_training.py](model_training.py) and read these in order:

1. `RequestData.training_examples()` selects earlier offers and counts the reads seen by the training cutoff. It divides the count for each offer by the number of distinct blocks offered.
2. `fit_tree_model()` tries tree counts on an earlier six-hour window. It then fits that many trees using the examples available at the current cutoff.
3. `predict_reuse()` takes the four inputs and returns one prediction per request.
4. `fit_hourly_model()` puts those steps together and saves the model and its predictions for the next hour.

[tree_encoding.py](tree_encoding.py) maps known user IDs to category codes so the model treats each ID as a name. [model_settings.json](model_settings.json) has the settings used for fitting. [file_helpers.py](file_helpers.py) reads and saves the model's records.

[run_experiment.py](../experiments/run_experiment.py) calls the fitting code once per hour. The [CPU demo](../examples/README.md) supplies synthetic training data.
