# Reuse predictor

The model gives each request one estimate: the average number of later reads per block offered for writing. Its four inputs are user ID, turn number, chat age and time since the previous request in the chat.

Start in [model_training.py](model_training.py) and read these in order:

1. `RequestData.get_training_rows_and_reuse()` selects earlier offers and counts the reads seen by the training cutoff. It divides the count for each offer by the number of distinct blocks offered.
2. `fit_tree_model()` tries tree counts on an earlier six-hour window. It then fits that many trees using the examples available at the current cutoff.
3. `predict_reuse()` takes the four inputs and returns one prediction per request.
4. `fit_and_predict_one_hour()` puts those steps together and saves the model and its predictions for the next hour.

[tree_encoding.py](tree_encoding.py) maps known user IDs to category codes so the model treats each ID as a name. [model_settings.json](model_settings.json) has the settings used for fitting. [file_helpers.py](file_helpers.py) reads and saves the model's records.

[run_experiment.py](../experiments/run_experiment.py) calls the fitting code once per hour. The [CPU demo](../examples/README.md) supplies synthetic training data.

## Model files

| File | What it contains |
| --- | --- |
| `training_recipe.json` | The feature list and tree settings under the `recipe` key |
| `request_stream_files.json` | The location of the arrival-time array under `raw` → `_ts.i64` |
| `training_data_files.json` | Paths to the feature table, feature types and initial reuse counts |
| `model_fit_summary.json` | The cutoff, tree count and file checksums saved after one fit |

The first two files go in the model folder. The training data file list goes one level above them. Older prepared folders used `PLAN.json`, `INPUTS.json` and `RESOLVED_MODEL_INPUTS.json`. The loader still reads those names. The CPU demo already supplies its own inputs.
