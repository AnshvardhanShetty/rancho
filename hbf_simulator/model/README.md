# Reuse predictor

The model gives each request one estimate: the average number of later reads per block offered for writing. Its four inputs are user ID, turn number, chat age and time since the previous request in the chat.

Start in [model_training.py](model_training.py) and read these in order:

1. `RequestData.get_training_rows_and_reuse()` selects earlier offers and counts the reads seen by the training cutoff. It divides the count for each offer by the number of distinct blocks offered.
2. `fit_tree_model()` tries tree counts on an earlier six-hour window. It then fits that many trees using the examples available at the current cutoff.
3. `predict_reuse()` takes the four inputs and returns one prediction per request.
4. `fit_and_predict_one_hour()` puts those steps together and saves the model and its predictions for the next hour.

[model_settings.json](model_settings.json) lists the four inputs and the tree settings. [tree_encoding.py](tree_encoding.py) prepares user IDs as categories for the tree. [training_files.py](training_files.py) opens saved training files and writes fit records.

The [CPU demo](../examples/README.md) supplies its own training examples. [synthetic_training.py](../experiments/synthetic_training.py) shows where those examples come from.

## Model files

| File | What it contains |
| --- | --- |
| `request_features.npz` | One array per model input, in request order |
| `arrival_order_timestamps_us.i64` | When each request arrived, in microseconds |
| `offered_block_counts.i32` | How many distinct blocks each request offered |
| `hourly_reuse_counts.u64` | How many times those blocks were read again in each hour |
| `model_fit_summary.json` | The cutoff, tree count and file checksums saved after one fit |

Pass input paths directly to `RequestData`. All rows must describe the same requests in the same order.
