# Reuse predictor

`model_training.py` contains the data loading, observed-reuse labels, weighting, tree fitting, prediction, and output checks for one hourly update.

Read `RequestData`, then `training_weights()`, `prepare_training_arrays()`, `fit_tree_model()`, `predict_reuse()`, and `fit_hourly_model()`.

`run_hourly_models()` in `experiments/run_experiment.py` supplies the outer loop. `tree_encoding.py` prepares categorical user IDs. `model_settings.json` is the four-feature tree recipe. `file_helpers.py` handles JSON and content fingerprints.

Training data are supplied by the caller. The bundled demonstration uses only synthetic sessions. Production inputs, trained production models, per-request predictions, and detailed experiment outputs are not included.
