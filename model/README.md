# Reuse prediction

The predictor estimates how often later requests will access a request's KV. It uses user ID, request position in the conversation, conversation age and time since the previous request.

[model_training.py](model_training.py) prepares the observed reuse labels, fits gradient-boosted trees and makes predictions. Each fit uses accesses observed before its cutoff. The tree count is chosen using an earlier validation period.

- `RequestData.training_examples()` selects earlier requests and their reuse labels.
- `fit_tree_model()` trains the model with the settings in [model_settings.json](model_settings.json).
- `predict_reuse()` applies the fitted model to request features.
- `fit_hourly_model()` saves a model and predictions for the next hour.

[tree_encoding.py](tree_encoding.py) handles categorical user IDs. [file_helpers.py](file_helpers.py) reads and writes model records.

The [CPU demo](../examples/README.md) trains this model on synthetic requests using [synthetic_training.py](../experiments/synthetic_training.py).
