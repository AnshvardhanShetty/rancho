"""Train the reuse predictor and score one hour of requests.

1. RequestData loads features, offer sizes and read counts. It builds each
   training label from the reads already seen when the hour starts.
2. fit_tree_model chooses the number of trees on an earlier six-hour window,
   then fits a new model with the data available at the current cutoff.
3. fit_and_predict_one_hour checks the labels, predicts the requests arriving
   in that hour and saves the model and predictions.

The experiment runner calls fit_and_predict_one_hour for each hour.
"""

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np

from .tree_encoding import build_category_values, encode_tree_features
from .training_files import (
    file_fingerprint,
    find_training_file,
    read_json,
    write_json,
)

HOUR_US = 3_600_000_000
VALIDATION_WINDOW_US = 6 * HOUR_US


class RequestData:
    """Keep the request inputs and reuse counts in the same request order"""

    def __init__(self, arrival_times_path, features_path, feature_names, label_directory):
        """Load request features and the saved reads of each offered suffix."""
        self.arrival_times_us = np.fromfile(arrival_times_path, "<i8")
        self.request_count = len(self.arrival_times_us)

        # Position i in every array belongs to request i in arrival order.
        with np.load(features_path) as feature_file:
            self.features_by_name = {name: feature_file[name] for name in feature_names}

        audit_path = find_training_file(label_directory, "offer_label_audit.json", "AUDIT.json")
        label_audit = read_json(audit_path)
        assert label_audit["status"] == "PASS_FULL_INDEPENDENT_SUFFIX_LABEL_AUDIT"

        # One offer size per request and one read count per hour and request.
        offer_sizes_path = find_training_file(
            label_directory, "offered_block_counts.i32", "c.i32"
        )
        hourly_reads_path = find_training_file(
            label_directory, "hourly_reuse_counts.u64", "Y.u64"
        )
        self.offer_block_counts = np.fromfile(offer_sizes_path, "<i4")
        self.hourly_read_counts = np.memmap(
            hourly_reads_path, "<u8", "r", shape=(168, self.request_count)
        )
        self.label_file_sha256 = file_fingerprint(hourly_reads_path)

        # Start the running observed-read totals at zero.
        self.counted_until_hour = 0
        self.observed_read_totals = np.zeros(self.request_count, dtype=np.int64)
        assert len(self.offer_block_counts) == self.request_count and np.all(
            self.offer_block_counts >= 0
        )

    def get_training_rows_and_reuse(self, cutoff_us):
        """Return request positions and reads per block known by this cutoff"""
        # Convert the hourly cutoff into a row boundary in the read-count table.
        assert cutoff_us % HOUR_US == 0 and 0 <= cutoff_us <= 168 * HOUR_US
        cutoff_hour = cutoff_us // HOUR_US

        # Add newly completed hours, excluding reads at or after the cutoff.
        if cutoff_hour > self.counted_until_hour:
            new_hours = self.hourly_read_counts[self.counted_until_hour : cutoff_hour]
            self.observed_read_totals += new_hours.sum(axis=0, dtype=np.int64)
            self.counted_until_hour = cutoff_hour

        # For an earlier validation cutoff, subtract later hours without changing stored totals.
        observed_reads = self.observed_read_totals
        if cutoff_hour < self.counted_until_hour:
            hours_after_cutoff = self.hourly_read_counts[cutoff_hour : self.counted_until_hour]
            observed_reads = observed_reads - hours_after_cutoff.sum(axis=0, dtype=np.int64)

        # One label per earlier offer: observed reads divided by blocks offered.
        offered_blocks = self.offer_block_counts
        training_rows = np.flatnonzero((self.arrival_times_us < cutoff_us) & (offered_blocks > 0))
        reads_per_block = observed_reads[training_rows] / offered_blocks[training_rows]
        assert np.all(np.isfinite(reads_per_block)) and np.all(reads_per_block >= 0)
        assert not len(training_rows) or self.arrival_times_us[training_rows].max() < cutoff_us
        return training_rows, reads_per_block

    def feature_table(self, request_rows, feature_names):
        """Put requests into rows and features into columns"""
        return np.column_stack(
            [self.features_by_name[name][request_rows] for name in feature_names]
        ).astype(float)

    def categorical_columns(self, feature_names):
        """Tell LightGBM which inputs are names such as user IDs"""
        return [column for column, name in enumerate(feature_names) if name == "user"]


# Stop with a readable message when a required check fails.
def check_condition(condition, message):
    if not condition:
        raise ValueError(message)


def compute_training_weights(request_data, request_rows, cutoff_us):
    """Give requests half as much weight for each extra day since arrival"""
    weights = np.exp2(
        -(cutoff_us - np.asarray(request_data.arrival_times_us[request_rows], dtype=float))
        / (24 * HOUR_US)
    )
    check_condition(
        np.all(np.isfinite(weights)) and np.all(weights > 0),
        "Invalid training weights",
    )

    return weights / weights.mean() if len(weights) else weights


def prepare_training_arrays(request_data, cutoff_us, training_settings):
    """Match each earlier offer's reuse label to its features and weight."""
    fit_started_at = time.monotonic()
    request_rows, reads_per_block = request_data.get_training_rows_and_reuse(cutoff_us)
    check_condition(
        len(reads_per_block) > 0
        and np.all(np.isfinite(reads_per_block))
        and np.all(reads_per_block >= 0),
        "No finite nonnegative training labels",
    )

    feature_names = training_settings["features"]
    features = request_data.feature_table(request_rows, feature_names)
    category_columns = request_data.categorical_columns(feature_names)
    weights = compute_training_weights(request_data, request_rows, cutoff_us)
    average_reuse = float(np.average(reads_per_block, weights=weights))

    # Record which examples, weights, and settings entered this fit.
    fit_summary = dict(
        cut_us=cutoff_us,
        train_rows=len(request_rows),
        target_available_no_later_than_us=cutoff_us,
        label="OBS6H observed density",
        expanded_recipe=training_settings,
        sample_weight_sum=float(weights.sum()),
        sample_weight_min=float(weights.min()),
        sample_weight_max=float(weights.max()),
    )
    return (
        fit_started_at,
        request_rows,
        reads_per_block,
        features,
        category_columns,
        weights,
        average_reuse,
        fit_summary,
    )


def fit_tree_model(request_data, cutoff_us, training_settings):
    """Choose a tree count on earlier data, then fit at the current cutoff."""
    check_condition(
        training_settings["model"] == "tree"
        and training_settings["objective"] == "poisson"
        and training_settings["sample_weight"] == "recency24h"
        and training_settings["window_hours"] == 0,
        "Expected the recorded Poisson tree with recency24h weights and no time window",
    )
    (
        fit_started_at,
        request_rows,
        reads_per_block,
        features,
        category_columns,
        weights,
        average_reuse,
        fit_summary,
    ) = prepare_training_arrays(request_data, cutoff_us, training_settings)
    import lightgbm as lgb

    # Translate model_settings.json into LightGBM settings; keep training deterministic.
    tree_parameters = dict(
        objective=training_settings["objective"],
        learning_rate=training_settings["learning_rate"],
        num_leaves=training_settings["leaves"],
        min_child_samples=training_settings["minimum"],
        reg_alpha=training_settings["l1"],
        reg_lambda=training_settings["l2"],
        n_jobs=1,
        verbosity=-1,
        random_state=0,
        deterministic=True,
        force_col_wise=True,
        bagging_fraction=1.0,
        feature_fraction=1.0,
    )

    # At noon, fit on what we knew at 6 am. Check the tree counts on requests
    # between 6 am and noon, using only reads that have happened by noon.
    validation_start_us = cutoff_us - VALIDATION_WINDOW_US
    earlier_request_rows, earlier_reads_per_block = request_data.get_training_rows_and_reuse(
        validation_start_us
    )
    validation_mask = request_data.arrival_times_us[request_rows] >= validation_start_us

    tree_count = training_settings["max_rounds"]
    validation_errors = []
    if (
        len(earlier_request_rows) >= 2 * training_settings["minimum"]
        and validation_mask.any()
        and earlier_reads_per_block.sum() > 0
    ):
        earlier_features = request_data.feature_table(
            earlier_request_rows, training_settings["features"]
        )
        category_values = build_category_values(earlier_features, category_columns)
        earlier_weights = compute_training_weights(
            request_data, earlier_request_rows, validation_start_us
        )
        validation_model = lgb.LGBMRegressor(n_estimators=tree_count, **tree_parameters)
        validation_model.fit(
            encode_tree_features(earlier_features, category_values),
            earlier_reads_per_block,
            sample_weight=earlier_weights,
            eval_set=[
                (
                    encode_tree_features(features[validation_mask], category_values),
                    reads_per_block[validation_mask],
                )
            ],
            eval_sample_weight=[weights[validation_mask]],
            eval_metric=training_settings["objective"],
            categorical_feature=category_columns,
        )

        # Entry zero is the validation error after the first tree.
        validation_errors = validation_model.evals_result_["valid_0"][
            training_settings["objective"]
        ]
        check_condition(
            np.isfinite(validation_errors).all(), "Nonfinite chronological validation curve"
        )
        tree_count = int(np.argmin(validation_errors)) + 1

    # Fit a fresh model at the current cutoff using the chosen number of trees.
    # With too little data or no observed reuse, use the weighted average.
    model = dict(kind="global", mean=average_reuse)
    if len(reads_per_block) >= 2 * training_settings["minimum"] and reads_per_block.sum() > 0:
        category_values = build_category_values(features, category_columns)
        trained_model = lgb.LGBMRegressor(n_estimators=tree_count, **tree_parameters)
        trained_model.fit(
            encode_tree_features(features, category_values),
            reads_per_block,
            sample_weight=weights,
            categorical_feature=category_columns,
        )
        model = dict(
            kind="expanded_tree",
            mean=average_reuse,
            fit=trained_model,
            vocab=category_values,
            transform="none",
        )

    fit_summary.update(
        inner_fit_cut_us=validation_start_us,
        inner_validation_us=VALIDATION_WINDOW_US,
        inner_curve=validation_errors,
        tree_count=tree_count,
        parameters=tree_parameters,
        target_transform="none",
        validation_metric=training_settings["objective"],
        clipped_negative_predictions=True,
    )
    return record_fit_details(model, features, reads_per_block, fit_started_at, fit_summary)


def predict_reuse(model, features):
    """Predict reads per offered block with the tree or its mean fallback."""
    # With too little training data, return the weighted average instead of a tree.
    if model["kind"] == "global":
        return np.full(len(features), model["mean"])
    check_condition(model["kind"] == "expanded_tree", "Expected a tree model")

    # Apply the saved user-ID mapping, then ask the fitted trees for predictions.
    predictions = model["fit"].predict(
        encode_tree_features(features, model["vocab"]), num_threads=1
    )
    return np.maximum(predictions, 0.0)


def record_fit_details(model, features, reads_per_block, fit_started_at, fit_summary):
    """Check sample predictions and record fit time and model size."""
    fit_summary["fit_seconds"] = time.monotonic() - fit_started_at
    sample_rows = np.arange(
        0, len(reads_per_block), max(1, len(reads_per_block) // 50000), dtype=int
    )
    predictions = predict_reuse(model, features[sample_rows])
    check_condition(
        np.isfinite(predictions).all() and np.all(predictions >= 0),
        "Invalid training predictions",
    )
    fit_summary["prediction_bins"] = np.unique(
        np.quantile(predictions, np.arange(1, 10) / 10)
    ).tolist()
    fit_summary["model_bytes"] = len(pickle.dumps(model, protocol=5))
    return model, fit_summary


def fit_and_predict_one_hour(request_data, cutoff_us, training_settings, output_directory):
    """Train on earlier observations and predict reuse for requests in this hour."""
    # Save each hourly fit in its own output folder.
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    # Fit using only observations from before this hour.
    model, fit_summary = fit_tree_model(request_data, cutoff_us, training_settings)
    training_rows, reads_per_block = request_data.get_training_rows_and_reuse(cutoff_us)
    assert len(training_rows) > 0

    # Recount the reads from scratch to check both sets of labels.
    for audit_cutoff_us in [cutoff_us, cutoff_us - VALIDATION_WINDOW_US]:
        checked_rows, checked_reads_per_block = request_data.get_training_rows_and_reuse(
            audit_cutoff_us
        )
        expected_rows = np.flatnonzero(
            (request_data.arrival_times_us < audit_cutoff_us)
            & (request_data.offer_block_counts > 0)
        )
        recounted_reads = request_data.hourly_read_counts[: audit_cutoff_us // HOUR_US].sum(
            axis=0, dtype=np.int64
        )
        assert np.array_equal(checked_rows, expected_rows) and np.array_equal(
            checked_reads_per_block,
            recounted_reads[expected_rows] / request_data.offer_block_counts[expected_rows],
        )

    # Find the requests in this hour. The prediction uses only their four features.
    hour_end_us = min(cutoff_us + HOUR_US, 168 * HOUR_US)
    first_row = int(np.searchsorted(request_data.arrival_times_us, cutoff_us))
    stop_row = int(np.searchsorted(request_data.arrival_times_us, hour_end_us))
    prediction_rows = np.arange(first_row, stop_row)
    predictions = predict_reuse(
        model,
        request_data.feature_table(prediction_rows, training_settings["features"]),
    )
    assert np.all(np.isfinite(predictions)) and np.all(predictions >= 0)

    # Save the model, the predictions and a summary of the fit.
    with (output_directory / "model.pkl").open("xb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    predictions.astype("<f8").tofile(output_directory / "hourly_predictions.f64")

    fit_summary.update(
        status="PASS_CAUSAL_FIXED_RECIPE_FIT",
        label="cutoff-censored observed offered_suffix reuse per distinct block",
        target="offered_suffix",
        cut_us=cutoff_us,
        label_available_strictly_before_us=cutoff_us,
        latest_training_request_us=int(request_data.arrival_times_us[training_rows].max()),
        training_rows=len(training_rows),
        training_row_hash=array_fingerprint(training_rows.astype("<i8")),
        training_target_hash=array_fingerprint(reads_per_block.astype("<f8")),
        label_source_sha256=request_data.label_file_sha256,
        prediction_start=first_row,
        prediction_stop=stop_row,
        model_sha256=file_fingerprint(output_directory / "model.pkl"),
        prediction_sha256=file_fingerprint(output_directory / "hourly_predictions.f64"),
        future_buckets_excluded_from_training=True,
        direct_prefix_and_inner_label_audit=True,
    )
    write_json(output_directory / "model_fit_summary.json", fit_summary, exclusive=True)

    # The runner uses these positions to place the predictions in replay order.
    return predictions, first_row, stop_row


def array_fingerprint(values):
    """Hash the array in request order for the saved fit record."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
