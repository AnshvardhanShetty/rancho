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

from tree_encoding import build_category_values, encode_tree_features
from file_helpers import (
    MODEL_DIRECTORY,
    SIMULATOR_DIRECTORY,
    file_fingerprint,
    read_json,
    read_training_json,
    write_json,
)

# Request times are microseconds. Both names denote one hour in these units.
HOUR = 3_600_000_000
H = HOUR
VALIDATION_HOURS = 6 * HOUR


class RequestData:
    """Keep the request inputs and reuse counts in the same request order"""

    def __init__(self, label_dir=None):
        """Load checked features and the saved counts of later reads"""
        # Read the model settings and find the files with the prepared data
        training_recipe = read_training_json(MODEL_DIRECTORY, "training_recipe.json", "PLAN.json")
        training_files = read_training_json(
            SIMULATOR_DIRECTORY,
            "training_data_files.json",
            "RESOLVED_MODEL_INPUTS.json",
        )
        request_files = read_training_json(
            MODEL_DIRECTORY, "request_stream_files.json", "INPUTS.json"
        )

        # Load one arrival time per request.
        arrival_times_file = request_files["raw"]["_ts.i64"]["path"]
        self.arrival_times_us = np.fromfile(arrival_times_file, "<i8")
        self.request_count = len(self.arrival_times_us)

        # Position i in every array belongs to request i in arrival order.
        feature_names = training_recipe["recipe"]["features"]
        with np.load(training_files["features"]["path"]) as feature_file:
            self.features = {name: feature_file[name] for name in feature_names}

        # User ID is a category. A larger ID does not mean more of anything.
        descriptions = read_json(training_files["feature_registry"]["path"])["features"]
        self.feature_schema = {feature["name"]: feature for feature in descriptions}
        assert all(self.feature_schema[name]["status"] == "ALLOWED" for name in feature_names)

        # The saved labels give us the size of each offer and its reads in each hour.
        if label_dir is not None:
            folder = Path(label_dir)
            assert (
                read_json(folder / "AUDIT.json")["status"]
                == "PASS_FULL_INDEPENDENT_SUFFIX_LABEL_AUDIT"
            )
            self.offer_block_counts = np.fromfile(folder / "c.i32", "<i4")
            self.hourly_read_counts = np.memmap(
                folder / "Y.u64", "<u8", "r", shape=(168, self.request_count)
            )
            self.label_file_sha256 = file_fingerprint(folder / "Y.u64")
            self.label_kind = "offered_suffix"
        # The full rebuild uses initial trace-derived counts when no label folder is supplied.
        else:
            self.offer_block_counts = np.fromfile(
                training_files["original_offer_counts"]["path"], "<i4"
            )
            self.hourly_read_counts = np.memmap(
                training_files["hourly_labels"]["path"],
                "<i4",
                "r",
                shape=(168, self.request_count),
            )
            self.label_file_sha256 = training_files["hourly_labels"]["sha256"]
            self.label_kind = "original_increment"

        # Start the running observed-read totals at zero.
        self.counted_until_hour = 0
        self.observed_read_totals = np.zeros(self.request_count, dtype=np.int64)
        assert len(self.offer_block_counts) == self.request_count and np.all(
            self.offer_block_counts >= 0
        )

    def get_training_rows_and_reuse(self, cutoff_us):
        """Return request positions and reads per block known by this cutoff"""
        # Convert the hourly cutoff into a row boundary in the read-count table.
        assert cutoff_us % H == 0 and 0 <= cutoff_us <= 168 * H
        cutoff_hour = cutoff_us // H

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
            [self.features[name][request_rows] for name in feature_names]
        ).astype(float)

    def categorical_columns(self, feature_names):
        """Tell LightGBM which inputs are names such as user IDs"""
        return [
            column
            for column, name in enumerate(feature_names)
            if self.feature_schema[name]["encoding"] == "categorical"
        ]


# Stop with a readable message when a required check fails.
def check_condition(condition, message):
    if not condition:
        raise ValueError(message)


def compute_training_weights(data, request_rows, cutoff_us):
    """Give requests half as much weight for each extra day since arrival"""
    weights = np.exp2(
        -(cutoff_us - np.asarray(data.arrival_times_us[request_rows], dtype=float)) / (24 * HOUR)
    )
    check_condition(
        np.all(np.isfinite(weights)) and np.all(weights > 0),
        "Invalid training weights",
    )

    return weights / weights.mean() if len(weights) else weights


def prepare_training_arrays(data, cutoff_us, settings):
    """Match each earlier offer's reuse label to its features and weight."""
    started = time.monotonic()
    request_rows, reads_per_block = data.get_training_rows_and_reuse(cutoff_us)
    check_condition(
        len(reads_per_block) > 0
        and np.all(np.isfinite(reads_per_block))
        and np.all(reads_per_block >= 0),
        "No finite nonnegative training labels",
    )

    feature_names = settings["features"]
    features = data.feature_table(request_rows, feature_names)
    category_columns = data.categorical_columns(feature_names)
    weights = compute_training_weights(data, request_rows, cutoff_us)
    average_reuse = float(np.average(reads_per_block, weights=weights))

    # Record which examples, weights, and settings entered this fit.
    record = dict(
        cut_us=cutoff_us,
        train_rows=len(request_rows),
        target_available_no_later_than_us=cutoff_us,
        label="OBS6H observed density",
        expanded_recipe=settings,
        sample_weight_sum=float(weights.sum()),
        sample_weight_min=float(weights.min()),
        sample_weight_max=float(weights.max()),
    )
    return (
        started,
        request_rows,
        reads_per_block,
        features,
        category_columns,
        weights,
        average_reuse,
        record,
    )


def fit_tree_model(data, cutoff_us, settings):
    """Choose a tree count on earlier data, then fit at the current cutoff."""
    check_condition(
        settings["model"] == "tree"
        and settings["objective"] == "poisson"
        and settings["sample_weight"] == "recency24h"
        and settings["window_hours"] == 0,
        "Expected the recorded Poisson tree with recency24h weights and no time window",
    )
    (
        started,
        request_rows,
        reads_per_block,
        features,
        category_columns,
        weights,
        average_reuse,
        record,
    ) = prepare_training_arrays(data, cutoff_us, settings)
    import lightgbm as lgb

    # Translate model_settings.json into LightGBM settings; keep training deterministic.
    parameters = dict(
        objective=settings["objective"],
        learning_rate=settings["learning_rate"],
        num_leaves=settings["leaves"],
        min_child_samples=settings["minimum"],
        reg_alpha=settings["l1"],
        reg_lambda=settings["l2"],
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
    earlier_cut = cutoff_us - VALIDATION_HOURS
    earlier_rows, earlier_reuse = data.get_training_rows_and_reuse(earlier_cut)
    validation_rows = data.arrival_times_us[request_rows] >= earlier_cut

    tree_count = settings["max_rounds"]
    errors = []
    if (
        len(earlier_rows) >= 2 * settings["minimum"]
        and validation_rows.any()
        and earlier_reuse.sum() > 0
    ):
        earlier_features = data.feature_table(earlier_rows, settings["features"])
        category_values = build_category_values(earlier_features, category_columns)
        earlier_weights = compute_training_weights(data, earlier_rows, earlier_cut)
        fitted = lgb.LGBMRegressor(n_estimators=tree_count, **parameters)
        fitted.fit(
            encode_tree_features(earlier_features, category_values),
            earlier_reuse,
            sample_weight=earlier_weights,
            eval_set=[
                (
                    encode_tree_features(features[validation_rows], category_values),
                    reads_per_block[validation_rows],
                )
            ],
            eval_sample_weight=[weights[validation_rows]],
            eval_metric=settings["objective"],
            categorical_feature=category_columns,
        )

        # Entry zero is the validation error after the first tree.
        errors = fitted.evals_result_["valid_0"][settings["objective"]]
        check_condition(np.isfinite(errors).all(), "Nonfinite chronological validation curve")
        tree_count = int(np.argmin(errors)) + 1

    # Fit a fresh model at the current cutoff using the chosen number of trees.
    # With too little data or no observed reuse, use the weighted average.
    model = dict(kind="global", mean=average_reuse)
    if len(reads_per_block) >= 2 * settings["minimum"] and reads_per_block.sum() > 0:
        category_values = build_category_values(features, category_columns)
        fitted = lgb.LGBMRegressor(n_estimators=tree_count, **parameters)
        fitted.fit(
            encode_tree_features(features, category_values),
            reads_per_block,
            sample_weight=weights,
            categorical_feature=category_columns,
        )
        model = dict(
            kind="expanded_tree",
            mean=average_reuse,
            fit=fitted,
            vocab=category_values,
            transform="none",
        )

    record.update(
        inner_fit_cut_us=earlier_cut,
        inner_validation_us=VALIDATION_HOURS,
        inner_curve=errors,
        tree_count=tree_count,
        parameters=parameters,
        target_transform="none",
        validation_metric=settings["objective"],
        clipped_negative_predictions=True,
    )
    return record_fit_details(model, features, reads_per_block, started, record)


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


def record_fit_details(model, features, reads_per_block, started, record):
    """Check sample predictions and record fit time and model size."""
    record["fit_seconds"] = time.monotonic() - started
    sample_rows = np.arange(
        0, len(reads_per_block), max(1, len(reads_per_block) // 50000), dtype=int
    )
    predictions = predict_reuse(model, features[sample_rows])
    check_condition(
        np.isfinite(predictions).all() and np.all(predictions >= 0),
        "Invalid training predictions",
    )
    record["prediction_bins"] = np.unique(
        np.quantile(predictions, np.arange(1, 10) / 10)
    ).tolist()
    record["model_bytes"] = len(pickle.dumps(model, protocol=5))
    return model, record


def fit_and_predict_one_hour(data, cutoff_us, output_directory):
    """Train on earlier observations and predict reuse for requests in this hour."""
    # Set up the output folder and load the model settings.
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    recipe = read_training_json(MODEL_DIRECTORY, "training_recipe.json", "PLAN.json")["recipe"]

    # Fit using only observations from before this hour.
    model, fit_summary = fit_tree_model(data, cutoff_us, recipe)
    training_rows, reads_per_block = data.get_training_rows_and_reuse(cutoff_us)
    assert len(training_rows) > 0

    # Recount the reads from scratch to check both sets of labels.
    for check_cut in [cutoff_us, cutoff_us - 6 * H]:
        checked_rows, checked_targets = data.get_training_rows_and_reuse(check_cut)
        expected_rows = np.flatnonzero(
            (data.arrival_times_us < check_cut) & (data.offer_block_counts > 0)
        )
        direct_counts = data.hourly_read_counts[: check_cut // H].sum(axis=0, dtype=np.int64)
        assert np.array_equal(checked_rows, expected_rows) and np.array_equal(
            checked_targets,
            direct_counts[expected_rows] / data.offer_block_counts[expected_rows],
        )

    # Find the requests in this hour. The prediction uses only their four features.
    hour_end = min(cutoff_us + H, 168 * H)
    first_row = int(np.searchsorted(data.arrival_times_us, cutoff_us))
    stop_row = int(np.searchsorted(data.arrival_times_us, hour_end))
    prediction_rows = np.arange(first_row, stop_row)
    predictions = predict_reuse(
        model,
        data.feature_table(prediction_rows, recipe["features"]),
    )
    assert np.all(np.isfinite(predictions)) and np.all(predictions >= 0)

    # Save the model, the predictions and a summary of the fit.
    with (output_directory / "model.pkl").open("xb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    predictions.astype("<f8").tofile(output_directory / "segment.f64")

    fit_summary.update(
        status="PASS_CAUSAL_FIXED_RECIPE_FIT",
        label="cutoff-censored observed " + data.label_kind + " reuse per distinct block",
        target=data.label_kind,
        cut_us=cutoff_us,
        label_available_strictly_before_us=cutoff_us,
        latest_training_request_us=int(data.arrival_times_us[training_rows].max()),
        training_rows=len(training_rows),
        training_row_hash=array_fingerprint(training_rows.astype("<i8")),
        training_target_hash=array_fingerprint(reads_per_block.astype("<f8")),
        label_source_sha256=data.label_file_sha256,
        prediction_start=first_row,
        prediction_stop=stop_row,
        model_sha256=file_fingerprint(output_directory / "model.pkl"),
        prediction_sha256=file_fingerprint(output_directory / "segment.f64"),
        future_buckets_excluded_from_training=True,
        direct_prefix_and_inner_label_audit=True,
    )
    write_json(output_directory / "model_fit_summary.json", fit_summary, exclusive=True)

    # The runner uses these positions to place the predictions in replay order.
    return predictions, first_row, stop_row


def array_fingerprint(values):
    """Hash the array in request order for the saved fit record."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
