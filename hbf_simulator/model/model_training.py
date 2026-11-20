"""Train the predictor and score requests one hour at a time

RequestData gets the examples and reuse counts available at a cutoff
fit_tree_model picks a tree count using earlier requests and fits the trees
predict_reuse returns one reads per block estimate for each new request
fit_hourly_model runs these steps and saves the model and predictions
"""

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np

from tree_encoding import build_category_values, encode_tree_features
from file_helpers import CODE, ROOT, read_json, write_json, file_fingerprint

# Request times are microseconds. Both names denote one hour in these units.
HOUR = 3_600_000_000
H = HOUR
VALIDATION_HOURS = 6 * HOUR


class RequestData:
    """Keep the request inputs and reuse counts in the same request order"""

    def __init__(self, label_dir=None):
        """Load checked features and the saved counts of later reads"""
        # Find the recipe and the files it uses
        plan = read_json(CODE / "PLAN.json")
        paths = read_json(ROOT / "RESOLVED_MODEL_INPUTS.json")
        inputs = read_json(CODE / "INPUTS.json")

        # Load one arrival time per request.
        self.ts = np.fromfile(inputs["raw"]["_ts.i64"]["path"], "<i8")
        self.n = len(self.ts)

        # Load the named feature arrays; matching positions describe the same request.
        feature_names = plan["recipe"]["features"]
        with np.load(paths["features"]["path"]) as feature_file:
            self.features = {name: feature_file[name] for name in feature_names}

        # Load feature types, including categorical user ID, and check they are allowed.
        descriptions = read_json(paths["feature_registry"]["path"])["features"]
        self.schema = {feature["name"]: feature for feature in descriptions}
        assert all(self.schema[name]["status"] == "ALLOWED" for name in feature_names)

        # Open offered-block counts c[i] and observed reads Y[hour, i] after checking their audit.
        if label_dir is not None:
            folder = Path(label_dir)
            assert (
                read_json(folder / "AUDIT.json")["status"]
                == "PASS_FULL_INDEPENDENT_SUFFIX_LABEL_AUDIT"
            )
            self.c = np.fromfile(folder / "c.i32", "<i4")
            self.Y = np.memmap(folder / "Y.u64", "<u8", "r", shape=(168, self.n))
            self.label_hash = file_fingerprint(folder / "Y.u64")
            self.label_kind = "offered_suffix"
        # The full rebuild uses initial trace-derived counts when no label folder is supplied.
        else:
            self.c = np.fromfile(paths["original_offer_counts"]["path"], "<i4")
            self.Y = np.memmap(
                paths["hourly_labels"]["path"], "<i4", "r", shape=(168, self.n)
            )
            self.label_hash = paths["hourly_labels"]["sha256"]
            self.label_kind = "original_increment"

        # Start the running observed-read totals at zero.
        self.latest = 0
        self.observed = np.zeros(self.n, dtype=np.int64)
        assert len(self.c) == self.n and np.all(self.c >= 0)

    def training_examples(self, cut):
        """Return request positions and reads per block known by this cutoff"""
        # Convert the hourly cutoff into a row boundary in the read-count table.
        assert cut % H == 0 and 0 <= cut <= 168 * H
        cutoff_hour = cut // H

        # Add newly completed hours, excluding reads at or after the cutoff.
        if cutoff_hour > self.latest:
            new_hours = self.Y[self.latest : cutoff_hour]
            self.observed += new_hours.sum(axis=0, dtype=np.int64)
            self.latest = cutoff_hour

        # For an earlier validation cutoff, subtract later hours without changing stored totals.
        observed_reads = self.observed
        if cutoff_hour < self.latest:
            later_hours = self.Y[cutoff_hour : self.latest]
            observed_reads = observed_reads - later_hours.sum(axis=0, dtype=np.int64)

        # Select earlier offers and divide observed reads by offered blocks to form training labels.
        offered_blocks = self.c
        training_rows = np.flatnonzero((self.ts < cut) & (offered_blocks > 0))
        reads_per_block = observed_reads[training_rows] / offered_blocks[training_rows]
        assert np.all(np.isfinite(reads_per_block)) and np.all(reads_per_block >= 0)
        assert not len(training_rows) or self.ts[training_rows].max() < cut
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
            if self.schema[name]["encoding"] == "categorical"
        ]


# Stop with a readable message when a required check fails.
def check_condition(condition, message):
    if not condition:
        raise ValueError(message)


def training_weights(data, request_rows, cut):
    """Give requests half as much weight for each extra day since arrival"""
    weights = np.exp2(
        -(cut - np.asarray(data.ts[request_rows], dtype=float)) / (24 * HOUR)
    )
    check_condition(
        np.all(np.isfinite(weights)) and np.all(weights > 0),
        "Invalid training weights",
    )

    return weights / weights.mean() if len(weights) else weights


# Load earlier examples and the matching features and weights.
def prepare_training_arrays(data, cut, settings):
    """Prepare the examples available before this hourly cutoff."""
    started = time.monotonic()
    request_rows, reads_per_block = data.training_examples(cut)
    check_condition(
        len(reads_per_block) > 0
        and np.all(np.isfinite(reads_per_block))
        and np.all(reads_per_block >= 0),
        "No finite nonnegative training labels",
    )

    feature_names = settings["features"]
    features = data.feature_table(request_rows, feature_names)
    category_columns = data.categorical_columns(feature_names)
    weights = training_weights(data, request_rows, cut)
    average_reuse = float(np.average(reads_per_block, weights=weights))

    # Record which examples, weights, and settings entered this fit.
    record = dict(
        cut_us=cut,
        train_rows=len(request_rows),
        target_available_no_later_than_us=cut,
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


# Choose the tree count on an earlier period, then refit at the current cutoff.
def fit_tree_model(data, cut, settings):
    """Fit the recorded tree recipe using only earlier observations."""
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
    ) = prepare_training_arrays(data, cut, settings)
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

    # The earlier fit sees only reads before earlier_cut. Validation uses reads before cut.
    earlier_cut = cut - VALIDATION_HOURS
    earlier_rows, earlier_reuse = data.training_examples(earlier_cut)
    validation_rows = data.ts[request_rows] >= earlier_cut
    tree_count, errors = settings["max_rounds"], []
    if (
        len(earlier_rows) >= 2 * settings["minimum"]
        and validation_rows.any()
        and earlier_reuse.sum() > 0
    ):
        earlier_features = data.feature_table(earlier_rows, settings["features"])
        category_values = build_category_values(earlier_features, category_columns)
        earlier_weights = training_weights(data, earlier_rows, earlier_cut)
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

        errors = fitted.evals_result_["valid_0"][settings["objective"]]
        check_condition(
            np.isfinite(errors).all(), "Nonfinite chronological validation curve"
        )
        tree_count = int(np.argmin(errors)) + 1

    # Keep the weighted mean when there are too few examples or no observed reuse.
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


# Predict from features using the tree or its low-data fallback.
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


# Check sample predictions and record fitting time and saved model size.
def record_fit_details(model, features, reads_per_block, started, record):
    """Check and describe the completed fit."""
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


# FIT AND PREDICT ONE HOUR: the experiment runner calls this function.
def fit_hourly_model(data, cut, out):
    """Train on earlier observations and predict reuse for requests in this hour."""
    # 1. Set up a fresh output folder and load the model recipe.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    recipe = read_json(CODE / "PLAN.json")["recipe"]

    # 2. Call fit_tree_model() to choose the tree count and fit on earlier observations.
    model, record = fit_tree_model(data, cut, recipe)
    training_rows, reads_per_block = data.training_examples(cut)
    assert len(training_rows) > 0

    # 3. Independently recount labels at the fitting and earlier validation cutoffs.
    for check_cut in [cut, cut - 6 * H]:
        checked_rows, checked_targets = data.training_examples(check_cut)
        expected_rows = np.flatnonzero((data.ts < check_cut) & (data.c > 0))
        direct_counts = data.Y[: check_cut // H].sum(axis=0, dtype=np.int64)
        assert np.array_equal(checked_rows, expected_rows) and np.array_equal(
            checked_targets, direct_counts[expected_rows] / data.c[expected_rows]
        )

    # 4. Predict requests in [cut, cut+H) from their features.
    hour_end = min(cut + H, 168 * H)
    first_row = int(np.searchsorted(data.ts, cut))
    stop_row = int(np.searchsorted(data.ts, hour_end))
    prediction_rows = np.arange(first_row, stop_row)
    predictions = predict_reuse(
        model,
        data.feature_table(prediction_rows, recipe["features"]),
    )
    assert np.all(np.isfinite(predictions)) and np.all(predictions >= 0)

    # 5. Save the fitted model and current-hour prediction array.
    with (out / "model.pkl").open("xb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    predictions.astype("<f8").tofile(out / "segment.f64")

    # 6. Record the cutoff, training examples, checks, and output fingerprints.
    record.update(
        status="PASS_CAUSAL_FIXED_RECIPE_FIT",
        label="cutoff-censored observed "
        + data.label_kind
        + " reuse per distinct block",
        target=data.label_kind,
        cut_us=cut,
        label_available_strictly_before_us=cut,
        latest_training_request_us=int(data.ts[training_rows].max()),
        training_rows=len(training_rows),
        training_row_hash=array_fingerprint(training_rows.astype("<i8")),
        training_target_hash=array_fingerprint(reads_per_block.astype("<f8")),
        label_source_sha256=data.label_hash,
        prediction_start=first_row,
        prediction_stop=stop_row,
        model_sha256=file_fingerprint(out / "model.pkl"),
        prediction_sha256=file_fingerprint(out / "segment.f64"),
        future_buckets_excluded_from_training=True,
        direct_prefix_and_inner_label_audit=True,
    )
    write_json(out / "COMPLETE.json", record, exclusive=True)

    # 7. Return predictions and their request positions for the runner's full scores array.
    return predictions, first_row, stop_row


# RECORD AN ARRAY FINGERPRINT: support the checks saved with each fit.
def array_fingerprint(values):
    # Hash ordered array contents for the saved fit record.
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
