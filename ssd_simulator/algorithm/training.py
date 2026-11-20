"""Fit the recorded eight-input SSD tree using only earlier observed outcomes.

The tree-count check uses an earlier six-hour period. Higher access/eviction counts
are constrained not to lower predictions. Raw-label squared error is the selected fit.
"""

import time
import numpy as np
from .training_data import validation_start_time, select_validation_rows
from .settings import validate_recorded_settings


def build_category_values(feature_values, category_columns):
    """Remember category values seen in the training examples."""
    return [
        (
            np.unique(
                feature_values[
                    np.isfinite(feature_values[:, column])
                    & (feature_values[:, column] >= 0),
                    column,
                ]
            )
            if column in category_columns
            else None
        )
        for column in range(feature_values.shape[1])
    ]


def encode_tree_features(feature_values, vocab):
    """Use the same category codes during fitting and prediction."""
    encoded_features = np.where(
        np.isfinite(feature_values) & (feature_values >= 0), feature_values, np.nan
    )
    for column, known_categories in enumerate(vocab):
        if known_categories is None:
            continue
        encoded_features[:, column] = -1
        if len(known_categories):
            category_indices = np.searchsorted(
                known_categories, feature_values[:, column]
            )
            known_category_mask = category_indices < len(known_categories)
            known_category_mask &= (
                known_categories[
                    np.minimum(category_indices, len(known_categories) - 1)
                ]
                == feature_values[:, column]
            )
            encoded_features[known_category_mask, column] = category_indices[
                known_category_mask
            ]
    return encoded_features


def predict_saved_reads(model, feature_values):
    """Predict marginal reuse from the supplied feature rows."""
    eviction_count, average_reuse, kind = (
        len(feature_values),
        model["mean"],
        model["kind"],
    )
    if kind == "global":
        predicted_reuse = np.full(eviction_count, average_reuse)
    elif kind == "tree":
        predicted_reuse = model["fit"].predict(
            encode_tree_features(feature_values, model["vocab"]), num_threads=1
        )
        if model["target"] == "log1p":
            predicted_reuse = np.expm1(np.minimum(predicted_reuse, 700))
    else:
        raise ValueError(kind)
    predicted_reuse = np.maximum(np.asarray(predicted_reuse, dtype=np.float64), 0)
    if (
        predicted_reuse.shape != (eviction_count,)
        or not np.isfinite(predicted_reuse).all()
    ):
        raise ValueError("Invalid predictor output")
    return predicted_reuse


def fit_eviction_reuse_model(data, cutoff, config):
    """Choose a tree count on earlier data, then fit all usable examples."""
    started = time.monotonic()
    config = validate_recorded_settings(config)
    example_ids, observed_reuse, weights = data.training_examples(cutoff, config)
    feature_values, category_columns = (
        data.feature_table(example_ids, config),
        data.categorical_columns(config),
    )
    average_reuse = (
        float(np.average(observed_reuse, weights=weights))
        if len(observed_reuse)
        else config["cold_prior"]
    )
    kind = config["model"]
    model = dict(kind="global", mean=average_reuse)
    example_times = data.decision_times(example_ids)
    record = dict(
        cut_us=int(cutoff),
        train_rows=len(example_ids),
        label_horizon_hours=config["horizon_hours"],
        cohort_lag_hours=config["cohort_lag_hours"],
        config=config,
        min_train_decision_us=int(example_times.min()) if len(example_times) else None,
        max_train_decision_us=int(example_times.max()) if len(example_times) else None,
        cold_start=not len(example_ids),
        training_target=(
            "cumulative_HBM_remisses_strictly_before_cutoff"
            if config["target_mode"] == "cumulative"
            else "future_repeated_hbm_miss_count"
        ),
    )
    if len(example_ids) and feature_values.shape[1] and (kind == "tree"):
        import lightgbm as lgb

        target_name = "raw"
        transform = np.asarray
        metric = "l2"
        params = dict(
            objective="regression",
            learning_rate=config["learning_rate"],
            num_leaves=config["leaves"],
            min_child_samples=config["minimum"],
            n_jobs=1,
            verbosity=-1,
            random_state=config["seed"],
            deterministic=True,
            force_col_wise=True,
            bagging_fraction=1.0,
            feature_fraction=1.0,
        )
        params["monotone_constraints"] = config["monotone_constraints"]
        inner = validation_start_time(
            cutoff, config["cohort_lag_hours"], config["inner_validation_hours"]
        )
        earlier_ids, earlier_reuse, earlier_weights = data.training_examples(
            inner, config
        )
        validation_rows = select_validation_rows(
            example_times, cutoff, inner, config["cohort_lag_hours"]
        )
        if np.intersect1d(example_ids[validation_rows], earlier_ids).size:
            raise ValueError("Inner train/validation event overlap")
        tree_count, validation_errors = (config["max_rounds"], [])
        if (
            len(earlier_ids) >= 2 * config["minimum"]
            and validation_rows.any()
            and (earlier_reuse.sum() > 0)
        ):
            earlier_features = data.feature_table(earlier_ids, config)
            vocab = build_category_values(earlier_features, category_columns)
            estimator = lgb.LGBMRegressor(n_estimators=config["max_rounds"], **params)
            estimator.fit(
                encode_tree_features(earlier_features, vocab),
                transform(earlier_reuse),
                sample_weight=earlier_weights,
                eval_set=[
                    (
                        encode_tree_features(feature_values[validation_rows], vocab),
                        transform(observed_reuse[validation_rows]),
                    )
                ],
                eval_sample_weight=[weights[validation_rows]],
                eval_metric=metric,
                categorical_feature=category_columns,
            )
            validation_errors = list(estimator.evals_result_["valid_0"][metric])
            if not np.isfinite(validation_errors).all():
                raise ValueError("Nonfinite inner validation loss")
            tree_count = int(np.argmin(validation_errors)) + 1
        if len(example_ids) >= 2 * config["minimum"] and observed_reuse.sum() > 0:
            vocab = build_category_values(feature_values, category_columns)
            estimator = lgb.LGBMRegressor(n_estimators=tree_count, **params)
            estimator.fit(
                encode_tree_features(feature_values, vocab),
                transform(observed_reuse),
                sample_weight=weights,
                categorical_feature=category_columns,
            )
            model = dict(
                kind=kind,
                mean=average_reuse,
                fit=estimator,
                vocab=vocab,
                target=target_name,
            )
        record.update(
            inner_cut_us=int(inner),
            inner_train_rows=len(earlier_ids),
            inner_validation_rows=int(validation_rows.sum()),
            inner_curve=validation_errors,
            tree_count=tree_count,
            strict_chronological_inner_validation=True,
        )
    model["seen_users"] = (
        np.unique(data.user_ids(example_ids)) if len(example_ids) else np.empty(0)
    )
    model["seen_users"] = model["seen_users"][
        np.isfinite(model["seen_users"]) & (model["seen_users"] >= 0)
    ]
    model["config"] = config
    model["data_identity_sha256"] = data.identity
    model["cut_us"] = int(cutoff)
    probe = np.arange(
        0, len(example_ids), max(1, len(example_ids) // 50000), dtype=np.int64
    )
    predicted = predict_saved_reads(model, feature_values[probe])
    record["prediction_bins"] = (
        np.unique(np.quantile(predicted, np.arange(0.1, 1, 0.1))).tolist()
        if len(predicted)
        else []
    )
    record["fit_seconds"] = time.monotonic() - started
    return (model, record)


def predict_reuse_for_evictions(data, model, indices):
    """Score eviction rows directly; an empty hour has no predictions."""
    example_ids = np.asarray(indices, dtype=np.int64)
    if not len(example_ids):
        return (np.empty(0, dtype=np.float64), 0)
    return (
        predict_saved_reads(model, data.feature_table(example_ids, model["config"])),
        len(example_ids),
    )
