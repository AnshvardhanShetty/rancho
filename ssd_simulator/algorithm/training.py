"""Fit the recorded eight-input SSD tree using only earlier observed outcomes.

The tree-count check uses an earlier six-hour period. Higher access/eviction counts
are constrained not to lower predictions. Raw-label squared error is the selected fit.
"""

import time
import numpy as np
from .training_data import chronological_inner_cut, validation_mask
from .settings import normalize


def vocab_fit(feature_values, category_columns):
    """Remember category values seen in the training examples."""
    return [
        (
            np.unique(
                feature_values[
                    np.isfinite(feature_values[:, i]) & (feature_values[:, i] >= 0), i
                ]
            )
            if i in category_columns
            else None
        )
        for i in range(feature_values.shape[1])
    ]


def vocab_encode(feature_values, vocab):
    """Use the same category codes during fitting and prediction."""
    z = np.where(
        np.isfinite(feature_values) & (feature_values >= 0), feature_values, np.nan
    )
    for i, v in enumerate(vocab):
        if v is None:
            continue
        z[:, i] = -1
        if len(v):
            at = np.searchsorted(v, feature_values[:, i])
            good = at < len(v)
            good &= v[np.minimum(at, len(v) - 1)] == feature_values[:, i]
            z[good, i] = at[good]
    return z


def predict(model, feature_values):
    """Predict marginal reuse from the supplied feature rows."""
    n, average_reuse, kind = (len(feature_values), model["mean"], model["kind"])
    if kind == "global":
        out = np.full(n, average_reuse)
    elif kind == "tree":
        out = model["fit"].predict(
            vocab_encode(feature_values, model["vocab"]), num_threads=1
        )
        if model["target"] == "log1p":
            out = np.expm1(np.minimum(out, 700))
    else:
        raise ValueError(kind)
    out = np.maximum(np.asarray(out, dtype=np.float64), 0)
    if out.shape != (n,) or not np.isfinite(out).all():
        raise ValueError("Invalid predictor output")
    return out


def fit_model(data, cutoff, config):
    """Choose a tree count on earlier data, then fit all usable examples."""
    started = time.monotonic()
    config = normalize(config)
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
        inner = chronological_inner_cut(
            cutoff, config["cohort_lag_hours"], config["inner_validation_hours"]
        )
        earlier_ids, earlier_reuse, earlier_weights = data.training_examples(
            inner, config
        )
        validation_rows = validation_mask(
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
            vocab = vocab_fit(earlier_features, category_columns)
            estimator = lgb.LGBMRegressor(n_estimators=config["max_rounds"], **params)
            estimator.fit(
                vocab_encode(earlier_features, vocab),
                transform(earlier_reuse),
                sample_weight=earlier_weights,
                eval_set=[
                    (
                        vocab_encode(feature_values[validation_rows], vocab),
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
            vocab = vocab_fit(feature_values, category_columns)
            estimator = lgb.LGBMRegressor(n_estimators=tree_count, **params)
            estimator.fit(
                vocab_encode(feature_values, vocab),
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
    predicted = predict(model, feature_values[probe])
    record["prediction_bins"] = (
        np.unique(np.quantile(predicted, np.arange(0.1, 1, 0.1))).tolist()
        if len(predicted)
        else []
    )
    record["fit_seconds"] = time.monotonic() - started
    return (model, record)


def predict_events(data, model, indices):
    """Score eviction rows directly; an empty hour has no predictions."""
    example_ids = np.asarray(indices, dtype=np.int64)
    if not len(example_ids):
        return (np.empty(0, dtype=np.float64), 0)
    return (
        predict(model, data.feature_table(example_ids, model["config"])),
        len(example_ids),
    )
