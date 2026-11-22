"""Load the four-feature reuse model and predict later accesses per KV block.

Checks model metadata, extracts native tree text and applies the saved user
encoding. Pickle opcodes are read as data rather than executed. The caller
supplies the model folder and its expected file hashes."""

from pathlib import Path
import hashlib, json, pickletools
import numpy as np
from user_id_encoding import encode_tree_features

FEATURES = (
    "user",
    "causal_request_position",
    "session_age_seconds",
    "previous_session_gap_seconds",
)
CUT = 259200000000


def model_fit_summary_path(model_directory):
    """Find the fit summary in a new or an archived model folder"""
    summary_file = model_directory / "model_fit_summary.json"
    if summary_file.exists():
        return summary_file
    return model_directory / "COMPLETE.json"


class SuffixReusePredictor:
    """Hold the trained trees and the user-ID mapping used during training."""

    def __init__(self, predictor_settings):
        """Verify the model record and load native tree text without executing pickle."""
        from lightgbm import Booster

        model_directory = Path(predictor_settings["directory"])
        summary_bytes = model_fit_summary_path(model_directory).read_bytes()
        assert hashlib.sha256(summary_bytes).hexdigest() == predictor_settings["receipt_sha256"]
        fit_summary = json.loads(summary_bytes)
        self.cut_us = fit_summary["cut_us"]
        self.features = FEATURES
        assert (
            self.cut_us == predictor_settings["cut_us"] == CUT
            and tuple(fit_summary["expanded_recipe"]["features"]) == FEATURES
        )
        assert (
            fit_summary["target"] == "offered_suffix"
            and fit_summary["target_transform"] == "none"
        )
        assert (
            fit_summary["latest_training_request_us"] < CUT
            and fit_summary["label_available_strictly_before_us"] == CUT
        )
        assert (
            fit_summary["direct_prefix_and_inner_label_audit"]
            and fit_summary["future_buckets_excluded_from_training"]
        )
        model_bytes = (model_directory / "model.pkl").read_bytes()
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        assert self.model_sha256 == fit_summary["model_sha256"]
        pickle_operations = list(pickletools.genops(model_bytes))
        trees = [
            arg
            for operation, arg, pos in pickle_operations
            if isinstance(arg, str) and arg.startswith("tree\nversion=")
        ]
        assert len(trees) == 1
        start = next(
            operation_index
            for operation_index, (operation, arg, pos) in enumerate(pickle_operations)
            if arg == "vocab"
        )
        stop = next(
            operation_index
            for operation_index in range(start + 1, len(pickle_operations))
            if pickle_operations[operation_index][1] == "transform"
        )
        segment = pickle_operations[start:stop]
        arrays = [
            bytes(arg)
            for operation, arg, pos in segment
            if operation.name in ("BYTEARRAY8", "BINBYTES", "SHORT_BINBYTES")
        ]
        assert (
            len(arrays) == 1
            and sum(operation.name == "NONE" for operation, arg, pos in segment) == 6
        )
        strings = [argument for operation, argument, pos in segment if isinstance(argument, str)]
        assert "f8" in strings and "<" in strings and "C" in strings
        values = np.frombuffer(arrays[0], dtype="<f8").copy()
        assert len(values) > 0 and np.isfinite(values).all() and (values[1:] > values[:-1]).all()
        self.vocab = [values, None, None, None]
        self.booster = Booster(model_str=trees[0])
        assert self.booster.num_trees() == fit_summary["tree_count"]
        self.native_model_sha256 = hashlib.sha256(trees[0].encode()).hexdigest()
        self.receipt = fit_summary

    def predict_suffix_reuse(self, rows, now_us):
        """Encode the supplied request features and predict reads per offered block."""
        assert now_us >= self.cut_us and all(
            set(request_features) == set(FEATURES) for request_features in rows
        )
        feature_table = np.array(
            [[request[name] for name in FEATURES] for request in rows], dtype=np.float64
        )
        out = np.maximum(
            np.asarray(
                self.booster.predict(
                    encode_tree_features(feature_table, self.vocab), num_threads=1
                )
            ),
            0,
        )
        assert out.shape == (len(rows),) and np.isfinite(out).all()
        return out
