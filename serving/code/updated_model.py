"""Load a saved four-input tree and predict later reads per KV block.

SuffixGBT checks the supplied model files, extracts native tree text and records
user-ID codes. predict() applies those same codes and runs the trained trees.

Assumes the documented offered-suffix training format and day-four model cutoff.
Pickle contents are inspected as data, not executed. Production models are not
included; file digests detect changes but do not establish who produced a file."""

from pathlib import Path
import hashlib, json, pickletools
import numpy as np
from predictor_bridge import encode

FEATURES = (
    "user",
    "causal_request_position",
    "session_age_seconds",
    "previous_session_gap_seconds",
)
CUT = 259200000000


class SuffixGBT:
    """Hold the trained trees and the user-ID mapping used during training."""

    def __init__(self, spec):
        """Verify the model record and load native tree text without executing pickle."""
        from lightgbm import Booster

        p = Path(spec["directory"])
        raw = (p / "COMPLETE.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec["receipt_sha256"]
        r = json.loads(raw)
        self.cut_us = r["cut_us"]
        self.features = FEATURES
        assert (
            self.cut_us == spec["cut_us"] == CUT
            and tuple(r["expanded_recipe"]["features"]) == FEATURES
        )
        assert r["target"] == "offered_suffix" and r["target_transform"] == "none"
        assert (
            r["latest_training_request_us"] < CUT
            and r["label_available_strictly_before_us"] == CUT
        )
        assert (
            r["direct_prefix_and_inner_label_audit"]
            and r["future_buckets_excluded_from_training"]
        )
        raw = (p / "model.pkl").read_bytes()
        self.model_sha256 = hashlib.sha256(raw).hexdigest()
        assert self.model_sha256 == r["model_sha256"]
        ops = list(pickletools.genops(raw))
        trees = [
            arg
            for op, arg, pos in ops
            if isinstance(arg, str) and arg.startswith("tree\nversion=")
        ]
        assert len(trees) == 1
        start = next(i for i, (op, arg, pos) in enumerate(ops) if arg == "vocab")
        stop = next(i for i in range(start + 1, len(ops)) if ops[i][1] == "transform")
        segment = ops[start:stop]
        arrays = [
            bytes(arg)
            for op, arg, pos in segment
            if op.name in ("BYTEARRAY8", "BINBYTES", "SHORT_BINBYTES")
        ]
        assert (
            len(arrays) == 1 and sum(op.name == "NONE" for op, arg, pos in segment) == 6
        )
        strings = [a for op, a, pos in segment if isinstance(a, str)]
        assert "f8" in strings and "<" in strings and "C" in strings
        values = np.frombuffer(arrays[0], dtype="<f8").copy()
        assert (
            len(values) > 0
            and np.isfinite(values).all()
            and (values[1:] > values[:-1]).all()
        )
        self.vocab = [values, None, None, None]
        self.booster = Booster(model_str=trees[0])
        assert self.booster.num_trees() == r["tree_count"]
        self.native_model_sha256 = hashlib.sha256(trees[0].encode()).hexdigest()
        self.receipt = r

    def predict(self, rows, now_us):
        """Encode the supplied request features and predict reads per offered block."""
        assert now_us >= self.cut_us and all(set(r) == set(FEATURES) for r in rows)
        x = np.array([[r[n] for n in FEATURES] for r in rows], dtype=np.float64)
        out = np.maximum(
            np.asarray(self.booster.predict(encode(x, self.vocab), num_threads=1)), 0
        )
        assert out.shape == (len(rows),) and np.isfinite(out).all()
        return out
