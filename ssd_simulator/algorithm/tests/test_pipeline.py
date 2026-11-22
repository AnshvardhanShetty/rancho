"""Check cutoff safety, encoding, label counting and source-registration settings.

Inputs are tiny synthetic arrays. These checks do not replay the private trace.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ssd_simulator.algorithm.recomputation_counts import count_eligible_reads_saved_and_compare
from ssd_simulator.algorithm.reuse_labels import build_hourly_saved_read_counts
from ssd_simulator.algorithm.experiment_settings import RECORDED, SETTINGS, HOUR
from ssd_simulator.algorithm.model_training import (
    fit_eviction_reuse_model,
    predict_saved_reads,
    encode_tree_features,
    build_category_values,
)
from ssd_simulator.algorithm.training_data import EVICTION_RECORD_DTYPE, TrainingData

HERE = Path(__file__).resolve().parents[1]


class PipelineChecks(unittest.TestCase):

    def test_recorded_recipe_digest_and_selected_override(self):
        digest = hashlib.sha256((HERE / "recorded_settings.json").read_bytes()).hexdigest()
        proof = json.loads((HERE / "SOURCE_VERIFICATION.json").read_text())
        self.assertEqual(digest, proof["recorded_recipe_sha256"])
        self.assertEqual(SETTINGS, dict(RECORDED, objective="raw_l2"))
        self.assertEqual(SETTINGS["monotone_constraints"], [0, 0, 1, 1, 0, 0, 0, 0])

    def test_user_codes_are_reused_and_unseen_users_are_unknown(self):
        x = np.array([[4.0, 1.0], [8.0, 2.0]])
        vocabulary = build_category_values(x, [0])
        np.testing.assert_array_equal(
            encode_tree_features(np.array([[8.0, 3.0], [99.0, 4.0]]), vocabulary),
            [[1.0, 3.0], [-1.0, 4.0]],
        )

    def test_same_session_credit_is_applied_after_hbm(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            np.array([0, 4], dtype="<u8").tofile(root / "offsets.i64")
            np.array([13], dtype="<u8").tofile(root / "credit.bits")
            np.array([0, 3], dtype="<u8").tofile(root / "credit.rank")
            row = np.zeros((1, 13), dtype="<i8")
            row[0, 2:5] = [4, 1, 3]
            row.tofile(root / "ours")
            row[0, 4] = 1
            row.tofile(root / "reference")
            self.assertEqual(
                count_eligible_reads_saved_and_compare(root / "ours", root / "reference", root),
                (2, 1, 1, 0),
            )

    def test_label_read_at_cutoff_is_excluded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            np.array([0, HOUR], dtype="<i8").tofile(root / "times")
            np.save(root / "sample.npy", np.array([0], dtype=np.int64))
            np.array([[0, 0, 0, 1, 0, 0]], dtype="<i4").tofile(root / "evictions")
            np.array([[1, 0, 3]], dtype="<i4").tofile(root / "events")
            build_hourly_saved_read_counts(
                root / "times",
                root / "sample.npy",
                root / "evictions",
                [root / "events"],
                root / "observed",
            )
            y = np.fromfile(root / "observed", "<u4").reshape(169, 1)
            self.assertEqual(y[1, 0], 0)
            self.assertEqual(y[2, 0], 3)

    def test_training_examples_exclude_future_requests_and_labels(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = TrainingData.__new__(TrainingData)
            data.sampled_eviction_ids = np.arange(3)
            data.sample_times = np.array([HOUR, 2 * HOUR, 4 * HOUR])
            matrix = np.zeros((169, 3), dtype="<u4")
            matrix[3] = [2, 7, 0]
            matrix[4] = [100, 200, 300]
            matrix.tofile(root / "counts")
            data.manifest = dict(
                cumulative_labels=dict(
                    path=str(root / "counts"),
                    sha256=hashlib.sha256((root / "counts").read_bytes()).hexdigest(),
                )
            )
            ids, labels, weights = data.training_examples(3 * HOUR, SETTINGS)
            np.testing.assert_array_equal(ids, [0, 1])
            np.testing.assert_array_equal(labels, [2, 7])
            np.testing.assert_array_equal(weights, [1, 1])


if __name__ == "__main__":
    unittest.main()
