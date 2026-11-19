"""Check measured-only accounting on the bundled synthetic serving workload.

These checks concern the public synthetic predictor and prompt-cache simulator.
Matching aggregate hit ratios is not a check of GPU scheduling or serving speed.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hbf_simulator/experiments"))
from synthetic_training import SyntheticTrainingData, model_training


class ServingReplayTest(unittest.TestCase):
    def test_default_workload_counts_and_write_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.json"
            command = [sys.executable, str(ROOT / "hbf_simulator/examples/serving_replay.py"),
                       "--output", str(output)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            summary = json.loads(output.read_text())
            rows = summary["results"]
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row["eligible_accesses"], 1200 * 123)
                self.assertEqual(row["cache_hits"], 1200 * 117)
                self.assertEqual(row["prefill_blocks_recomputed"], 1200 * 6)
                self.assertLessEqual(row["flash_blocks_written"],
                                     row["write_allowance_blocks"])
                self.assertAlmostEqual(row["hit_ratio_percent"], 100 * 117 / 123)
            self.assertEqual(rows[0]["hbm_blocks"], 960)
            self.assertEqual(rows[1]["hbm_blocks"], 320)
            self.assertEqual(rows[2]["hbm_blocks"], 144)
            self.assertEqual(rows[2]["hbm_replaced_percent"], 85)
            for row in rows[1:]:
                self.assertEqual(row["training"]["model_kind"], "expanded_tree")
                self.assertGreater(row["training"]["unique_labels"], 1)
            before = output.read_bytes()
            repeated = subprocess.run(command, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), before)

    def test_research_labels_and_cutoff(self):
        hour = model_training.H
        blocks = [[10, 20, 10], [10], [20], [30], [10], [10]]
        times = np.arange(len(blocks)) * hour
        sessions = [1, 1, 1, 2, 2, 2]
        features = np.zeros((len(blocks), 1))
        initial = SyntheticTrainingData(blocks, times, sessions, features, ["user"])
        rows, labels = initial.training_examples(6 * hour)
        np.testing.assert_array_equal(rows, [0, 3, 4])
        np.testing.assert_array_equal(labels, [1, 0, 1])
        offered = SyntheticTrainingData(blocks, times, sessions, features, ["user"],
                                         offered=blocks)
        rows, labels = offered.training_examples(2 * hour)
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(labels, [0.5, 0])
        _, later = offered.training_examples(6 * hour)
        np.testing.assert_array_equal(later, [1.5, 1, 0, 0, 1, 0])
        _, earlier_again = offered.training_examples(2 * hour)
        np.testing.assert_array_equal(earlier_again, labels)


if __name__ == "__main__":
    unittest.main()
