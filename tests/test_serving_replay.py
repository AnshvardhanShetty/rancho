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

ROOT = Path(__file__).resolve().parents[1]


class ServingReplayTest(unittest.TestCase):
    def test_default_workload_counts_and_write_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.json"
            command = [sys.executable, str(ROOT / "examples/serving_replay.py"),
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
            before = output.read_bytes()
            repeated = subprocess.run(command, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
