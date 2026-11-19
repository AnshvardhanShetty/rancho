"""Check synthetic inputs and controller parity without GPU execution.

These fixtures test accounting and packaging, not benchmark performance.
"""

import json
from pathlib import Path
import unittest

from controlled_workload import prompts, records, eligibility
from updated_policy import BandWallet, FilteredHistory
from reference_wallet_check import BandWallet as ReferenceWallet


class PublicInputs(unittest.TestCase):
    def test_eight_context_workload(self):
        setup = json.loads(
            (Path(__file__).resolve().parents[1] / "synthetic_setup.json").read_text()
        )
        payload = prompts(setup)
        earlier, requests = records(setup)
        bits = eligibility(setup)
        self.assertEqual(len(payload), 1208)
        self.assertEqual(len(earlier), 128)
        self.assertEqual(len(requests), len(payload))
        self.assertTrue(all(len(p) == 1968 for p in payload))
        self.assertTrue(all(sum(b) == 117 for b in bits[:8]))
        self.assertTrue(all(all(b) and len(b) == 123 for b in bits[8:]))
        self.assertEqual(sum(map(sum, bits[8:])), 147600)

    def test_history_updates_after_decision(self):
        history = FilteredHistory(1)
        first = history.prepare("first", ["A", "B"], 1, 0)
        self.assertEqual(first["past_filtered_reuses"], 0)
        history.commit(first)
        second = history.prepare("second", ["A", "B"], 1, 0)
        self.assertEqual(second["past_filtered_reuses"], 0)
        self.assertEqual(second["pending"], ["A", "B"])
        history.commit(second)
        self.assertEqual(
            history.prepare("third", ["A", "B"], 1, 0)["past_filtered_reuses"], 2
        )

    def test_cutoffs_match_independent_reference(self):
        from controllers import Observation, DAY
        from controllers_reference import Observation as ReferenceObservation

        live = BandWallet("hbf", 50000, 17, end_us=DAY, release_us=DAY, reserve=0.0)
        reference = ReferenceWallet(
            "hbf", 50000, 17, end_us=DAY, release_us=DAY, reserve=0.0
        )
        for i in range(1200):
            values = (i, float(i % 37), 1 << (i % 4), i * 1000000, 1 << (i % 4))
            live._observe(Observation(*values))
            reference._observe(ReferenceObservation(*values))
        live.recompute(1200000000)
        reference.recompute(1200000000)
        self.assertEqual(live.bandcuts, reference.bandcuts)
        self.assertEqual(
            (live.threshold, live.tie), (reference.threshold, reference.tie)
        )


if __name__ == "__main__":
    unittest.main()
