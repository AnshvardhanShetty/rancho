"""Check synthetic inputs and controller parity without GPU execution.

These fixtures test accounting and packaging, not benchmark performance.
"""

import json
from pathlib import Path
import unittest

from synthetic_requests import (
    build_workload_prompts,
    build_request_feature_records,
    build_eligible_read_mask,
)
from hbf_admission import CostBandWriteController, HbmMissReuseHistory
from reference_wallet_check import CostBandWriteController as ReferenceWallet


class PublicInputs(unittest.TestCase):
    def test_eight_context_workload(self):
        setup = json.loads(
            (Path(__file__).resolve().parents[1] / "synthetic_setup.json").read_text()
        )
        payload = build_workload_prompts(setup)
        earlier, requests = build_request_feature_records(setup)
        bits = build_eligible_read_mask(setup)
        self.assertEqual(len(payload), 1208)
        self.assertEqual(len(earlier), 128)
        self.assertEqual(len(requests), len(payload))
        self.assertTrue(all(len(p) == 1968 for p in payload))
        self.assertTrue(all(sum(b) == 117 for b in bits[:8]))
        self.assertTrue(all(all(b) and len(b) == 123 for b in bits[8:]))
        self.assertEqual(sum(map(sum, bits[8:])), 147600)

    def test_history_updates_after_decision(self):
        history = HbmMissReuseHistory(1)
        first = history.collect_request_reuse("first", ["A", "B"], 1, 0)
        self.assertEqual(first["past_filtered_reuses"], 0)
        history.record_request_reuse(first)
        second = history.collect_request_reuse("second", ["A", "B"], 1, 0)
        self.assertEqual(second["past_filtered_reuses"], 0)
        self.assertEqual(second["pending"], ["A", "B"])
        history.record_request_reuse(second)
        self.assertEqual(
            history.collect_request_reuse("third", ["A", "B"], 1, 0)["past_filtered_reuses"],
            2,
        )

    def test_cutoffs_match_independent_reference(self):
        from write_controller import Observation, DAY
        from reference_write_controller import Observation as ReferenceObservation

        live = CostBandWriteController("hbf", 50000, 17, end_us=DAY, release_us=DAY, reserve=0.0)
        reference = ReferenceWallet("hbf", 50000, 17, end_us=DAY, release_us=DAY, reserve=0.0)
        for i in range(1200):
            values = (i, float(i % 37), 1 << (i % 4), i * 1000000, 1 << (i % 4))
            live._record_offer(Observation(*values))
            reference._record_offer(ReferenceObservation(*values))
        live.update_admission_cutoffs(1200000000)
        reference.update_admission_cutoffs(1200000000)
        self.assertEqual(live.cutoffs_by_cost_band, reference.cutoffs_by_cost_band)
        self.assertEqual(
            (live.threshold, live.tie_probability),
            (reference.threshold, reference.tie_probability),
        )


if __name__ == "__main__":
    unittest.main()
