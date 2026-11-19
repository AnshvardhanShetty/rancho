"""Stress physical-time accounting without a GPU or a trained predictor.
Assumptions: synthetic monotonic timestamps isolate accounting invariants, not
measured serving performance. Tests cover startup charging, no initial steady
credit, daily exhaustion, time-regression rejection, and rejected-write payment.
"""

import unittest
from physical_time import PhysicalBudget, DAY_NS
from controllers import Wallet, DAY


class PhysicalTests(unittest.TestCase):
    def test_zero_initial_steady_credit_and_real_refill(self):
        b = PhysicalBudget(1440, 100)
        a = b.limit(200, True)
        b.charge(100, True, a)
        self.assertEqual(b.limit(1000, False, 1000), 0)
        self.assertEqual(b.limit(1000 + 60_000_000_000, False, 1000), 1)
        b.charge(1, False, 1)
        self.assertEqual(b.limit(1000 + 60_000_000_000, False, 1000), 0)
        self.assertEqual(b.limit(1000 + 120_000_000_000, False, 1000), 1)

    def test_startup_is_charged_and_no_refunds(self):
        b = PhysicalBudget(10, 0)
        b.charge(9, True, b.limit(1, True))
        self.assertEqual(b.limit(DAY_NS // 2, False, 2), 1)
        b.charge(1, False, 1)
        self.assertEqual(b.limit(DAY_NS - 1, False, 2), 0)
        with self.assertRaises(ValueError):
            b.charge(1, False, 0)

    def test_time_and_phase_regressions_are_rejected(self):
        b = PhysicalBudget(100, 10)
        with self.assertRaises(ValueError):
            b.limit(9, True)
        with self.assertRaises(ValueError):
            b.limit(20, False, None)
        b.limit(20, False, 15)
        with self.assertRaises(ValueError):
            b.limit(21, True)
        with self.assertRaises(ValueError):
            b.limit(21, False, 16)
        with self.assertRaises(ValueError):
            b.limit(10 + DAY_NS, False, 15)

    def test_c7_observes_but_never_charges_unfunded_offer(self):
        w = Wallet("hbf", 100, 17, end_us=DAY, release_us=DAY)
        self.assertFalse(w.decide(0, 1, 10.0, distinct=8, cost=8, payment_limit=0))
        self.assertEqual(w.W, 0)
        self.assertEqual(w.observations, 1)
        self.assertTrue(w.decide(1, 2, 10.0, distinct=8, cost=8, payment_limit=8))
        self.assertEqual(w.W, 8)
        self.assertEqual(w.observations, 2)

    def test_zero_cost_reuse_does_not_consume_write_credit(self):
        w = Wallet("hbf", 100, 17, end_us=DAY, release_us=DAY)
        self.assertTrue(w.decide(0, 1, 10.0, distinct=8, cost=0, payment_limit=0))
        self.assertEqual(w.W, 0)

    def test_daily_cap_survives_large_external_credit(self):
        w = Wallet("hbf", 2, 17, end_us=DAY, release_us=DAY)
        self.assertTrue(w.decide(0, 1, 10.0, distinct=2, cost=2, payment_limit=100))
        self.assertFalse(w.decide(1, 2, 10.0, distinct=1, cost=1, payment_limit=100))
        self.assertEqual(w.W, 2)


if __name__ == "__main__":
    unittest.main()
