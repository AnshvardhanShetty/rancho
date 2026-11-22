"""Check throughput, latency and configuration checks using synthetic timestamps."""

import unittest
from metrics import summarize_request_performance, validate_comparison_settings


def row(i, n, arrival, finish, first=None, last=None):
    return dict(
        row=i,
        status="COMPLETE",
        expected_output_tokens=n,
        generated_tokens=n,
        scheduled_ns=arrival * 10**9,
        dispatch_ns=arrival * 10**9,
        finish_ns=finish * 10**9,
        first_token_ns=None if first is None else first * 10**9,
        last_token_ns=None if last is None else last * 10**9,
    )


def plan():
    return dict(
        execution="real_GPU",
        simulation_only=False,
        model="Qwen/Qwen3-4B-Instruct-2507",
        exchange_ratio=8,
        warmup_requests=0,
        initial_cache="empty",
        measured_requests=200,
        baseline_HBM_bytes=10,
        RANCHO_HBM_bytes=2,
        RANCHO_HBF_bytes=64,
        HBM_accounting="shared_active_and_retained",
        native_block_bytes=1,
        cohort_sha256="test",
        arrivals_sha256="test",
        weights_accounting="test",
        controller_initial_state="test",
        policy_clock="test",
        write_allowance="test",
        write_channel="test",
        GPU_backend_certificate=dict(status="PASS_NATIVE_RANCHO_ENGINE", GPU_execution=True),
    )


class Checks(unittest.TestCase):
    def test_real_interval_and_undefined_populations(self):
        result = summarize_request_performance(
            [row(0, 3, 10, 15, 11, 13), row(1, 0, 11, 16), row(2, 1, 12, 14, 13, 13)]
        )
        self.assertEqual(result["first_arrival_to_last_completion_seconds"], 6)
        self.assertEqual(result["output_tokens_per_second"], 4 / 6)
        self.assertEqual(result["TTFT_seconds"]["n"], 2)
        self.assertEqual(result["TPOT_seconds"]["n"], 1)
        self.assertEqual(result["TPOT_seconds"]["mean"], 1)
        self.assertEqual(result["request_latency_seconds"]["n"], 3)

    def test_short_outputs_rejected(self):
        r = row(0, 3, 0, 4, 1, 3)
        r["generated_tokens"] = 2
        with self.assertRaises(ValueError):
            summarize_request_performance([r])

    def test_failed_cohort_rejected(self):
        r = row(0, 1, 0, 4, 1, 1)
        r["status"] = "FAILED"
        with self.assertRaises(ValueError):
            summarize_request_performance([r])

    def test_equal_resource_pair(self):
        self.assertTrue(validate_comparison_settings(plan()))

    def test_old_fixed_working_HBM_rejected(self):
        p = plan()
        p["HBM_accounting"] = "separate_fixed_working_quota"
        with self.assertRaises(ValueError):
            validate_comparison_settings(p)

    def test_wrong_exchange_rejected(self):
        p = plan()
        p["RANCHO_HBF_bytes"] = 65
        with self.assertRaises(ValueError):
            validate_comparison_settings(p)

    def test_missing_native_GPU_gate_rejected(self):
        p = plan()
        p["GPU_backend_certificate"] = {"status": "PASS_CPU_COMPONENT"}
        with self.assertRaises(ValueError):
            validate_comparison_settings(p)

    def test_missing_capacity_rejected(self):
        p = plan()
        p["baseline_HBM_bytes"] = None
        with self.assertRaises(ValueError):
            validate_comparison_settings(p)


if __name__ == "__main__":
    unittest.main()
