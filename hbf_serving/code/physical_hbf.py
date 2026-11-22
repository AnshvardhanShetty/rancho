"""Check that a write fits both the full run budget and the credits earned so far"""

import json
import re
import time
from pathlib import Path
from direct_backend import log_event
from direct_backend_hbf import HBFConnector as NativeHBFConnector
from write_bandwidth import PhysicalBudget


class PhysicalTimedPolicy:
    """Limit writes using credits earned from real elapsed time."""

    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
        self.origin = time.monotonic_ns()
        self.ledger = PhysicalBudget(config["HBF_pages"], self.origin)
        self.context = None
        self.begin = None
        log_event(
            config,
            "worker.jsonl",
            dict(
                event="PHYSICAL_CONTROLLER_CLOCK_START",
                origin_ns=self.origin,
                daily_capacity_pages=config["HBF_pages"],
                steady_initial_credit_pages=0,
            ),
        )

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def set_request_context(self, request_id):
        """Identify whether the next offer belongs to warm-up or measurement."""
        if self.context is not None:
            raise ValueError("Unconsumed physical request context")
        match = re.fullmatch(r"row(\d+)(?:-[0-9a-f]{8})?", request_id)
        if not match:
            raise ValueError("Unrecognized physical request identity")
        self.context = int(match[1])

    def decide_hbf_write_after_prefill(self, row, logical_us, features, distinct, cost, **kwargs):
        """Apply the elapsed-time write limit before charging an accepted offer."""
        if self.context is None:
            raise ValueError("Physical phase must be bound to actual request")
        warmup = self.context < self.config["shared_groups"]
        self.context = None
        if not warmup and self.begin is None:
            receipt = json.loads(
                (Path(self.config["output"]) / "MEASUREMENT_CLOCK.json").read_text()
            )
            self.begin = receipt["measurement_start_ns"]
        total_start = time.perf_counter_ns()
        now_ns = time.monotonic_ns()
        allowance = self.ledger.available_write_pages(now_ns, warmup, self.begin)
        physical_us = self.config["fit"]["cut_us"] + (now_ns - self.origin) // 1000
        start = time.perf_counter_ns()
        result = self.policy.decide_hbf_write_after_prefill(
            row, physical_us, features, distinct, cost, payment_limit=allowance, **kwargs
        )
        cpu_ns = time.perf_counter_ns() - start
        self.ledger.charge_write_pages(result["paid_cost"], warmup, allowance)
        assert self.policy.wallet.blocks_written == self.ledger.total
        total_cpu_ns = time.perf_counter_ns() - total_start
        result.update(
            decision_cpu_ns=cpu_ns,
            total_admission_cpu_ns=total_cpu_ns,
            controller_time_us=physical_us,
            physical_decision_ns=now_ns,
            physical_origin_ns=self.origin,
            measurement_start_ns=self.begin,
            physical_write_limit_pages=allowance,
            physical_warmup=warmup,
        )
        return result


class HBFConnector(NativeHBFConnector):
    """Wrap the prefill policy with the elapsed time budget"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.policy is not None:
            if self.config.get("physical_write_accounting") is not True:
                raise ValueError("Physical-time benchmark requires physical accounting")
            self.policy = PhysicalTimedPolicy(self.policy, self.config)
