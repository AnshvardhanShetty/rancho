"""Physical-time write accounting for a fresh independent emulated device-day.

Assumptions: 1 DWPD is one user-capacity of host writes per real 24 hours.
The actual startup fills may burst within that daily total, but all later
writes additionally need credits earned since measured serving began. There is
zero initial steady credit; unused credit carries forward within this run.
We never refund paid writes or sleep to manufacture an endurance denominator.
Startup and steady writes are separately reported. A short run does not prove
long-term flash behavior, and this module does not simulate write amplification.
"""

DAY_NS = 86_400_000_000_000


class PhysicalBudget:
    """Track startup and measured writes against one device-day allowance."""

    def __init__(self, capacity_pages, origin_ns):
        if (
            type(capacity_pages) is not int
            or capacity_pages <= 0
            or type(origin_ns) is not int
            or origin_ns < 0
        ):
            raise ValueError("Explicit positive physical capacity and clock required")
        self.capacity = capacity_pages
        self.origin = origin_ns
        self.begin = None
        self.total = self.startup = self.measured = 0
        self.last = origin_ns

    def limit(self, now_ns, warmup, measurement_begin_ns=None):
        """Return how many more pages may be written at this time."""
        if (
            type(now_ns) is not int
            or now_ns < self.last
            or now_ns - self.origin >= DAY_NS
        ):
            raise ValueError(
                "Physical clock regressed or independent device-day expired"
            )
        self.last = now_ns
        remaining = self.capacity - self.total
        if warmup:
            if self.begin is not None:
                raise ValueError("Startup after measured phase began")
            return remaining
        if (
            measurement_begin_ns is None
            or not self.origin <= measurement_begin_ns <= now_ns
        ):
            raise ValueError("Actual measurement-start timestamp required")
        if self.begin is None:
            self.begin = measurement_begin_ns
        if self.begin != measurement_begin_ns:
            raise ValueError("Measurement origin changed")
        earned = self.capacity * (now_ns - self.begin) // DAY_NS
        return max(0, min(remaining, earned - self.measured))

    def charge(self, cost, warmup, allowance):
        """Charge an accepted write without exceeding its available allowance."""
        if (
            type(cost) is not int
            or cost < 0
            or cost > allowance
            or cost > self.capacity - self.total
        ):
            raise ValueError("Payment exceeds physical write allowance")
        self.total += cost
        if warmup:
            self.startup += cost
        else:
            self.measured += cost

    def summary(self, end_ns, measurement_begin_ns, window_and_drain_seconds):
        """Report startup and measured spending separately."""
        if (
            not self.origin <= measurement_begin_ns <= end_ns
            or end_ns - self.origin >= DAY_NS
        ):
            raise ValueError("Invalid final physical timing")
        elapsed = (end_ns - self.origin) / 1e9
        measured_seconds = window_and_drain_seconds
        if measured_seconds <= 0 or elapsed <= 0:
            raise ValueError("Positive real duration required")
        return dict(
            clock="real monotonic elapsed time",
            DWPD_limit=1,
            fresh_independent_device_day=True,
            prior_writes_assumed_pages=0,
            daily_capacity_pages=self.capacity,
            daily_paid_pages=self.total,
            startup_paid_pages=self.startup,
            measured_paid_pages=self.measured,
            daily_allowance_respected=self.total <= self.capacity,
            physical_accounting_seconds=elapsed,
            measured_window_and_drain_seconds=measured_seconds,
            measured_write_rate_DWPD=self.measured
            / self.capacity
            * 86400
            / measured_seconds,
            startup_inclusive_write_rate_DWPD=self.total
            / self.capacity
            * 86400
            / elapsed,
            startup_amortization_seconds_at_1_DWPD=self.startup / self.capacity * 86400,
            startup_rate_is_not_a_sustainable_rate=True,
            steady_initial_credit_pages=0,
            steady_allowance_earned_pages=self.capacity
            * (end_ns - measurement_begin_ns)
            // DAY_NS,
            steady_rate_respected=self.measured
            <= self.capacity * (end_ns - measurement_begin_ns) // DAY_NS,
        )
