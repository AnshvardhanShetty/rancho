"""Track writes and set score cutoffs using recent offers"""

from dataclasses import dataclass
import math

DAY = 86_400_000_000
MASK64 = (1 << 64) - 1


def seeded_tie_break_value(event_id, seed):
    hashed_value = ((event_id * 0x9E3779B97F4A7C15) & MASK64) ^ seed
    hashed_value ^= hashed_value >> 33
    hashed_value = (hashed_value * 0xFF51AFD7ED558CCD) & MASK64
    hashed_value ^= hashed_value >> 33
    hashed_value = (hashed_value * 0xC4CEB9FE1A85EC53) & MASK64
    hashed_value ^= hashed_value >> 33
    return (hashed_value >> 11) / 9007199254740992.0


@dataclass(frozen=True)
class Observation:
    """The score and cost saved for one offer"""

    event_id: int
    score: float
    cost: int
    time_us: int
    distinct: int


class WriteController:
    """Track the HBF write budget and recent offers."""

    def __init__(self, budget, seed, *, end_us, update=1000, history=20_000):
        if type(budget) is not int or not 0 <= budget < (1 << 63):
            raise ValueError("Invalid integer block-write budget")
        if type(seed) is not int or not 0 <= seed <= MASK64:
            raise ValueError("Invalid uint64 seed")
        if not (
            end_us > 0
            and type(update) is int
            and update > 0
            and type(history) is int
            and history >= 100
        ):
            raise ValueError("Invalid controller settings")
        self.budget, self.seed, self.end_us = budget, seed, end_us
        self.update, self.history_capacity = update, history
        self.blocks_written = self.observations = self.next_history_slot = 0
        self.cutoff_ready = False
        self.threshold, self.tie_probability = -math.inf, 1.0
        self.offer_history = []
        self.last_id, self.last_us = -1, -(1 << 63)

    def update_admission_cutoffs(self, now_us):
        """Estimate future writes and recalculate the score cutoff"""
        if not self.offer_history:
            return
        # Native HBF sums the physical ring slots, not chronological order.
        values = [
            offer for offer in self.offer_history if offer.cost > 0 and math.isfinite(offer.score)
        ]
        if len(values) < 100:
            return
        total = sum(float(offer.cost) for offer in values)
        span = max(
            (
                max(offer.time_us for offer in self.offer_history)
                - min(offer.time_us for offer in self.offer_history)
            )
            / 1e6,
            1.0,
        )
        rate, mean = len(values) / span, total / len(values)
        horizon = max(
            (self.end_us - now_us) / 1e6,
            1.0,
        )
        fraction = max(
            0.0,
            min(
                1.0,
                max(float(self.budget - self.blocks_written), 0.0)
                / max(rate * horizon * mean, 1e-9),
            ),
        )
        if fraction <= 0:
            self.threshold, self.tie_probability = math.inf, 0.0
        elif fraction >= 1:
            self.threshold, self.tie_probability = -math.inf, 1.0
        else:
            values = sorted(values, key=lambda offer: -offer.score)
            wanted, above = fraction * total, 0.0
            self.threshold = values[-1].score
            score_start = 0
            while score_start < len(values):
                score_end, equal = score_start, 0.0
                while (
                    score_end < len(values)
                    and values[score_end].score == values[score_start].score
                ):
                    equal += values[score_end].cost
                    score_end += 1
                if above + equal >= wanted:
                    self.threshold = values[score_start].score
                    break
                above += equal
                score_start = score_end
            self.tie_probability = max(
                0.0, min(1.0, (wanted - above) / equal if equal > 0 else 0.0)
            )
        self.cutoff_ready = True

    def _record_offer(self, offer):
        """Save this offer and record what the policy decided"""
        if len(self.offer_history) < self.history_capacity:
            self.offer_history.append(offer)
        else:
            self.offer_history[self.next_history_slot] = offer
        self.next_history_slot = (self.next_history_slot + 1) % self.history_capacity
        self.observations += 1
        if self.observations % self.update == 0:
            self.update_admission_cutoffs(offer.time_us)


# Limit actual writes using time since the demo began.
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

    def available_write_pages(self, now_ns, warmup, measurement_begin_ns=None):
        """Return how many more pages may be written at this time."""
        if type(now_ns) is not int or now_ns < self.last or now_ns - self.origin >= DAY_NS:
            raise ValueError("Physical clock regressed or independent device-day expired")
        self.last = now_ns
        remaining = self.capacity - self.total
        if warmup:
            if self.begin is not None:
                raise ValueError("Startup after measured phase began")
            return remaining
        if measurement_begin_ns is None or not self.origin <= measurement_begin_ns <= now_ns:
            raise ValueError("Actual measurement-start timestamp required")
        if self.begin is None:
            self.begin = measurement_begin_ns
        if self.begin != measurement_begin_ns:
            raise ValueError("Measurement origin changed")
        earned = self.capacity * (now_ns - self.begin) // DAY_NS
        return max(0, min(remaining, earned - self.measured))

    def charge_write_pages(self, cost, warmup, allowance):
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

    def summarize_write_spending(self, end_ns, measurement_begin_ns, window_and_drain_seconds):
        """Report startup and measured spending separately."""
        if not self.origin <= measurement_begin_ns <= end_ns or end_ns - self.origin >= DAY_NS:
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
            measured_write_rate_DWPD=self.measured / self.capacity * 86400 / measured_seconds,
            startup_inclusive_write_rate_DWPD=self.total / self.capacity * 86400 / elapsed,
            startup_amortization_seconds_at_1_DWPD=self.startup / self.capacity * 86400,
            startup_rate_is_not_a_sustainable_rate=True,
            steady_initial_credit_pages=0,
            steady_allowance_earned_pages=self.capacity
            * (end_ns - measurement_begin_ns)
            // DAY_NS,
            steady_rate_respected=self.measured
            <= self.capacity * (end_ns - measurement_begin_ns) // DAY_NS,
        )
