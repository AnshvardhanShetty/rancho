"""Keep a separate controller to check the decisions made by the serving code"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

DAY = 86_400_000_000
MASK64 = (1 << 64) - 1


def hash01(event_id, seed):
    x = ((event_id * 0x9E3779B97F4A7C15) & MASK64) ^ seed
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MASK64
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MASK64
    x ^= x >> 33
    return (x >> 11) / 9007199254740992.0


@dataclass(frozen=True)
class Observation:
    """One offer kept by the reference controller"""

    event_id: int
    score: float
    cost: int
    time_us: int
    distinct: int


class Wallet:
    """Exact selected-controller rules; cache physics remain the caller's job."""

    def __init__(
        self,
        design,
        budget,
        seed,
        *,
        end_us=7 * DAY,
        release_us=None,
        reserve=None,
        update=1000,
        history=20000,
    ):
        if design not in ("hbf", "ssd"):
            raise ValueError("Unknown design")
        if type(budget) is not int or not 0 <= budget < (1 << 63):
            raise ValueError("Invalid integer block-write budget")
        if type(seed) is not int or not 0 <= seed <= MASK64:
            raise ValueError("Invalid uint64 seed")
        self.design, self.budget, self.seed = design, budget, seed
        self.end_us = end_us
        self.release_us = (
            release_us
            if release_us is not None
            else (7 * DAY if design == "hbf" else 5 * DAY)
        )
        self.reserve = (
            reserve if reserve is not None else (0.0 if design == "hbf" else 0.3)
        )
        if not (
            0 < self.release_us <= end_us
            and 0 <= self.reserve < 1
            and type(update) is int
            and update > 0
            and type(history) is int
            and history >= 100
        ):
            raise ValueError("Invalid controller geometry")
        self.update, self.limit = update, history
        self.cap = math.floor((1 - self.reserve) * budget)
        self.W = self.observations = self.rp = 0
        self.released = self.have = False
        self.threshold, self.tie = -math.inf, 1.0
        self.ring = []
        self.last_id, self.last_us = -1, -(1 << 63)
        self.warm_provenance = None

    def phase(self, now_us):
        if not self.released and now_us >= self.release_us:
            self.cap, self.released, self.have = self.budget, True, False

    def chronological(self):
        if len(self.ring) < self.limit:
            return list(self.ring)
        return self.ring[self.rp :] + self.ring[: self.rp]

    def recompute(self, now_us):
        """Calculate the reference cutoff from recent offers"""
        if self.design == "ssd":
            self.phase(now_us)
            values = self.chronological()
            if len(values) < 100:
                self.threshold, self.tie, self.have = -math.inf, 1.0, False
                return
            span = max((values[-1].time_us - values[0].time_us) / 1e6, 1.0)
            horizon = max(
                ((self.end_us if self.released else self.release_us) - now_us) / 1e6,
                1.0,
            )
            rate = len(values) / span
            fraction = max(
                0.0,
                min(
                    1.0, max(float(self.cap - self.W), 0.0) / max(rate * horizon, 1e-9)
                ),
            )
            scores = sorted(v.score for v in values)
            if fraction <= 0:
                self.threshold, self.tie = math.inf, 0.0
            elif fraction >= 1:
                self.threshold, self.tie = -math.inf, 1.0
            else:
                wanted = fraction * len(values)
                k = min(len(values) - 1, math.floor(len(values) - wanted))
                self.threshold = scores[k]
                above = sum(s > self.threshold for s in scores)
                equal = sum(s == self.threshold for s in scores)
                self.tie = max(0.0, min(1.0, (wanted - above) / equal))
        else:
            if not self.ring:
                return
            # Native HBF sums the physical ring slots, not chronological order.
            values = [v for v in self.ring if v.cost > 0 and math.isfinite(v.score)]
            if len(values) < 100:
                return
            total = sum(float(v.cost) for v in values)
            span = max(
                (max(v.time_us for v in self.ring) - min(v.time_us for v in self.ring))
                / 1e6,
                1.0,
            )
            rate, mean = len(values) / span, total / len(values)
            horizon = max(
                ((self.end_us if self.released else self.release_us) - now_us) / 1e6,
                1.0,
            )
            fraction = max(
                0.0,
                min(
                    1.0,
                    max(float(self.cap - self.W), 0.0)
                    / max(rate * horizon * mean, 1e-9),
                ),
            )
            if fraction <= 0:
                self.threshold, self.tie = math.inf, 0.0
            elif fraction >= 1:
                self.threshold, self.tie = -math.inf, 1.0
            else:
                values = sorted(values, key=lambda v: -v.score)
                wanted, above = fraction * total, 0.0
                self.threshold = values[-1].score
                i = 0
                while i < len(values):
                    j, equal = i, 0.0
                    while j < len(values) and values[j].score == values[i].score:
                        equal += values[j].cost
                        j += 1
                    if above + equal >= wanted:
                        self.threshold = values[i].score
                        break
                    above += equal
                    i = j
                self.tie = max(
                    0.0, min(1.0, (wanted - above) / equal if equal > 0 else 0.0)
                )
        self.have = True

    def _observe(self, v):
        if len(self.ring) < self.limit:
            self.ring.append(v)
        else:
            self.ring[self.rp] = v
        self.rp = (self.rp + 1) % self.limit
        self.observations += 1
        if self.observations % self.update == 0:
            self.recompute(v.time_us)

    def decide(
        self,
        event_id,
        now_us,
        prediction,
        *,
        distinct=1,
        cost=1,
        forced_warm=None,
        split_us=3 * DAY,
    ):
        """One eligible HBF request or positive-cost absent SSD eviction.

        Resident SSD refreshes, negative-time SSD evictions and ineligible HBF
        requests must not call this method. forced_warm replays an actual saved
        decision and is prohibited at/after split, never invented by this class.
        """
        if type(event_id) is not int or not self.last_id < event_id < (1 << 63):
            raise ValueError("Event IDs must strictly increase")
        if (
            type(now_us) is not int
            or not self.last_us <= now_us < self.end_us
            or now_us < 0
        ):
            raise ValueError("Invalid logical event clock")
        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError("Invalid causal prediction")
        if (
            type(cost) is not int
            or type(distinct) is not int
            or cost < 0
            or distinct < 0
        ):
            raise ValueError("Invalid cost or distinct count")
        if self.design == "ssd" and (cost != 1 or distinct != 1):
            raise ValueError("SSD wallet sees only absent single-block offers")
        if self.design == "hbf" and distinct == 0 and cost != 0:
            raise ValueError("An empty suffix cannot cost writes")
        if forced_warm is not None and (
            type(forced_warm) is not bool or now_us >= split_us
        ):
            raise ValueError("Saved warm decisions are warmup-only")
        self.last_id, self.last_us = event_id, now_us
        score = (
            prediction * float(distinct) / float(cost)
            if self.design == "hbf" and cost
            else (math.inf if self.design == "hbf" else prediction)
        )
        if not (self.design == "ssd" and forced_warm is not None):
            self.phase(now_us)
        if forced_warm is not None:
            admit = forced_warm
            if admit and (
                cost > self.budget - self.W
                or (self.design == "hbf" and cost > self.cap - self.W)
            ):
                raise ValueError("Saved warm decision exceeds the original wallet")
        else:
            admit = distinct > 0 and (
                cost == 0
                or (
                    cost <= self.budget - self.W
                    and cost <= self.cap - self.W
                    and (
                        not self.have
                        or score > self.threshold
                        or (
                            score == self.threshold
                            and hash01(event_id, self.seed ^ 0x5BF03635) < self.tie
                        )
                    )
                )
            )
        observation = Observation(event_id, score, cost, now_us, distinct)
        if self.design == "hbf":
            self._observe(observation)
            if admit:
                self.W += cost
        else:
            if admit:
                self.W += 1
            self._observe(observation)
        return admit

    def reprice(self, event_ids, predictions, cut_us, *, initialization=False):
        """Re-score the exact retained past history without changing past actions."""
        values = self.chronological()
        if list(event_ids) != [v.event_id for v in values] or len(predictions) != len(
            values
        ):
            raise ValueError("Repricing must cover the exact ordered history")
        if cut_us < self.last_us or any(v.time_us >= cut_us for v in values):
            raise ValueError("Repricing requires strictly past observations")
        if self.design == "hbf" and not initialization:
            raise ValueError("Selected HBF C7 does not reprice at later model updates")
        if any(not math.isfinite(p) or p < 0 for p in predictions):
            raise ValueError("Invalid reprice prediction")
        by_id = dict(zip(event_ids, predictions))
        self.ring = [
            Observation(
                v.event_id,
                (
                    (
                        by_id[v.event_id] * float(v.distinct) / float(v.cost)
                        if v.cost
                        else math.inf
                    )
                    if self.design == "hbf"
                    else by_id[v.event_id]
                ),
                v.cost,
                v.time_us,
                v.distinct,
            )
            for v in self.ring
        ]
        self.have, self.threshold, self.tie = False, -math.inf, 1.0
        self.recompute(cut_us)

    def snapshot(self):
        """Full state including physical ring position; provenance is external."""
        state = {k: v for k, v in vars(self).items() if k != "ring"}
        state["ring"] = [asdict(v) for v in self.ring]

        def encode(x):
            if isinstance(x, float) and math.isinf(x):
                return "+inf" if x > 0 else "-inf"
            if isinstance(x, dict):
                return {k: encode(v) for k, v in x.items()}
            if isinstance(x, list):
                return [encode(v) for v in x]
            return x

        return encode(state)

    @classmethod
    def restore(cls, state, expected_sha256):
        """Restore controller fields from a checked saved state."""
        raw = json.dumps(
            state, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Controller snapshot hash mismatch")

        def decode(x):
            if x == "+inf":
                return math.inf
            if x == "-inf":
                return -math.inf
            if isinstance(x, dict):
                return {k: decode(v) for k, v in x.items()}
            if isinstance(x, list):
                return [decode(v) for v in x]
            return x

        s = decode(state)
        w = cls(
            s["design"],
            s["budget"],
            s["seed"],
            end_us=s["end_us"],
            release_us=s["release_us"],
            reserve=s["reserve"],
            update=s["update"],
            history=s["limit"],
        )
        if set(s) != set(vars(w)):
            raise ValueError("Incomplete controller snapshot")
        w.__dict__.update(s)
        w.ring = [Observation(**v) for v in s["ring"]]
        if not (
            0 <= w.W <= w.budget
            and 0 <= len(w.ring) <= w.limit
            and w.observations >= len(w.ring)
            and w.rp == w.observations % w.limit
            and len(w.ring) == min(w.observations, w.limit)
        ):
            raise ValueError("Controller snapshot geometry mismatch")
        return w
