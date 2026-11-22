"""Track writes and set score cutoffs using recent offers"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

DAY = 86_400_000_000
MASK64 = (1 << 64) - 1

# Keep the keys used by saved controller snapshots
SNAPSHOT_KEYS = {
    "blocks_written": "W",
    "next_history_slot": "rp",
    "cutoff_ready": "have",
    "tie_probability": "tie",
    "current_spending_limit": "cap",
    "offer_history": "ring",
    "history_capacity": "limit",
    "cutoffs_by_cost_band": "bandcuts",
}


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
            release_us if release_us is not None else (7 * DAY if design == "hbf" else 5 * DAY)
        )
        self.reserve = reserve if reserve is not None else (0.0 if design == "hbf" else 0.3)
        if not (
            0 < self.release_us <= end_us
            and 0 <= self.reserve < 1
            and type(update) is int
            and update > 0
            and type(history) is int
            and history >= 100
        ):
            raise ValueError("Invalid controller geometry")
        self.update, self.history_capacity = update, history
        self.current_spending_limit = math.floor((1 - self.reserve) * budget)
        self.blocks_written = self.observations = self.next_history_slot = 0
        self.released = self.cutoff_ready = False
        self.threshold, self.tie_probability = -math.inf, 1.0
        self.offer_history = []
        self.last_id, self.last_us = -1, -(1 << 63)
        self.warm_provenance = None

    def release_reserved_budget_if_due(self, now_us):
        if not self.released and now_us >= self.release_us:
            self.current_spending_limit, self.released, self.cutoff_ready = (
                self.budget,
                True,
                False,
            )

    def offers_in_arrival_order(self):
        if len(self.offer_history) < self.history_capacity:
            return list(self.offer_history)
        return (
            self.offer_history[self.next_history_slot :]
            + self.offer_history[: self.next_history_slot]
        )

    def update_admission_cutoffs(self, now_us):
        """Estimate future writes and recalculate the score cutoff"""
        if self.design == "ssd":
            self.release_reserved_budget_if_due(now_us)
            values = self.offers_in_arrival_order()
            if len(values) < 100:
                self.threshold, self.tie_probability, self.cutoff_ready = (
                    -math.inf,
                    1.0,
                    False,
                )
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
                    1.0,
                    max(float(self.current_spending_limit - self.blocks_written), 0.0)
                    / max(rate * horizon, 1e-9),
                ),
            )
            scores = sorted(offer.score for offer in values)
            if fraction <= 0:
                self.threshold, self.tie_probability = math.inf, 0.0
            elif fraction >= 1:
                self.threshold, self.tie_probability = -math.inf, 1.0
            else:
                wanted = fraction * len(values)
                cutoff_index = min(len(values) - 1, math.floor(len(values) - wanted))
                self.threshold = scores[cutoff_index]
                above = sum(score > self.threshold for score in scores)
                equal = sum(score == self.threshold for score in scores)
                self.tie_probability = max(0.0, min(1.0, (wanted - above) / equal))
        else:
            if not self.offer_history:
                return
            # Native HBF sums the physical ring slots, not chronological order.
            values = [
                offer
                for offer in self.offer_history
                if offer.cost > 0 and math.isfinite(offer.score)
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
                ((self.end_us if self.released else self.release_us) - now_us) / 1e6,
                1.0,
            )
            fraction = max(
                0.0,
                min(
                    1.0,
                    max(float(self.current_spending_limit - self.blocks_written), 0.0)
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

    def decide_write_admission(
        self,
        event_id,
        now_us,
        prediction,
        *,
        distinct=1,
        cost=1,
        forced_warm=None,
        split_us=3 * DAY,
        payment_limit=None,
    ):
        """One eligible HBF request or positive-cost absent SSD eviction.

        Resident SSD refreshes, negative-time SSD evictions and ineligible HBF
        requests must not call this method. forced_warm replays an actual saved
        decision and is prohibited at/after split, never invented by this class.
        """
        if type(event_id) is not int or not self.last_id < event_id < (1 << 63):
            raise ValueError("Event IDs must strictly increase")
        if type(now_us) is not int or not self.last_us <= now_us < self.end_us or now_us < 0:
            raise ValueError("Invalid logical event clock")
        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError("Invalid causal prediction")
        if type(cost) is not int or type(distinct) is not int or cost < 0 or distinct < 0:
            raise ValueError("Invalid cost or distinct count")
        if self.design == "ssd" and (cost != 1 or distinct != 1):
            raise ValueError("SSD wallet sees only absent single-block offers")
        if self.design == "hbf" and distinct == 0 and cost != 0:
            raise ValueError("An empty suffix cannot cost writes")
        if forced_warm is not None and (type(forced_warm) is not bool or now_us >= split_us):
            raise ValueError("Saved warm decisions are warmup-only")
        self.last_id, self.last_us = event_id, now_us
        score = (
            prediction * float(distinct) / float(cost)
            if self.design == "hbf" and cost
            else (math.inf if self.design == "hbf" else prediction)
        )
        if not (self.design == "ssd" and forced_warm is not None):
            self.release_reserved_budget_if_due(now_us)
        if forced_warm is not None:
            admit = forced_warm
            if admit and (
                cost > self.budget - self.blocks_written
                or (
                    self.design == "hbf"
                    and cost > self.current_spending_limit - self.blocks_written
                )
            ):
                raise ValueError("Saved warm decision exceeds the original wallet")
        else:
            admit = distinct > 0 and (
                cost == 0
                or (
                    cost <= self.budget - self.blocks_written
                    and cost <= self.current_spending_limit - self.blocks_written
                    and (
                        not self.cutoff_ready
                        or score > self.threshold
                        or (
                            score == self.threshold
                            and seeded_tie_break_value(event_id, self.seed ^ 0x5BF03635)
                            < self.tie_probability
                        )
                    )
                )
            )
        if payment_limit is not None:
            if type(payment_limit) is not int or payment_limit < 0:
                raise ValueError("Invalid external physical payment limit")
            admit = admit and cost <= payment_limit
        observation = Observation(event_id, score, cost, now_us, distinct)
        if self.design == "hbf":
            self._record_offer(observation)
            if admit:
                self.blocks_written += cost
        else:
            if admit:
                self.blocks_written += 1
            self._record_offer(observation)
        return admit

    def refresh_offer_scores(self, event_ids, predictions, cut_us, *, initialization=False):
        """Re-score the exact retained past history without changing past actions."""
        values = self.offers_in_arrival_order()
        if list(event_ids) != [offer.event_id for offer in values] or len(predictions) != len(
            values
        ):
            raise ValueError("Repricing must cover the exact ordered history")
        if cut_us < self.last_us or any(offer.time_us >= cut_us for offer in values):
            raise ValueError("Repricing requires strictly past observations")
        if self.design == "hbf" and not initialization:
            raise ValueError("Selected HBF C7 does not reprice at later model updates")
        if any(not math.isfinite(prediction) or prediction < 0 for prediction in predictions):
            raise ValueError("Invalid reprice prediction")
        by_id = dict(zip(event_ids, predictions))
        self.offer_history = [
            Observation(
                offer.event_id,
                (
                    (
                        by_id[offer.event_id] * float(offer.distinct) / float(offer.cost)
                        if offer.cost
                        else math.inf
                    )
                    if self.design == "hbf"
                    else by_id[offer.event_id]
                ),
                offer.cost,
                offer.time_us,
                offer.distinct,
            )
            for offer in self.offer_history
        ]
        self.cutoff_ready, self.threshold, self.tie_probability = False, -math.inf, 1.0
        self.update_admission_cutoffs(cut_us)

    def snapshot_controller_state(self):
        """Full state including physical ring position; provenance is external."""
        state = {
            SNAPSHOT_KEYS.get(field, field): value
            for field, value in vars(self).items()
            if field != "offer_history"
        }
        state["ring"] = [asdict(v) for v in self.offer_history]

        def encode_infinite_values(value):
            if isinstance(value, float) and math.isinf(value):
                return "+inf" if value > 0 else "-inf"
            if isinstance(value, dict):
                return {key: encode_infinite_values(item) for key, item in value.items()}
            if isinstance(value, list):
                return [encode_infinite_values(item) for item in value]
            return value

        return encode_infinite_values(state)

    @classmethod
    def restore_controller_state(cls, state, expected_sha256):
        """Restore controller fields from a checked saved state."""
        raw = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Controller snapshot hash mismatch")

        def decode_infinite_values(value):
            if value == "+inf":
                return math.inf
            if value == "-inf":
                return -math.inf
            if isinstance(value, dict):
                return {key: decode_infinite_values(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode_infinite_values(item) for item in value]
            return value

        saved_fields = decode_infinite_values(state)
        controller = cls(
            saved_fields["design"],
            saved_fields["budget"],
            saved_fields["seed"],
            end_us=saved_fields["end_us"],
            release_us=saved_fields["release_us"],
            reserve=saved_fields["reserve"],
            update=saved_fields["update"],
            history=saved_fields["limit"],
        )
        expected_keys = {SNAPSHOT_KEYS.get(field, field) for field in vars(controller)}
        if set(saved_fields) != expected_keys:
            raise ValueError("Incomplete controller snapshot")
        fields_by_key = {key: field for field, key in SNAPSHOT_KEYS.items()}
        restored_fields = {
            fields_by_key.get(key, key): value for key, value in saved_fields.items()
        }
        controller.__dict__.update(restored_fields)
        controller.offer_history = [Observation(**offer) for offer in saved_fields["ring"]]
        history_count = len(controller.offer_history)
        if not (
            0 <= controller.blocks_written <= controller.budget
            and 0 <= history_count <= controller.history_capacity
            and controller.observations >= history_count
            and controller.next_history_slot
            == controller.observations % controller.history_capacity
            and history_count == min(controller.observations, controller.history_capacity)
        ):
            raise ValueError("Controller snapshot geometry mismatch")
        return controller
