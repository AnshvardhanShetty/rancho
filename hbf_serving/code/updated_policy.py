"""Decide whether to save completed KV to HBF

FilteredHistory counts earlier reuse that GPU memory alone would miss
UpdatedAdmission combines the prediction with that history
BandWallet adjusts the cutoff and keeps track of writes
"""

from collections import OrderedDict
import math, time
from controllers import Wallet, Observation, seeded_tie_break_value, DAY
from updated_model import SuffixGBT, CUT


class BandWallet(Wallet):
    """Keep write spending and choose cutoffs for groups of similar write costs."""

    def update_admission_cutoffs(self, now):
        """Estimate affordable demand, then update each populated cost group’s cutoff."""
        super().update_admission_cutoffs(now)
        # Estimate future write demand from recent positive-cost offers.
        values = [
            offer
            for offer in self.offer_history
            if offer.cost > 0 and math.isfinite(offer.score)
        ]
        if len(values) < 100:
            return
        history_seconds = max(
            1.0,
            (
                max(offer.time_us for offer in self.offer_history)
                - min(offer.time_us for offer in self.offer_history)
            )
            / 1e6,
        )
        demand = (
            len(values)
            / history_seconds
            * max(1.0, (self.end_us - now) / 1e6)
            * (sum(offer.cost for offer in values) / len(values))
        )
        fraction = min(
            1.0,
            max(
                0.0,
                (self.current_spending_limit - self.blocks_written) / max(1e-9, demand),
            ),
        )
        # Find score cutoffs within groups of similarly priced writes.
        groups = {}
        self.cutoffs_by_cost_band = {}
        for offer in values:
            groups.setdefault(offer.cost.bit_length() - 1, []).append(offer)
        for cost_band, group in groups.items():
            if len(group) < 100:
                continue
            if fraction in (0.0, 1.0):
                self.cutoffs_by_cost_band[cost_band] = (
                    (math.inf, 0.0) if fraction == 0 else (-math.inf, 1.0)
                )
                continue
            weights = {}
            for offer in group:
                weights[offer.score] = weights.get(offer.score, 0) + offer.cost
            wanted = fraction * sum(weights.values())
            above = 0
            for score in sorted(weights, reverse=True):
                equal = weights[score]
                if above + equal >= wanted:
                    self.cutoffs_by_cost_band[cost_band] = (
                        score,
                        max(0.0, min(1.0, (wanted - above) / equal)),
                    )
                    break
                above += equal

    def decide_suffix_admission(
        self, event, now, prediction, distinct, cost, past, split, payment_limit=None
    ):
        """Score one offer, check its cutoff and budget, then record the observation."""
        assert (
            self.last_id < event
            and self.last_us <= now < self.end_us
            and past >= 0
            and distinct >= 0
            and cost >= 0
        )
        assert distinct or not cost
        self.last_id, self.last_us = event, now
        self.release_reserved_budget_if_due(now)
        # Combine the prediction with reuse observed before this request.
        multiplier = 1 + past / distinct if distinct else 1.0
        score = (
            prediction * float(distinct) / float(cost) * multiplier
            if cost
            else math.inf
        )
        # Select the cutoff; coin controls ties at exactly that score.
        cutoff, coin = (
            getattr(self, "cutoffs_by_cost_band", {}).get(
                cost.bit_length() - 1, (self.threshold, self.tie_probability)
            )
            if cost and self.cutoff_ready
            else (self.threshold, self.tie_probability)
        )
        behind = bool(
            cost
            and self.cutoff_ready
            and self.blocks_written * (self.end_us - split)
            < self.budget * (now - split)
        )
        if behind:
            cutoff *= 0.5
        # Check the score and both remaining write limits.
        admit = bool(
            distinct
            and (
                not cost
                or cost <= self.budget - self.blocks_written
                and cost <= self.current_spending_limit - self.blocks_written
                and (
                    not self.cutoff_ready
                    or score > cutoff
                    or score == cutoff
                    and seeded_tie_break_value(event, self.seed ^ 0x5BF03635) < coin
                )
            )
        )
        before_guard = admit
        if payment_limit is not None:
            assert type(payment_limit) is int and payment_limit >= 0
            admit = admit and cost <= payment_limit
        details = dict(
            score=score if math.isfinite(score) else str(score),
            cutoff=cutoff if math.isfinite(cutoff) else str(cutoff),
            tie=float(coin),
            have_threshold=bool(self.cutoff_ready),
            half_behind=behind,
            history_multiplier=multiplier,
            past_filtered_reuses=past,
            score_admit_before_physical_guard=before_guard,
            physical_guard_rejected=before_guard and not admit,
        )
        # Record every offer, including declines; charge accepted writes only.
        self._record_offer(Observation(event, score, cost, now, distinct))
        if admit:
            self.blocks_written += cost
        return admit, details


class FilteredHistory:
    """Keep past reuse counts and an independent HBM-only comparison cache."""

    def __init__(self, capacity):
        """Start with empty comparison-cache contents and reuse counts."""
        self.capacity = capacity
        self.cache = OrderedDict()
        self.stamps = {}
        self.counts = {}
        self.processed = set()

    def collect_request_reuse(self, request_id, keys, session, suffix_begin):
        """Read earlier reuse and stage new observations for after this decision."""
        assert request_id not in self.processed and 0 <= suffix_begin <= len(keys)
        readable = 0
        for key in keys:
            if key not in self.cache:
                break
            self.cache.move_to_end(key)
            readable += 1
        suffix = list(dict.fromkeys(keys[suffix_begin:]))
        past_filtered_reuses = sum(
            self.counts.get(block_hash, 0) for block_hash in suffix
        )
        pending = []
        seen = set()
        for position, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)
            if position >= readable and self.stamps.get(key) == session:
                pending.append(key)
        return dict(
            request_id=request_id,
            keys=keys,
            session=session,
            suffix_begin=suffix_begin,
            shadow_cached_pages=readable,
            past_filtered_reuses=past_filtered_reuses,
            pending=pending,
        )

    def record_request_reuse(self, reuse_history):
        """Apply the staged observations after the write decision."""
        for key in reuse_history["keys"][reuse_history["shadow_cached_pages"] :]:
            if self.capacity:
                self.cache[key] = True
                self.cache.move_to_end(key)
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
        for key in reuse_history["pending"]:
            self.counts[key] = self.counts.get(key, 0) + 1
        for key in reuse_history["keys"]:
            self.stamps[key] = reuse_history["session"]
        self.processed.add(reuse_history["request_id"])


class UpdatedAdmission:
    """Connect the saved predictor to reuse history and the write controller."""

    def __init__(self, config):
        """Check the supplied cold-start record and load the predictor."""
        self.config = config
        from pathlib import Path
        import hashlib, json

        controller_start_bytes = Path(config["controller_history"]).read_bytes()
        assert (
            hashlib.sha256(controller_start_bytes).hexdigest()
            == config["controller_history_sha256"]
        )
        receipt = json.loads(controller_start_bytes)
        assert (
            receipt["cold"]
            and receipt["observations"] == receipt["paid_pages"] == 0
            and receipt["fit"] == config["fit"]
        )
        self.model = SuffixGBT(config["fit"])
        assert self.model.model_sha256 == receipt["model_sha256"]
        self.wallet = BandWallet(
            "hbf",
            config["HBF_pages"],
            config["repetition"],
            end_us=CUT + DAY,
            release_us=CUT + DAY,
            reserve=0.0,
        )
        self.completed = set()
        self.past = 0

    def set_suffix_reuse_count(self, past):
        """Supply the earlier reuse count for the next offered suffix."""
        assert type(past) is int and past >= 0
        self.past = past

    def decide_hbf_write_after_prefill(
        self,
        row,
        now,
        features,
        distinct,
        cost,
        *,
        computed_prompt_tokens,
        prompt_tokens,
        gpu_completed,
        payment_limit=None
    ):
        """Predict reuse after GPU computation and return a paid or declined offer."""
        assert (
            gpu_completed is True
            and computed_prompt_tokens == prompt_tokens
            and prompt_tokens > 0
            and row not in self.completed
        )
        start = time.perf_counter_ns()
        prediction = float(self.model.predict_suffix_reuse([features], now)[0])
        predictor_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        admit, detail = self.wallet.decide_suffix_admission(
            row, now, prediction, distinct, cost, self.past, CUT, payment_limit
        )
        controller_ns = time.perf_counter_ns() - start
        self.completed.add(row)
        return dict(
            admit=admit,
            prediction=prediction,
            paid_cost=cost if admit else 0,
            total_paid_writes=self.wallet.blocks_written,
            fit_cut_us=CUT,
            model_sha256=self.model.model_sha256,
            predictor_cpu_ns=predictor_ns,
            controller_cpu_ns=controller_ns,
            **detail
        )
