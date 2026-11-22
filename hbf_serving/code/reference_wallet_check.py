"""Check cost-group score thresholds using a separate controller implementation."""

import array, math, struct
from pathlib import Path
from reference_write_controller import Wallet, Observation, seeded_tie_break_value

DECISION_RECORD_LAYOUT = struct.Struct("<qqqqqddddqqq")
REUSE_HISTORY_RECORD_LAYOUT = struct.Struct("<qq")
OFFER_HISTORY_RECORD_LAYOUT = struct.Struct("<qqqq")


def scores_match(a, b):
    return a == b or (
        math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    )


def compute_effective_cutoff(threshold, have, spent, budget, now, split, end, enabled):
    """Use exact integer time/spending comparison, with no forced admission."""
    return (
        threshold / 2
        if enabled and have and spent * (end - split) < budget * (now - split)
        else threshold
    )


class CostBandWriteController(Wallet):
    """Compute rank boundaries independently, then compare every live decision."""

    def update_admission_cutoffs(self, now):
        """Rebuild the score cutoff for each group of similar write sizes"""
        super().update_admission_cutoffs(now)
        priced = [
            offer for offer in self.offer_history if offer.cost > 0 and math.isfinite(offer.score)
        ]
        if len(priced) < 100:
            return
        span = max(
            1.0,
            (
                max(offer.time_us for offer in self.offer_history)
                - min(offer.time_us for offer in self.offer_history)
            )
            / 1e6,
        )
        horizon = max(1.0, (self.end_us - now) / 1e6)
        total = sum(offer.cost for offer in priced)
        demand = (len(priced) / span) * horizon * (total / len(priced))
        affordable_fraction = min(
            1.0,
            max(
                0.0,
                float(self.current_spending_limit - self.blocks_written) / max(1e-9, demand),
            ),
        )
        groups = {}
        for offer in priced:
            groups.setdefault(offer.cost.bit_length() - 1, []).append(offer)
        self.cutoffs_by_cost_band = {}
        for cost_band, group in groups.items():
            if len(group) < 100:
                continue
            if affordable_fraction in (0.0, 1.0):
                self.cutoffs_by_cost_band[cost_band] = (
                    (math.inf, 0.0) if affordable_fraction == 0 else (-math.inf, 1.0)
                )
                continue
            write_cost_by_score = {}
            for offer in group:
                write_cost_by_score[offer.score] = (
                    write_cost_by_score.get(offer.score, 0) + offer.cost
                )
            wanted = affordable_fraction * sum(write_cost_by_score.values())
            above = 0
            for rank in sorted(write_cost_by_score, reverse=True):
                weight = write_cost_by_score[rank]
                if above + weight >= wanted:
                    self.cutoffs_by_cost_band[cost_band] = (
                        rank,
                        max(0.0, min(1.0, (wanted - above) / weight)),
                    )
                    break
                above += weight


def audit_admission_decisions(
    native,
    density_path,
    seed,
    budget,
    split,
    end,
    bonus,
    values,
    banded=False,
    half_behind=False,
    mode="control",
):
    """Replay the recorded offers and compare each cutoff and decision"""
    native = Path(native)
    density = array.array("d")
    density.frombytes(Path(density_path).read_bytes())
    reference_controller = (CostBandWriteController if banded else Wallet)(
        "hbf", budget, seed, end_us=end, release_us=end, reserve=0
    )
    reasons = [0] * 8
    count = 0
    startup = first1000 = 0
    previous_request_row = -1
    previous_arrival_us = split
    boost_sum = 0
    boost_max = 1
    boosted = 0
    discounted_offers = 0
    soft = mode in ("soft", "soft-rescue")
    partition = mode in ("reserve", "rescue", "soft-rescue")
    lottery = mode in ("rescue", "soft-rescue")
    rejected_volume = regular_spend = random_spend = 0
    counts = dict(
        ordinary_score_rejections=0,
        coin_trials=0,
        coin_paid=0,
        reserve_guard=0,
        regular_blocks=0,
        rescue_blocks=0,
        changed_score=0,
    )
    by_day = [dict(trials=0, paid=0, blocks=0, blocked_regular=0) for _ in range(4)]
    assert (native / "warm_history.bin").stat().st_size == 0
    with (native / "decisions.bin").open("rb") as decision_file, open(
        values, "rb"
    ) as reuse_history_file:
        while chunk := decision_file.read(DECISION_RECORD_LAYOUT.size):
            assert len(chunk) == DECISION_RECORD_LAYOUT.size
            (
                row,
                time,
                distinct,
                cost,
                before,
                predicted_reuse,
                score,
                threshold,
                tie,
                have,
                reason,
                admitted,
            ) = DECISION_RECORD_LAYOUT.unpack(chunk)
            past, history_distinct_blocks = REUSE_HISTORY_RECORD_LAYOUT.unpack(
                reuse_history_file.read(REUSE_HISTORY_RECORD_LAYOUT.size)
            )
            assert history_distinct_blocks == distinct and past >= 0
            assert previous_request_row < row and previous_arrival_us <= time < end
            previous_request_row, previous_arrival_us = row, time
            reference_controller.release_reserved_budget_if_due(time)
            assert predicted_reuse == density[row]
            expected = density[row] * float(distinct) / float(cost) if cost else math.inf
            boost = 1 + float(past) / float(distinct) if bonus and distinct else 1
            if soft and cost and boost > 1:
                counts["changed_score"] += 1
                boost = math.sqrt(boost)
            if cost and distinct:
                expected *= boost
            assert scores_match(score, expected), (row, "score", score, expected)
            cutoff, coin = (
                getattr(reference_controller, "cutoffs_by_cost_band", {}).get(
                    cost.bit_length() - 1,
                    (
                        reference_controller.threshold,
                        reference_controller.tie_probability,
                    ),
                )
                if banded and reference_controller.cutoff_ready and cost
                else (
                    reference_controller.threshold,
                    reference_controller.tie_probability,
                )
            )
            if (
                cost
                and reference_controller.cutoff_ready
                and half_behind
                and reference_controller.blocks_written * (end - split) < budget * (time - split)
            ):
                discounted_offers += 1
                cutoff = compute_effective_cutoff(
                    cutoff,
                    reference_controller.cutoff_ready,
                    reference_controller.blocks_written,
                    budget,
                    time,
                    split,
                    end,
                    True,
                )
            assert (
                before == reference_controller.blocks_written
                and bool(have) == reference_controller.cutoff_ready
                and scores_match(threshold, cutoff)
                and scores_match(tie, coin)
            ), (row, "controller")
            why = (
                1
                if not distinct
                else (
                    2
                    if not cost
                    else (
                        3
                        if cost > budget - reference_controller.blocks_written
                        else (
                            4
                            if cost
                            > reference_controller.current_spending_limit
                            - reference_controller.blocks_written
                            else (
                                6
                                if not reference_controller.cutoff_ready
                                else (
                                    7
                                    if expected > cutoff
                                    or (
                                        expected == cutoff
                                        and seeded_tie_break_value(row, seed ^ 0x5BF03635) < coin
                                    )
                                    else 5
                                )
                            )
                        )
                    )
                )
            )
            day = by_day[(time - split) // 86400000000]
            base_reason = why
            reserve = budget // 8
            if partition and why in (6, 7) and regular_spend + cost > budget - reserve:
                why = 4
                counts["reserve_guard"] += 1
                day["blocked_regular"] += 1
            if base_reason == 5:
                counts["ordinary_score_rejections"] += 1
                if lottery and rejected_volume and cost <= reserve - random_spend:
                    projected = (
                        rejected_volume * ((end - time) / (time - split))
                        if time > split
                        else math.inf
                    )
                    probability = min(1.0, (reserve - random_spend) / projected)
                    counts["coin_trials"] += 1
                    day["trials"] += 1
                    if seeded_tie_break_value(row, seed ^ 0x72A11D) < probability:
                        why = 7
                        random_spend += cost
                        counts["coin_paid"] += 1
                        day["paid"] += 1
                        day["blocks"] += cost
                rejected_volume += cost
            elif why in (6, 7):
                regular_spend += cost
            accept = why in (2, 6, 7)
            assert reason == why and bool(admitted) == accept, (
                row,
                "decision",
                reason,
                why,
            )
            if partition:
                assert 0 <= regular_spend <= budget - reserve and 0 <= random_spend <= reserve
            reasons[why] += 1
            count += 1
            boost_sum += boost
            boost_max = max(boost_max, boost)
            boosted += boost > 1
            reference_controller._record_offer(Observation(row, expected, cost, time, distinct))
            if accept:
                reference_controller.blocks_written += cost
                if count <= 1000:
                    first1000 += cost
                if why == 6:
                    startup += cost
            assert reference_controller.blocks_written <= budget
        assert not reuse_history_file.read(1)
    history = list(
        OFFER_HISTORY_RECORD_LAYOUT.iter_unpack((native / "final_history.bin").read_bytes())
    )
    assert history == [
        (v.event_id, v.distinct, v.cost, v.time_us)
        for v in reference_controller.offers_in_arrival_order()
    ]
    assert regular_spend + random_spend == reference_controller.blocks_written
    counts.update(regular_blocks=regular_spend, rescue_blocks=random_spend)
    return dict(
        experiment=mode,
        mechanism=counts,
        mechanism_daily=by_day,
        status="PASS_INDEPENDENT_HISTORY_BONUS_C7_AUDIT",
        observations=count,
        writes=reference_controller.blocks_written,
        bonus=bonus,
        reason_counts=reasons,
        startup_write_blocks=startup,
        first_1000_write_blocks=first1000,
        mean_multiplier=boost_sum / count if count else 1,
        max_multiplier=boost_max,
        boosted_offers=boosted,
        banded=banded,
        half_behind=half_behind,
        discounted_offers=discounted_offers,
        unchanged_C7_quantile=not banded,
        later_repricing=False,
    )
