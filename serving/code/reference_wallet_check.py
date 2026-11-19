"""Check cost-group score thresholds using a separate controller implementation."""

import array, math, struct
from pathlib import Path
from controllers_reference import Wallet, Observation, hash01

D = struct.Struct("<qqqqqddddqqq")
V = struct.Struct("<qq")
H = struct.Struct("<qqqq")


def eq(a, b):
    return a == b or (
        math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    )


def effective_cutoff(threshold, have, spent, budget, now, split, end, enabled):
    """Use exact integer time/spending comparison, with no forced admission."""
    return (
        threshold / 2
        if enabled and have and spent * (end - split) < budget * (now - split)
        else threshold
    )


class BandWallet(Wallet):
    """Compute rank boundaries independently, then compare every live decision."""

    def recompute(self, now):
        super().recompute(now)
        priced = [v for v in self.ring if v.cost > 0 and math.isfinite(v.score)]
        if len(priced) < 100:
            return
        span = max(
            1.0,
            (max(v.time_us for v in self.ring) - min(v.time_us for v in self.ring))
            / 1e6,
        )
        horizon = max(1.0, (self.end_us - now) / 1e6)
        total = sum(v.cost for v in priced)
        demand = (len(priced) / span) * horizon * (total / len(priced))
        p = min(1.0, max(0.0, float(self.cap - self.W) / max(1e-9, demand)))
        groups = {}
        for v in priced:
            groups.setdefault(v.cost.bit_length() - 1, []).append(v)
        self.bandcuts = {}
        for k, group in groups.items():
            if len(group) < 100:
                continue
            if p in (0.0, 1.0):
                self.bandcuts[k] = (math.inf, 0.0) if p == 0 else (-math.inf, 1.0)
                continue
            masses = {}
            for v in group:
                masses[v.score] = masses.get(v.score, 0) + v.cost
            wanted = p * sum(masses.values())
            above = 0
            for rank in sorted(masses, reverse=True):
                weight = masses[rank]
                if above + weight >= wanted:
                    self.bandcuts[k] = (
                        rank,
                        max(0.0, min(1.0, (wanted - above) / weight)),
                    )
                    break
                above += weight


def audit(
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
    native = Path(native)
    density = array.array("d")
    density.frombytes(Path(density_path).read_bytes())
    w = (BandWallet if banded else Wallet)(
        "hbf", budget, seed, end_us=end, release_us=end, reserve=0
    )
    reasons = [0] * 8
    count = 0
    startup = first1000 = 0
    lastrow = -1
    lasttime = split
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
    with (native / "decisions.bin").open("rb") as f, open(values, "rb") as vf:
        while chunk := f.read(D.size):
            assert len(chunk) == D.size
            (
                row,
                time,
                distinct,
                cost,
                before,
                pred,
                score,
                threshold,
                tie,
                have,
                reason,
                admitted,
            ) = D.unpack(chunk)
            past, vd = V.unpack(vf.read(V.size))
            assert vd == distinct and past >= 0
            assert lastrow < row and lasttime <= time < end
            lastrow, lasttime = row, time
            w.phase(time)
            assert pred == density[row]
            expected = (
                density[row] * float(distinct) / float(cost) if cost else math.inf
            )
            boost = 1 + float(past) / float(distinct) if bonus and distinct else 1
            if soft and cost and boost > 1:
                counts["changed_score"] += 1
                boost = math.sqrt(boost)
            if cost and distinct:
                expected *= boost
            assert eq(score, expected), (row, "score", score, expected)
            cutoff, coin = (
                getattr(w, "bandcuts", {}).get(
                    cost.bit_length() - 1, (w.threshold, w.tie)
                )
                if banded and w.have and cost
                else (w.threshold, w.tie)
            )
            if (
                cost
                and w.have
                and half_behind
                and w.W * (end - split) < budget * (time - split)
            ):
                discounted_offers += 1
                cutoff = effective_cutoff(
                    cutoff, w.have, w.W, budget, time, split, end, True
                )
            assert (
                before == w.W
                and bool(have) == w.have
                and eq(threshold, cutoff)
                and eq(tie, coin)
            ), (row, "controller")
            why = (
                1
                if not distinct
                else (
                    2
                    if not cost
                    else (
                        3
                        if cost > budget - w.W
                        else (
                            4
                            if cost > w.cap - w.W
                            else (
                                6
                                if not w.have
                                else (
                                    7
                                    if expected > cutoff
                                    or (
                                        expected == cutoff
                                        and hash01(row, seed ^ 0x5BF03635) < coin
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
                    if hash01(row, seed ^ 0x72A11D) < probability:
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
                assert (
                    0 <= regular_spend <= budget - reserve
                    and 0 <= random_spend <= reserve
                )
            reasons[why] += 1
            count += 1
            boost_sum += boost
            boost_max = max(boost_max, boost)
            boosted += boost > 1
            w._observe(Observation(row, expected, cost, time, distinct))
            if accept:
                w.W += cost
                if count <= 1000:
                    first1000 += cost
                if why == 6:
                    startup += cost
            assert w.W <= budget
        assert not vf.read(1)
    history = list(H.iter_unpack((native / "final_history.bin").read_bytes()))
    assert history == [
        (v.event_id, v.distinct, v.cost, v.time_us) for v in w.chronological()
    ]
    assert regular_spend + random_spend == w.W
    counts.update(regular_blocks=regular_spend, rescue_blocks=random_spend)
    return dict(
        experiment=mode,
        mechanism=counts,
        mechanism_daily=by_day,
        status="PASS_INDEPENDENT_HISTORY_BONUS_C7_AUDIT",
        observations=count,
        writes=w.W,
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
