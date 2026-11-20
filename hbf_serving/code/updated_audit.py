"""Check serving admission decisions against the recorded events.

Rebuild earlier reuse in completed-prefill order, regenerate tree predictions,
and compare decisions with the separate controller implementation.
Hit counts and physical write limits are checked by the other run checks."""

from pathlib import Path
import hashlib, json, math, re
from reference_wallet_check import BandWallet
from controllers_reference import Observation, hash01
from updated_model import SuffixGBT, CUT
from controlled_workload import prompts, block_keys, records


def audit_policy(root, cfg, saved):
    """Check the saved policy inputs and replay the admission decisions"""
    read = lambda p: json.loads(Path(p).read_text())
    receipt = Path(cfg["controller_history"])
    assert (
        hashlib.sha256(receipt.read_bytes()).hexdigest()
        == cfg["controller_history_sha256"]
    )
    r = read(receipt)
    assert (
        r["cold"]
        and r["observations"] == r["paid_pages"] == 0
        and r["fit"] == cfg["fit"]
    )
    model = SuffixGBT(cfg["fit"])
    assert model.model_sha256 == r["model_sha256"]
    setup = read(root / "SETUP.json")
    payload = prompts(setup)
    info = records(setup)[1]
    w = BandWallet(
        "hbf",
        cfg["HBF_pages"],
        cfg["repetition"],
        end_us=CUT + 86400000000,
        release_us=CUT + 86400000000,
        reserve=0.0,
    )
    offers = [
        o
        for line in (root / "worker.jsonl").read_text().splitlines()
        for o in json.loads(line).get("offers", [])
    ]
    assert len(offers) == cfg["n_requests"]
    cache = []
    counts = {}
    stamps = {}
    seen_requests = set()
    event = 0
    boosted = half = threshold = 0
    for o in offers:
        i = int(re.fullmatch(r"row(\d+)(?:-[0-9a-f]{8})?", o["request_id"])[1])
        assert i not in seen_requests
        seen_requests.add(i)
        keys = [b.hex() for b in block_keys(payload[i])]
        session = info[i]["session"]
        h = o["history"]
        first = h["suffix_begin"]
        readable = 0
        for key in keys:
            if key not in cache:
                break
            cache.remove(key)
            cache.append(key)
            readable += 1
        past = sum(counts.get(k, 0) for k in dict.fromkeys(keys[first:]))
        assert h == dict(
            shadow_cached_pages=readable, past_filtered_reuses=past, suffix_begin=first
        )
        pending = [
            key
            for j, key in enumerate(keys)
            if key not in keys[:j] and j >= readable and stamps.get(key) == session
        ]
        if "decision" in o:
            d = o["decision"]
            now = d["controller_time_us"]
            n, c = o["source_pages"], o["write_cost_pages"]
            assert n == len(set(keys[first:]))
            prediction = float(model.predict([o["features"]], now)[0])
            assert prediction == d["prediction"]
            assert d["model_sha256"] == model.model_sha256 and d["fit_cut_us"] == CUT
            score = prediction * float(n) / float(c) * (1 + past / n) if c else math.inf
            w.phase(now)
            cut, coin = (
                getattr(w, "bandcuts", {}).get(c.bit_length() - 1, (w.threshold, w.tie))
                if c and w.have
                else (w.threshold, w.tie)
            )
            behind = bool(
                c and w.have and w.W * (w.end_us - CUT) < w.budget * (now - CUT)
            )
            if behind:
                cut /= 2
            before = bool(
                n
                and (
                    not c
                    or c <= w.budget - w.W
                    and c <= w.cap - w.W
                    and (
                        not w.have
                        or score > cut
                        or score == cut
                        and hash01(event, w.seed ^ 0x5BF03635) < coin
                    )
                )
            )
            admit = before and c <= d["physical_write_limit_pages"]
            safe = lambda v: v if math.isfinite(v) else str(v)
            assert (
                d["score"] == safe(score)
                and d["cutoff"] == safe(cut)
                and d["tie"] == coin
            )
            assert (
                d["admit"] == admit
                and d["have_threshold"] == w.have
                and d["half_behind"] == behind
                and d["history_multiplier"] == 1 + past / n
                and d["past_filtered_reuses"] == past
            )
            assert d["score_admit_before_physical_guard"] == before and d[
                "physical_guard_rejected"
            ] == (before and not admit)
            w._observe(Observation(event, score, c, now, n))
            if admit:
                w.W += c
            assert d["total_paid_writes"] == w.W and d["paid_cost"] == (
                c if admit else 0
            )
            assert o["action"] in (
                ["COPIED", "REUSED_EXISTING_ONLY"] if admit else ["POLICY_REJECT"]
            )
            if admit:
                assert (
                    o["native_bytes"] == c * cfg["native_block_bytes"]
                    and o["copy_and_service_ns"]
                    >= o["native_bytes"] * 1e9 / cfg["write_bandwidth_bytes_s"]
                )
            boosted += past > 0
            half += behind
            threshold += w.have
            event += 1
        else:
            assert o["action"] in [
                "NO_NEW_FULL_SUFFIX",
                "NO_PHYSICAL_DESTINATION_CAPACITY",
            ]
        for key in keys[readable:]:
            if key in cache:
                cache.remove(key)
            cache.append(key)
            while len(cache) > cfg["HBM_pages"]:
                cache.pop(0)
        for key in pending:
            counts[key] = counts.get(key, 0) + 1
        for key in keys:
            stamps[key] = session
    assert (
        w.W == saved["admission"]["paid_pages"] <= w.budget
        and len(seen_requests) == cfg["n_requests"]
    )
    return dict(
        status="PASS_ARCHIVED_UPDATED_POLICY_AND_INDEPENDENT_HISTORY",
        model_sha256=model.model_sha256,
        decisions=event,
        paid_pages=w.W,
        history_boosted=boosted,
        half_behind=half,
        threshold_active=threshold,
        history_order="actual completed-prefill order",
        cache_initially_empty=True,
        controller_initially_empty=True,
    )
