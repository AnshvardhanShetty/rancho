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
from campaign_state import read_run_json


def audit_policy(root, engine_settings, saved):
    """Check the saved policy inputs and replay the admission decisions"""
    read = lambda p: json.loads(Path(p).read_text())
    receipt = Path(engine_settings["controller_history"])
    assert (
        hashlib.sha256(receipt.read_bytes()).hexdigest()
        == engine_settings["controller_history_sha256"]
    )
    controller_start_record = read(receipt)
    assert (
        controller_start_record["cold"]
        and controller_start_record["observations"]
        == controller_start_record["paid_pages"]
        == 0
        and controller_start_record["fit"] == engine_settings["fit"]
    )
    model = SuffixGBT(engine_settings["fit"])
    assert model.model_sha256 == controller_start_record["model_sha256"]
    workload_settings = read_run_json(root, "workload_settings.json", "SETUP.json")
    payload = prompts(workload_settings)
    info = records(workload_settings)[1]
    reference_controller = BandWallet(
        "hbf",
        engine_settings["HBF_pages"],
        engine_settings["repetition"],
        end_us=CUT + 86400000000,
        release_us=CUT + 86400000000,
        reserve=0.0,
    )
    offers = [
        offer
        for line in (root / "worker.jsonl").read_text().splitlines()
        for offer in json.loads(line).get("offers", [])
    ]
    assert len(offers) == engine_settings["n_requests"]
    cache = []
    counts = {}
    stamps = {}
    seen_requests = set()
    event = 0
    boosted = half = threshold = 0
    for offer in offers:
        i = int(re.fullmatch(r"row(\d+)(?:-[0-9a-f]{8})?", offer["request_id"])[1])
        assert i not in seen_requests
        seen_requests.add(i)
        keys = [b.hex() for b in block_keys(payload[i])]
        session = info[i]["session"]
        reuse_history = offer["history"]
        first = reuse_history["suffix_begin"]
        readable = 0
        for key in keys:
            if key not in cache:
                break
            cache.remove(key)
            cache.append(key)
            readable += 1
        past = sum(counts.get(k, 0) for k in dict.fromkeys(keys[first:]))
        assert reuse_history == dict(
            shadow_cached_pages=readable, past_filtered_reuses=past, suffix_begin=first
        )
        pending = [
            key
            for j, key in enumerate(keys)
            if key not in keys[:j] and j >= readable and stamps.get(key) == session
        ]
        if "decision" in offer:
            d = offer["decision"]
            now = d["controller_time_us"]
            n, c = offer["source_pages"], offer["write_cost_pages"]
            assert n == len(set(keys[first:]))
            prediction = float(model.predict([offer["features"]], now)[0])
            assert prediction == d["prediction"]
            assert d["model_sha256"] == model.model_sha256 and d["fit_cut_us"] == CUT
            score = prediction * float(n) / float(c) * (1 + past / n) if c else math.inf
            reference_controller.phase(now)
            cut, coin = (
                getattr(reference_controller, "bandcuts", {}).get(
                    c.bit_length() - 1,
                    (reference_controller.threshold, reference_controller.tie),
                )
                if c and reference_controller.have
                else (reference_controller.threshold, reference_controller.tie)
            )
            behind = bool(
                c
                and reference_controller.have
                and reference_controller.W * (reference_controller.end_us - CUT)
                < reference_controller.budget * (now - CUT)
            )
            if behind:
                cut /= 2
            before = bool(
                n
                and (
                    not c
                    or c <= reference_controller.budget - reference_controller.W
                    and c <= reference_controller.cap - reference_controller.W
                    and (
                        not reference_controller.have
                        or score > cut
                        or score == cut
                        and hash01(event, reference_controller.seed ^ 0x5BF03635) < coin
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
                and d["have_threshold"] == reference_controller.have
                and d["half_behind"] == behind
                and d["history_multiplier"] == 1 + past / n
                and d["past_filtered_reuses"] == past
            )
            assert d["score_admit_before_physical_guard"] == before and d[
                "physical_guard_rejected"
            ] == (before and not admit)
            reference_controller._observe(Observation(event, score, c, now, n))
            if admit:
                reference_controller.W += c
            assert d["total_paid_writes"] == reference_controller.W and d[
                "paid_cost"
            ] == (c if admit else 0)
            assert offer["action"] in (
                ["COPIED", "REUSED_EXISTING_ONLY"] if admit else ["POLICY_REJECT"]
            )
            if admit:
                assert (
                    offer["native_bytes"] == c * engine_settings["native_block_bytes"]
                    and offer["copy_and_service_ns"]
                    >= offer["native_bytes"]
                    * 1e9
                    / engine_settings["write_bandwidth_bytes_s"]
                )
            boosted += past > 0
            half += behind
            threshold += reference_controller.have
            event += 1
        else:
            assert offer["action"] in [
                "NO_NEW_FULL_SUFFIX",
                "NO_PHYSICAL_DESTINATION_CAPACITY",
            ]
        for key in keys[readable:]:
            if key in cache:
                cache.remove(key)
            cache.append(key)
            while len(cache) > engine_settings["HBM_pages"]:
                cache.pop(0)
        for key in pending:
            counts[key] = counts.get(key, 0) + 1
        for key in keys:
            stamps[key] = session
    assert (
        reference_controller.W
        == saved["admission"]["paid_pages"]
        <= reference_controller.budget
        and len(seen_requests) == engine_settings["n_requests"]
    )
    return dict(
        status="PASS_ARCHIVED_UPDATED_POLICY_AND_INDEPENDENT_HISTORY",
        model_sha256=model.model_sha256,
        decisions=event,
        paid_pages=reference_controller.W,
        history_boosted=boosted,
        half_behind=half,
        threshold_active=threshold,
        history_order="actual completed-prefill order",
        cache_initially_empty=True,
        controller_initially_empty=True,
    )
