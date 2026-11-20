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
        request_index = int(
            re.fullmatch(r"row(\d+)(?:-[0-9a-f]{8})?", offer["request_id"])[1]
        )
        assert request_index not in seen_requests
        seen_requests.add(request_index)
        keys = [b.hex() for b in block_keys(payload[request_index])]
        session = info[request_index]["session"]
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
            decision = offer["decision"]
            arrival_us = decision["controller_time_us"]
            n, write_cost = offer["source_pages"], offer["write_cost_pages"]
            assert n == len(set(keys[first:]))
            prediction = float(model.predict([offer["features"]], arrival_us)[0])
            assert prediction == decision["prediction"]
            assert (
                decision["model_sha256"] == model.model_sha256
                and decision["fit_cut_us"] == CUT
            )
            score = (
                prediction * float(n) / float(write_cost) * (1 + past / n)
                if write_cost
                else math.inf
            )
            reference_controller.phase(arrival_us)
            cutoff_score, coin = (
                getattr(reference_controller, "cutoffs_by_cost_band", {}).get(
                    write_cost.bit_length() - 1,
                    (
                        reference_controller.threshold,
                        reference_controller.tie_probability,
                    ),
                )
                if write_cost and reference_controller.cutoff_ready
                else (
                    reference_controller.threshold,
                    reference_controller.tie_probability,
                )
            )
            behind = bool(
                write_cost
                and reference_controller.cutoff_ready
                and reference_controller.blocks_written
                * (reference_controller.end_us - CUT)
                < reference_controller.budget * (arrival_us - CUT)
            )
            if behind:
                cutoff_score /= 2
            before = bool(
                n
                and (
                    not write_cost
                    or write_cost
                    <= reference_controller.budget - reference_controller.blocks_written
                    and write_cost
                    <= reference_controller.current_spending_limit
                    - reference_controller.blocks_written
                    and (
                        not reference_controller.cutoff_ready
                        or score > cutoff_score
                        or score == cutoff_score
                        and hash01(event, reference_controller.seed ^ 0x5BF03635) < coin
                    )
                )
            )
            admit = before and write_cost <= decision["physical_write_limit_pages"]
            safe = lambda v: v if math.isfinite(v) else str(v)
            assert (
                decision["score"] == safe(score)
                and decision["cutoff"] == safe(cutoff_score)
                and decision["tie"] == coin
            )
            assert (
                decision["admit"] == admit
                and decision["have_threshold"] == reference_controller.cutoff_ready
                and decision["half_behind"] == behind
                and decision["history_multiplier"] == 1 + past / n
                and decision["past_filtered_reuses"] == past
            )
            assert decision["score_admit_before_physical_guard"] == before and decision[
                "physical_guard_rejected"
            ] == (before and not admit)
            reference_controller._observe(
                Observation(event, score, write_cost, arrival_us, n)
            )
            if admit:
                reference_controller.blocks_written += write_cost
            assert decision[
                "total_paid_writes"
            ] == reference_controller.blocks_written and decision["paid_cost"] == (
                write_cost if admit else 0
            )
            assert offer["action"] in (
                ["COPIED", "REUSED_EXISTING_ONLY"] if admit else ["POLICY_REJECT"]
            )
            if admit:
                assert (
                    offer["native_bytes"]
                    == write_cost * engine_settings["native_block_bytes"]
                    and offer["copy_and_service_ns"]
                    >= offer["native_bytes"]
                    * 1e9
                    / engine_settings["write_bandwidth_bytes_s"]
                )
            boosted += past > 0
            half += behind
            threshold += reference_controller.cutoff_ready
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
        reference_controller.blocks_written
        == saved["admission"]["paid_pages"]
        <= reference_controller.budget
        and len(seen_requests) == engine_settings["n_requests"]
    )
    return dict(
        status="PASS_ARCHIVED_UPDATED_POLICY_AND_INDEPENDENT_HISTORY",
        model_sha256=model.model_sha256,
        decisions=event,
        paid_pages=reference_controller.blocks_written,
        history_boosted=boosted,
        half_behind=half,
        threshold_active=threshold,
        history_order="actual completed-prefill order",
        cache_initially_empty=True,
        controller_initially_empty=True,
    )
