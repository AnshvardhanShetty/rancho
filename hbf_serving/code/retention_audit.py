"""Check shared-prefix ownership and report concurrency after cache filling.

Each distinct shared block has one retention reference charged to HBM.
The checker follows recorded ownership changes and verifies reference counts."""

import json


def audit_retention(root, engine_settings, result):
    """Check which shared pages were kept and how many requests ran together"""
    rows = [json.loads(line) for line in (root / "retention.jsonl").open()]
    assert rows[0]["event"] == "RETENTION_START" and rows[0]["retained_pages"] == 0
    expected = (
        engine_settings["shared_groups"]
        * engine_settings["shared_prefix_tokens"]
        // engine_settings["block_tokens"]
    )
    retained_page_ids = set()
    hashes = set()
    full_step = None
    for row in rows:
        if row["event"] == "PREFIX_PAGES_RETAINED":
            assert row["gpu_completed"] is True
            for added in row["added"]:
                assert (
                    0 < added["block_id"] <= engine_settings["HBM_pages"]
                    and added["block_id"] not in retained_page_ids
                )
                assert added["block_hash"] not in hashes
                retained_page_ids.add(added["block_id"])
                hashes.add(added["block_hash"])
            assert row["retained_pages"] == len(retained_page_ids) <= expected
            assert row["eviction_attempts"] == 0 and row["protected_pages_in_free_queue"] == 0
            if len(retained_page_ids) == expected and full_step is None:
                full_step = row["step"]
    final_retention_record = rows[-1]
    assert (
        final_retention_record["event"] == "RETENTION_COMPLETE"
        and final_retention_record["retained_pages"] == len(retained_page_ids) == expected
    )
    assert (
        final_retention_record["eviction_attempts"] == 0
        and final_retention_record["protected_pages_in_free_queue"] == 0
    )
    steps = [json.loads(line) for line in (root / "concurrency.jsonl").open()]
    after = [line for line in steps if line["step"] > full_step]
    return dict(
        policy="retain declared shared prefixes; native memory admission",
        retained_pages=expected,
        retained_bytes=expected * engine_settings["native_block_bytes"],
        full_retention_step=full_step,
        eviction_attempts=0,
        after_fill_steps=len(after),
        after_fill_peak_resident_requests=max(
            (line["resident_requests"] for line in after), default=0
        ),
        after_fill_peak_scheduled_decode=max(
            (line["scheduled_decode_requests"] for line in after), default=0
        ),
        after_fill_allocation_refusals=sum(
            line["native_KV_allocation_refusals"] for line in after
        ),
        actual_hit_ratio=result["hit_ratio"],
        native_occupancy_includes_retained_idle_prefixes=True,
    )
