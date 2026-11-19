"""Check shared-prefix ownership and report concurrency after cache filling.

Each distinct shared block has one retention reference charged to HBM.
The checker follows recorded ownership changes and verifies reference counts."""

import json


def audit_retention(root, cfg, result):
    rows = [json.loads(x) for x in (root / "retention.jsonl").open()]
    assert rows[0]["event"] == "RETENTION_START" and rows[0]["retained_pages"] == 0
    expected = cfg["shared_groups"] * cfg["shared_prefix_tokens"] // cfg["block_tokens"]
    ids = set()
    hashes = set()
    full_step = None
    for row in rows:
        if row["event"] == "PREFIX_PAGES_RETAINED":
            assert row["gpu_completed"] is True
            for added in row["added"]:
                assert (
                    0 < added["block_id"] <= cfg["HBM_pages"]
                    and added["block_id"] not in ids
                )
                assert added["block_hash"] not in hashes
                ids.add(added["block_id"])
                hashes.add(added["block_hash"])
            assert row["retained_pages"] == len(ids) <= expected
            assert (
                row["eviction_attempts"] == 0
                and row["protected_pages_in_free_queue"] == 0
            )
            if len(ids) == expected and full_step is None:
                full_step = row["step"]
    end = rows[-1]
    assert (
        end["event"] == "RETENTION_COMPLETE"
        and end["retained_pages"] == len(ids) == expected
    )
    assert end["eviction_attempts"] == 0 and end["protected_pages_in_free_queue"] == 0
    steps = [json.loads(x) for x in (root / "concurrency.jsonl").open()]
    after = [x for x in steps if x["step"] > full_step]
    return dict(
        policy="retain declared shared prefixes; native memory admission",
        retained_pages=expected,
        retained_bytes=expected * cfg["native_block_bytes"],
        full_retention_step=full_step,
        eviction_attempts=0,
        after_fill_steps=len(after),
        after_fill_peak_resident_requests=max(
            (x["resident_requests"] for x in after), default=0
        ),
        after_fill_peak_scheduled_decode=max(
            (x["scheduled_decode_requests"] for x in after), default=0
        ),
        after_fill_allocation_refusals=sum(
            x["native_KV_allocation_refusals"] for x in after
        ),
        actual_hit_ratio=result["hit_ratio"],
        native_occupancy_includes_retained_idle_prefixes=True,
    )
