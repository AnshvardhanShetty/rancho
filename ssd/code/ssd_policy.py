"""Describe the information available when deciding whether to save evicted KV.

Offer groups the block's past accesses, request inputs and remaining write budget.
The engine passes one Offer to a supplied function and expects True or False.
The two fixture functions always accept or reject, for correctness checks only.

Assumes request metadata is supplied by the caller. These fixtures do not train
or substitute for the learned RANCHO predictor."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Offer:
    """Hold the data needed to decide whether to save one evicted block."""

    key: bytes
    eviction_id: int
    absent_offer_id: int
    now_seconds: float
    page_bytes: int
    written_pages: int
    write_budget_pages: int
    ssd_capacity_pages: int
    prior_eviction_count: int
    prior_touch_count: int
    seconds_in_HBM: float
    seconds_since_last_touch: float
    last_request_id: str
    request_features: Mapping


def accept_fixture(settings):
    """Accept every offer in the small storage correctness example."""
    return lambda offer: True


def reject_fixture(settings):
    """Reject every offer in the matching no-write example."""
    return lambda offer: False
