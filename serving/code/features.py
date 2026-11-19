"""Causal input extraction for the selected HBF and SSD RANCHO predictors.

Assumptions: caller supplies authenticated original integer user/session IDs,
arrival order and logical timestamps, including negative-time prehistory.
Request metadata advances on every arrival, irrespective of admissions. SSD
block counters advance only through actual HBM entry/touch/eviction callbacks;
the caller, not this module, implements HBM capacity/LRU and oversized bypass.

The same-request SSD history update occurs after its eviction offers, exactly
as in the pinned replay. This module does not infer cache state from a hit
metric, equate trace hashes with Qwen payloads, or decide how wall-clock request
compression affects the policy clock. Native callback wiring remains separate.
"""

from dataclasses import dataclass

SHARED = (
    "user",
    "causal_request_position",
    "session_age_seconds",
    "previous_session_gap_seconds",
)
HISTORY = (
    "prior_eviction_count",
    "prior_touch_count",
    "seconds_in_HBM",
    "seconds_since_last_touch",
)


@dataclass(frozen=True)
class RequestInfo:
    row: int
    time_us: int
    user: int
    position: int
    age_seconds: float
    gap_seconds: float

    def shared(self):
        return dict(
            user=self.user,
            causal_request_position=self.position,
            session_age_seconds=self.age_seconds,
            previous_session_gap_seconds=self.gap_seconds,
        )


class RequestHistory:
    def __init__(self):
        self.sessions = {}
        self.next_row = 0
        self.last_time_us = -(1 << 63)

    def arrival(self, row, time_us, session, user):
        if (
            type(row) is not int
            or row != self.next_row
            or type(time_us) is not int
            or time_us < self.last_time_us
            or type(session) is not int
            or session < 0
            or type(user) is not int
            or user < 0
        ):
            raise ValueError("Original request order/IDs/clock required")
        old = self.sessions.get(session)
        position, age, gap = (
            (0, 0.0, -1.0)
            if old is None
            else (old[0], (time_us - old[1]) / 1e6, (time_us - old[2]) / 1e6)
        )
        first = time_us if old is None else old[1]
        self.sessions[session] = (position + 1, first, time_us)
        self.next_row += 1
        self.last_time_us = time_us
        return RequestInfo(row, time_us, user, position, age, gap)


@dataclass
class Block:
    last: RequestInfo | None = None
    touches: int = 0
    evictions: int = 0
    born_us: int | None = None
    last_touch_row: int = -1


class BlockHistory:
    """One shared history per physical block identity, independent of admission."""

    def __init__(self):
        self.blocks = {}

    def enter(self, block_id, request):
        if type(block_id) is not int or block_id < 0:
            raise ValueError("Invalid block ID")
        b = self.blocks.setdefault(block_id, Block())
        if b.born_us is not None:
            raise ValueError("HBM entry for already resident block")
        b.born_us = request.time_us

    def touch_request(self, block_ids, request):
        # The physical replay counts a block once per request, even if repeated
        # at multiple prompt positions. No admission status enters this update.
        for block_id in dict.fromkeys(block_ids):
            b = self.blocks.get(block_id)
            if b is None or b.born_us is None:
                raise ValueError("HBM touch for absent block")
            if b.last_touch_row >= request.row or (
                b.last is not None and b.last.time_us > request.time_us
            ):
                raise ValueError("Duplicate or reordered request history update")
            b.last = request
            b.touches += 1
            b.last_touch_row = request.row

    def eviction(self, block_id, request):
        b = self.blocks.get(block_id)
        if b is None or b.last is None or b.born_us is None:
            raise ValueError("Eviction lacks complete HBM prehistory")
        if (
            b.last.row >= request.row
            or b.last.time_us > request.time_us
            or b.born_us > request.time_us
        ):
            raise ValueError("Eviction history is not strictly prior to this request")
        # SSD uses the evicted block's last-touch request metadata, not the
        # request that happened to displace it. Counters precede this eviction.
        result = dict(
            b.last.shared(),
            prior_eviction_count=b.evictions,
            prior_touch_count=b.touches,
            seconds_in_HBM=(request.time_us - b.born_us) / 1e6,
            seconds_since_last_touch=(request.time_us - b.last.time_us) / 1e6,
        )
        b.evictions += 1
        b.born_us = None
        return result
