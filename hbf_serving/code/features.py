"""Track earlier requests and block accesses for the reuse predictors"""

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
    """The request details available when that request arrives"""

    row: int
    time_us: int
    user: int
    position: int
    age_seconds: float
    gap_seconds: float

    def request_feature_values(self):
        return dict(
            user=self.user,
            causal_request_position=self.position,
            session_age_seconds=self.age_seconds,
            previous_session_gap_seconds=self.gap_seconds,
        )


class RequestHistory:
    """Remember when each chat started and when it last made a request"""

    def __init__(self):
        self.sessions = {}
        self.next_row = 0
        self.last_time_us = -(1 << 63)

    def record_request_arrival(self, row, time_us, session, user):
        """Read the earlier chat state before adding the current request"""
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
        earlier_session = self.sessions.get(session)
        position, session_age_seconds, previous_gap_seconds = (
            (0, 0.0, -1.0)
            if earlier_session is None
            else (
                earlier_session[0],
                (time_us - earlier_session[1]) / 1e6,
                (time_us - earlier_session[2]) / 1e6,
            )
        )
        first_arrival_us = time_us if earlier_session is None else earlier_session[1]
        self.sessions[session] = (position + 1, first_arrival_us, time_us)
        self.next_row += 1
        self.last_time_us = time_us
        return RequestInfo(
            row, time_us, user, position, session_age_seconds, previous_gap_seconds
        )


@dataclass
class Block:
    """The earlier accesses and evictions of one KV block"""

    last: RequestInfo | None = None
    touches: int = 0
    evictions: int = 0
    born_us: int | None = None
    last_touch_row: int = -1


class BlockHistory:
    """One shared history per physical block identity, independent of admission."""

    def __init__(self):
        self.blocks = {}

    def record_hbm_entry(self, block_id, request):
        if type(block_id) is not int or block_id < 0:
            raise ValueError("Invalid block ID")
        block_history = self.blocks.setdefault(block_id, Block())
        if block_history.born_us is not None:
            raise ValueError("HBM entry for already resident block")
        block_history.born_us = request.time_us

    def record_block_reads(self, block_ids, request):
        # The physical replay counts a block once per request, even if repeated
        # at multiple prompt positions. No admission status enters this update.
        """Update block history once per block in this request"""
        for block_id in dict.fromkeys(block_ids):
            block_history = self.blocks.get(block_id)
            if block_history is None or block_history.born_us is None:
                raise ValueError("HBM touch for absent block")
            if block_history.last_touch_row >= request.row or (
                block_history.last is not None
                and block_history.last.time_us > request.time_us
            ):
                raise ValueError("Duplicate or reordered request history update")
            block_history.last = request
            block_history.touches += 1
            block_history.last_touch_row = request.row

    def record_eviction_and_get_features(self, block_id, request):
        """Get the history of a block as it leaves memory"""
        block_history = self.blocks.get(block_id)
        if (
            block_history is None
            or block_history.last is None
            or block_history.born_us is None
        ):
            raise ValueError("Eviction lacks complete HBM prehistory")
        if (
            block_history.last.row >= request.row
            or block_history.last.time_us > request.time_us
            or block_history.born_us > request.time_us
        ):
            raise ValueError("Eviction history is not strictly prior to this request")
        # SSD uses the evicted block's last-touch request metadata, not the
        # request that happened to displace it. Counters precede this eviction.
        result = dict(
            block_history.last.request_feature_values(),
            prior_eviction_count=block_history.evictions,
            prior_touch_count=block_history.touches,
            seconds_in_HBM=(request.time_us - block_history.born_us) / 1e6,
            seconds_since_last_touch=(request.time_us - block_history.last.time_us)
            / 1e6,
        )
        block_history.evictions += 1
        block_history.born_us = None
        return result
