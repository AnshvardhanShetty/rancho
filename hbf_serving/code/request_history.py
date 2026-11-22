"""Track earlier requests in each chat to build the four HBF features"""

from dataclasses import dataclass


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
