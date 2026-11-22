"""Build the four reuse inputs from requests in arrival order."""

import numpy as np

from .trace_reader import (
    ARRIVAL_ORDER_PREFIX,
    ARRIVAL_ORDER_TO_INPUT_ROWS,
    MODEL,
    loads,
    trace_array_paths,
)


def build_request_features(trace_path, prepared_directory, features_path):
    """Compute each request’s features before updating its session history."""
    # Store the user ID from the trace as a number. The same ID gets the same
    # number throughout this trace. Save one number per kept request in file order.
    user_number_by_id = {}
    user_numbers_in_input_order = []
    with trace_path.open("rb") as trace_file:
        for line in trace_file:
            request_fields = loads(line)
            if request_fields.get("model_id") == MODEL:
                user_id = request_fields["user_id"]
                if not isinstance(user_id, str) or not user_id:
                    raise ValueError("Invalid user ID")
                if user_id not in user_number_by_id:
                    user_number_by_id[user_id] = len(user_number_by_id)
                user_numbers_in_input_order.append(user_number_by_id[user_id])

    # Each entry gives the original row of a sorted request. For example,
    # [2, 0, 1] means take the user numbers from rows 2, 0, and 1 in that order.
    input_row_for_each_arrival = np.fromfile(
        prepared_directory / ARRIVAL_ORDER_TO_INPUT_ROWS, "<i8"
    )
    user_numbers_in_arrival_order = np.array(user_numbers_in_input_order, dtype="<i8")[
        input_row_for_each_arrival
    ]
    _, session_numbers_path, arrival_times_path, _ = trace_array_paths(
        prepared_directory / ARRIVAL_ORDER_PREFIX
    )
    session_codes = np.fromfile(session_numbers_path, "<i4")
    arrival_times_us = np.fromfile(arrival_times_path, "<i8")

    # Work out what was known about each session when the request arrived.
    turn_number, session_age_seconds, previous_gap_seconds = np.empty(
        (3, len(arrival_times_us)), dtype=float
    )
    session_history = {}
    for request_index, (session_number, arrival_us) in enumerate(
        zip(session_codes, arrival_times_us)
    ):
        # On the first turn the count and age are zero. A gap of -1 means no previous turn.
        prior_turn_count, first_arrival_us, previous_arrival_us = session_history.get(
            session_number, (0, arrival_us, arrival_us)
        )
        turn_number[request_index] = prior_turn_count
        session_age_seconds[request_index] = (arrival_us - first_arrival_us) / 1e6
        previous_gap_seconds[request_index] = (
            (arrival_us - previous_arrival_us) / 1e6 if prior_turn_count else -1
        )
        session_history[session_number] = prior_turn_count + 1, first_arrival_us, arrival_us

    # Save the four features under the names used in model_settings.json.
    np.savez(
        features_path,
        user=user_numbers_in_arrival_order,
        causal_request_position=turn_number,
        session_age_seconds=session_age_seconds,
        previous_session_gap_seconds=previous_gap_seconds,
    )
    return arrival_times_us
