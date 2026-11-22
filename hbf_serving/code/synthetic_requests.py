"""Make synthetic chats with shared prompt blocks and unique endings"""

import hashlib, struct
from request_history import RequestHistory

SYNTHETIC_TRAINING_CUTOFF_US = 259_200_000_000


def build_prompt_tokens(setup, request_index, calibration=False, warm=False):
    """Build one synthetic prompt with a shared beginning and a unique ending."""
    conversation_index = request_index % setup["groups"]
    prefix_token_count = setup["prefix_tokens"]
    unique_token_count = setup["unique_suffix_tokens"]
    repetition = setup.get("repetition", 17)
    base = 4000 if calibration else 1000
    prefix = [
        base + conversation_index * 64 + prefix_position % 32
        for prefix_position in range(prefix_token_count)
    ]
    suffix_token = (
        10000 + request_index
        if calibration
        else (30000 + conversation_index if warm else 20000 + repetition * 1300 + request_index)
    )
    return prefix + [suffix_token] * unique_token_count


def build_workload_prompts(setup, calibration=False):
    """Build warm-up prompts followed by the measured prompts."""
    if calibration:
        return [
            build_prompt_tokens(setup, request_index, True)
            for request_index in range(setup["calibration_requests"])
        ]
    return [
        build_prompt_tokens(setup, conversation_index, warm=True)
        for conversation_index in range(setup["groups"])
    ] + [build_prompt_tokens(setup, request_index) for request_index in range(setup["requests"])]


def build_request_feature_records(setup):
    """Build request features from synthetic earlier session history."""
    request_history = RequestHistory()
    reference_request_count = setup["max_reference_requests"]
    calibration_request_count = setup["calibration_requests"]
    request_gap_us = setup["logical_gap_seconds"] * 1000000
    training_cutoff_us = SYNTHETIC_TRAINING_CUTOFF_US
    sequence_row = 0
    for request_index in range(reference_request_count):
        request_history.record_request_arrival(
            sequence_row,
            training_cutoff_us
            - (reference_request_count + calibration_request_count - request_index)
            * request_gap_us,
            request_index % setup["groups"],
            setup["synthetic_user"],
        )
        sequence_row += 1
    earlier = []
    for request_index in range(calibration_request_count):
        arrival_us = (
            training_cutoff_us - (calibration_request_count - request_index) * request_gap_us
        )
        info = request_history.record_request_arrival(
            sequence_row, arrival_us, 10000 + request_index, setup["synthetic_user"]
        )
        sequence_row += 1
        earlier.append(
            dict(
                sequence_row=sequence_row - 1,
                session=10000 + request_index,
                policy_us=arrival_us,
                features=info.request_feature_values(),
            )
        )
    measured = []
    for request_index in range(setup["groups"] + setup["requests"]):
        conversation_index = (
            request_index
            if request_index < setup["groups"]
            else (request_index - setup["groups"]) % setup["groups"]
        )
        arrival_us = training_cutoff_us + request_index * request_gap_us
        info = request_history.record_request_arrival(
            sequence_row, arrival_us, conversation_index, setup["synthetic_user"]
        )
        sequence_row += 1
        measured.append(
            dict(
                sequence_row=sequence_row - 1,
                session=conversation_index,
                policy_us=arrival_us,
                features=info.request_feature_values(),
                warmup=request_index < setup["groups"],
            )
        )
    return earlier, measured


def bind_features_to_request_ids(setup):
    """Associate each engine request ID with its features and session."""
    return {
        f"row{request_index}": dict(
            features=request_record["features"],
            policy_us=request_record["policy_us"],
            session=request_record["session"],
        )
        for request_index, request_record in enumerate(build_request_feature_records(setup)[1])
    }


def block_keys(prompt):
    """Assign content-based IDs to full pages, including their preceding context."""
    if len(prompt) % 16:
        raise ValueError("Only full input blocks in this controlled study")
    prefix_hash = bytes(32)
    block_hashes = []
    for token_start in range(0, len(prompt), 16):
        prefix_hash = hashlib.sha256(
            prefix_hash + struct.pack("<16I", *prompt[token_start : token_start + 16])
        ).digest()
        block_hashes.append(prefix_hash)
    return block_hashes


def build_eligible_read_mask(setup):
    # Reconstruct independent offline-style reference stamps before any cache fill.
    """Mark repeated reads using the independent synthetic reference history."""
    last_session_by_block = {}
    for request_index in range(setup["max_reference_requests"]):
        for block_hash in block_keys(build_prompt_tokens(setup, request_index)):
            last_session_by_block[block_hash] = request_index % setup["groups"]
    for request_index in range(setup["calibration_requests"]):
        for block_hash in block_keys(build_prompt_tokens(setup, request_index, calibration=True)):
            last_session_by_block[block_hash] = 10000 + request_index
    eligibility_by_request = []
    for request_index, prompt in enumerate(build_workload_prompts(setup)):
        group = (
            request_index
            if request_index < setup["groups"]
            else (request_index - setup["groups"]) % setup["groups"]
        )
        keys = block_keys(prompt)
        seen = set()
        bits = []
        for block_hash in keys:
            bits.append(block_hash not in seen and last_session_by_block.get(block_hash) == group)
            seen.add(block_hash)
            last_session_by_block[block_hash] = group
        eligibility_by_request.append(bits)
    return eligibility_by_request
