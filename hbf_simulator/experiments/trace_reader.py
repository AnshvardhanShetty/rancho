"""Convert a JSONL trace into the arrays used by the predictor and simulator.

Read the four numbered sections in preparation order:
1. parse_request checks one request after its JSON line has been read.
2. read_requests repeats that check, keeps the selected model and numbers sessions. It
   saves the input-order block IDs to disk for the sorting pass.
3. write_sorted_requests saves the same requests in arrival order in arrival_order_*
   files. Block order and repeated IDs stay the same within each request.
4. prepare_trace calls those steps and records the file checks.

The request container and file helpers are below the four main sections.

arrival_order_to_input_rows.i64 maps sorted requests back to their input rows.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np

try:
    import orjson

    loads = orjson.loads
except ImportError:
    loads = json.loads

MODEL = "deepseek-v4-flash"
TEST_START_US = 3 * 86400 * 1_000_000  # Start of day 4, in microseconds.
TEST_END_US = 7 * 86400 * 1_000_000  # End of day 7, excluded.
MAX_SIGNED_INT32 = 2**31 - 1
ARRIVAL_ORDER_PREFIX = "arrival_order"
ARRIVAL_ORDER_TO_INPUT_ROWS = "arrival_order_to_input_rows.i64"


# 1. Check one decoded request; skip other models and reject invalid values.
def parse_request(row):
    """Read the session, arrival time, and block IDs for the selected model."""
    if row.get("model_id") != MODEL:
        return None

    # Check the session and time before reading the block IDs.
    session_id = row["session_id"]
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Invalid session_id")
    arrival_seconds = row["timestamp_arrival"]
    if (
        isinstance(arrival_seconds, bool)
        or not isinstance(arrival_seconds, (float, int))
        or not math.isfinite(arrival_seconds)
    ):
        raise ValueError("Invalid timestamp_arrival")
    # Preserve block order and duplicates; use the four-byte format C++ expects.
    block_ids = row["hash_ids"]
    if not isinstance(block_ids, list):
        raise ValueError("hash_ids must be list")
    if block_ids:
        block_array = np.asarray(block_ids)
        if (
            block_array.dtype.kind not in "iu"
            or block_array.min() < 0
            or block_array.max() > MAX_SIGNED_INT32
        ):
            raise ValueError("Block IDs do not fit nonnegative int32: change schemas explicitly")
        block_array = block_array.astype("<i4", copy=False)
    else:
        block_array = np.empty(0, dtype="<i4")
    if len(block_array) > MAX_SIGNED_INT32:
        raise ValueError("Request length overflow")
    return session_id, float(arrival_seconds), block_array


# 2. Read the trace, number sessions, and save the input-order block lists.
def read_requests(trace_path, output_directory, started_at):
    """Save block lists and collect each request's length, session, and time."""
    session_codes_by_id = {}
    block_counts = []
    session_codes = []
    arrival_times = []
    block_offset = 0
    max_block_id = -1
    input_block_checksum = hashlib.sha256()
    trace_checksum = hashlib.sha256()
    skipped_requests = empty_requests = 0

    # Join block IDs in a scratch file so the whole trace need not fit in memory.
    block_ids_path = output_directory / "input_order_block_ids.i32"

    with trace_path.open("rb") as trace, block_ids_path.open(
        "xb", buffering=8 * 1024**2
    ) as block_file:
        for input_row, line in enumerate(trace):
            trace_checksum.update(line)
            request = parse_request(loads(line))
            if request is None:
                skipped_requests += 1
                continue
            session_id, arrival_seconds, block_ids = request

            # Assign codes in first-seen order. Different sessions stay separate.
            if session_id not in session_codes_by_id:
                session_codes_by_id[session_id] = len(session_codes_by_id)
            session_code = session_codes_by_id[session_id]
            request_index = len(block_counts)
            if session_code > MAX_SIGNED_INT32:
                raise ValueError("Session index overflow")
            if request_index > MAX_SIGNED_INT32:
                raise ValueError("Request indices require wider schema")

            # Join block lists into one file. Lengths let us recover each list.
            block_bytes = block_ids.tobytes()
            input_block_checksum.update(block_bytes)
            block_file.write(block_bytes)
            block_counts.append(len(block_ids))
            session_codes.append(session_code)
            arrival_times.append(arrival_seconds)
            block_offset += len(block_ids)
            if len(block_ids):
                max_block_id = max(max_block_id, int(block_ids.max()))
            else:
                empty_requests += 1
            if len(block_counts) % 100000 == 0:
                write_json(
                    output_directory / "progress.json",
                    dict(
                        phase="parse",
                        requests=len(block_counts),
                        raw_rows=input_row + 1,
                        accesses=block_offset,
                        seconds=time.time() - started_at,
                    ),
                )
    block_counts = np.asarray(block_counts, dtype="<i4")
    session_codes = np.asarray(session_codes, dtype="<i4")
    arrival_times = np.asarray(arrival_times, dtype="<f8")
    assert len(block_counts) > 0
    saved_block_count = block_ids_path.stat().st_size // block_counts.itemsize
    assert saved_block_count == int(block_counts.sum(dtype=np.int64)) == block_offset

    input_summary = dict(
        source_sha256=trace_checksum.hexdigest(),
        requests=len(block_counts),
        sessions=len(session_codes_by_id),
        raw_rows=input_row + 1,
        skipped_other_models=skipped_requests,
        block_occurrences=int(block_offset),
        max_block_id=max_block_id,
        empty_requests=empty_requests,
        raw_ids_sha256=input_block_checksum.hexdigest(),
    )
    return InputOrderRequests(
        block_counts,
        session_codes,
        arrival_times,
        block_ids_path,
        input_summary,
    )


# 3. Sort whole requests by arrival time and save the simulator arrays.
def write_sorted_requests(requests, output_directory, started_at):
    """Sort requests by time and write their blocks and metadata in that order."""
    block_counts = requests.block_counts
    session_codes = requests.session_codes
    arrival_seconds = requests.arrival_seconds

    # Find the new request order. Equal timestamps keep their original order.
    arrival_order = np.argsort(arrival_seconds, kind="stable").astype("<i8")
    assert np.array_equal(np.sort(arrival_order), np.arange(len(arrival_order)))
    ordered_times_seconds = arrival_seconds[arrival_order]
    assert np.all(ordered_times_seconds[1:] >= ordered_times_seconds[:-1])
    tied_times = ordered_times_seconds[1:] == ordered_times_seconds[:-1]
    assert np.all(arrival_order[1:][tied_times] > arrival_order[:-1][tied_times])

    # Use the simulator's time unit and mark which requests belong to the test window.
    arrival_microseconds = np.rint(ordered_times_seconds * 1e6).astype("<i8")
    test_window_flags = (
        (arrival_microseconds >= TEST_START_US) & (arrival_microseconds < TEST_END_US)
    ).astype("u1")
    arrival_order.tofile(output_directory / ARRIVAL_ORDER_TO_INPUT_ROWS)

    # Find each request's block list in the joined file. If the first request has
    # three blocks, the second starts at offset 3.
    request_block_offsets = np.empty(len(block_counts) + 1, dtype="<i8")
    request_block_offsets[0] = 0
    np.cumsum(block_counts, dtype=np.int64, out=request_block_offsets[1:])
    block_ids_in_input_order = np.memmap(
        requests.input_order_block_ids_path, mode="r", dtype="<i4"
    )
    sorted_block_checksum = hashlib.sha256()
    ordered_prefix = output_directory / ARRIVAL_ORDER_PREFIX
    block_counts_path, session_numbers_path, arrival_times_path, block_ids_path = (
        trace_array_paths(ordered_prefix)
    )

    # Copy each request's full block list in arrival order.
    with block_ids_path.open("xb", buffering=8 * 1024**2) as block_file:
        for position, request_index in enumerate(arrival_order):
            block_bytes = block_ids_in_input_order[
                request_block_offsets[request_index] : request_block_offsets[request_index + 1]
            ].tobytes()
            sorted_block_checksum.update(block_bytes)
            block_file.write(block_bytes)
            if (position + 1) % 200000 == 0:
                write_json(
                    output_directory / "progress.json",
                    dict(
                        phase="stable_arrival_order",
                        requests=position + 1,
                        seconds=time.time() - started_at,
                    ),
                )
    block_counts[arrival_order].tofile(block_counts_path)
    session_codes[arrival_order].tofile(session_numbers_path)
    arrival_microseconds.tofile(arrival_times_path)

    hashes = {str(block_ids_path): sorted_block_checksum.hexdigest()}
    for path in (block_counts_path, session_numbers_path, arrival_times_path):
        hashes[str(path)] = file_fingerprint(path)
    return dict(
        arrival_prefix=str(ordered_prefix),
        data_sha256=hashes,
        arrival_min=float(ordered_times_seconds[0]),
        arrival_max=float(ordered_times_seconds[-1]),
        test_start_us=TEST_START_US,
        test_end_us=TEST_END_US,
        test_rows=int(test_window_flags.sum()),
        pre_window_rows=int(np.count_nonzero(arrival_microseconds < TEST_START_US)),
        post_window_rows=int(np.count_nonzero(arrival_microseconds >= TEST_END_US)),
    )


# 4. Connect the steps above; this is the entry point used by experiment scripts.
def prepare_trace(trace_path, output_directory):
    """Read one trace, sort its requests, and save checked simulator inputs."""
    started_at = time.time()
    output_directory.mkdir(exist_ok=False)
    source_before_read = trace_path.stat()
    write_json(
        output_directory / "status.json",
        dict(
            status="running",
            source=str(trace_path),
            started_unix=started_at,
        ),
    )

    # 1. Read the selected model's requests and give each session an integer code.
    requests = read_requests(trace_path, output_directory, started_at)
    source_after_read = trace_path.stat()
    assert (source_before_read.st_size, source_before_read.st_mtime_ns) == (
        source_after_read.st_size,
        source_after_read.st_mtime_ns,
    ), "Source changed while reading"

    # 2. Sort whole requests by arrival time and save the simulator's arrays.
    arrival_order_summary = write_sorted_requests(requests, output_directory, started_at)

    # 3. Record counts and file hashes so later stages can check their inputs.
    trace_manifest = dict(
        status="PASS: input preparation only",
        model_filter=MODEL,
        merging="none",
        source=str(trace_path),
        source_bytes=source_before_read.st_size,
        source_mtime_ns=source_before_read.st_mtime_ns,
        source_unchanged=True,
        **requests.input_summary,
        **arrival_order_summary,
        stable_permutation_verified=True,
        duplicates_preserved=True,
        seconds=time.time() - started_at,
    )
    write_json(output_directory / "data_manifest.json", trace_manifest)
    write_json(
        output_directory / "status.json",
        dict(status="complete", seconds=time.time() - started_at),
    )
    return trace_manifest


# Supporting definitions used by the preparation steps above.
# read_requests() returns this container; one position describes the same request
# in each array.
@dataclass
class InputOrderRequests:
    """Hold the request arrays in their original file order."""

    block_counts: np.ndarray
    session_codes: np.ndarray
    arrival_seconds: np.ndarray
    input_order_block_ids_path: Path
    input_summary: dict


# Name the four output files. This does not create or read them.
def trace_array_paths(stream_prefix):
    """Return the block count, session, time and block ID paths in that order."""
    return (
        Path(f"{stream_prefix}_block_counts.i32"),
        Path(f"{stream_prefix}_session_numbers.i32"),
        Path(f"{stream_prefix}_timestamps_us.i64"),
        Path(f"{stream_prefix}_block_ids.i32"),
    )


# Check file contents and write progress/summary records.
def file_fingerprint(path):
    """Calculate a file hash for later input checks."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, record):
    """Write a JSON record."""
    Path(path).write_text(json.dumps(record, indent=2) + "\n")
