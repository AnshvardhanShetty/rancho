"""Read the selected model's requests and save them in arrival order

parse_request checks the dictionary decoded from one JSONL line
read_requests saves the blocks and collects each request's time and session
write_sorted_requests sorts whole requests and marks days 4 to 7 for testing
prepare_trace runs those steps and saves a summary of the files and checks

Input times are seconds from the week start
Sorting keeps each request's block order and any repeated IDs
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

BASE = Path(__file__).resolve().parent
MODEL = "deepseek-v4-flash"
TEST_START_US = 3 * 86400 * 1_000_000  # Start of day 4, in microseconds.
TEST_END_US = 7 * 86400 * 1_000_000  # End of day 7, excluded.
MAX_SIGNED_INT32 = 2**31 - 1


# One container for all requests; position i in every array describes request i.
@dataclass
class Requests:
    """Hold the request arrays in their original file order."""

    block_counts: np.ndarray
    session_codes: np.ndarray
    arrival_seconds: np.ndarray
    raw_prefix: Path
    details: dict


# 1. Check one decoded request; skip other models and reject invalid values.
def parse_request(row):
    """Read the session, arrival time, and block IDs for the selected model."""
    if row.get("model_id") != MODEL:
        return None
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
            raise ValueError(
                "Block IDs do not fit nonnegative int32: change schemas explicitly"
            )
        block_array = block_array.astype("<i4", copy=False)
    else:
        block_array = np.empty(0, dtype="<i4")
    if len(block_array) > MAX_SIGNED_INT32:
        raise ValueError("Request length overflow")
    return session_id, float(arrival_seconds), block_array


# 2. Read the trace, number sessions, and save the input-order block lists.
def read_requests(trace_path, output_directory, reference_prefix, started_at):
    """Save block lists and collect each request's length, session, and time."""
    sessions = {}
    block_counts, session_codes, arrival_times = [], [], []
    block_offset = 0
    max_block_id = -1
    block_hash = hashlib.sha256()
    source_hash = hashlib.sha256()
    skipped = empty = 0

    # Compare with an existing conversion, or create a new block-ID file.
    if reference_prefix:
        reference_counts = np.fromfile(str(reference_prefix) + "_nb.i32", "<i4")
        reference_sessions = np.fromfile(str(reference_prefix) + "_sess.i32", "<i4")
        reference_times = np.fromfile(str(reference_prefix) + "_ts.f64", "<f8")
        reference_blocks = np.memmap(
            str(reference_prefix) + "_ids.i32", mode="r", dtype="<i4"
        )
        assert len(reference_counts) == len(reference_sessions) == len(reference_times)
        block_file = None
    else:
        block_file = (output_directory / "raw_ids.i32").open(
            "xb", buffering=8 * 1024**2
        )

    # Parse each JSON line, keep the selected model, and append its request data.
    try:
        with trace_path.open("rb") as trace:
            for raw_row, line in enumerate(trace):
                source_hash.update(line)
                request = parse_request(loads(line))
                if request is None:
                    skipped += 1
                    continue
                session_id, arrival_seconds, block_ids = request

                # Assign codes in first-seen order. Different sessions stay separate.
                if session_id not in sessions:
                    sessions[session_id] = len(sessions)
                session_code = sessions[session_id]
                request_index = len(block_counts)
                if session_code > MAX_SIGNED_INT32:
                    raise ValueError("Session index overflow")
                if request_index > MAX_SIGNED_INT32:
                    raise ValueError("Request indices require wider schema")

                if reference_prefix:
                    assert (
                        request_index < len(reference_counts)
                        and reference_counts[request_index] == len(block_ids)
                        and reference_sessions[request_index] == session_code
                        and reference_times[request_index] == arrival_seconds
                    ), ("metadata mismatch", request_index)
                    assert np.array_equal(
                        block_ids,
                        reference_blocks[block_offset : block_offset + len(block_ids)],
                    ), ("block sequence mismatch", request_index)

                # Join block lists into one file. Lengths let us recover each list.
                block_bytes = block_ids.tobytes()
                block_hash.update(block_bytes)
                if block_file:
                    block_file.write(block_bytes)
                block_counts.append(len(block_ids))
                session_codes.append(session_code)
                arrival_times.append(arrival_seconds)
                block_offset += len(block_ids)
                if len(block_ids):
                    max_block_id = max(max_block_id, int(block_ids.max()))
                else:
                    empty += 1
                if len(block_counts) % 100000 == 0:
                    write_json(
                        output_directory / "progress.json",
                        dict(
                            phase="parse",
                            requests=len(block_counts),
                            raw_rows=raw_row + 1,
                            accesses=block_offset,
                            seconds=time.time() - started_at,
                        ),
                    )
    finally:
        if block_file:
            block_file.close()

    block_counts = np.asarray(block_counts, dtype="<i4")
    session_codes = np.asarray(session_codes, dtype="<i4")
    arrival_times = np.asarray(arrival_times, dtype="<f8")
    assert len(block_counts) > 0
    if reference_prefix:
        assert len(block_counts) == len(reference_counts) and block_offset == len(
            reference_blocks
        ), "Existing stream length mismatch"
        raw_prefix = reference_prefix
    else:
        raw_prefix = output_directory / "raw"
        block_counts.tofile(output_directory / "raw_nb.i32")
        session_codes.tofile(output_directory / "raw_sess.i32")
        arrival_times.tofile(output_directory / "raw_ts.f64")
        saved_blocks = np.memmap(
            output_directory / "raw_ids.i32", mode="r", dtype="<i4"
        )
        assert (
            len(saved_blocks) == int(block_counts.sum(dtype=np.int64)) == block_offset
        )

    details = dict(
        source_sha256=source_hash.hexdigest(),
        requests=len(block_counts),
        sessions=len(sessions),
        raw_rows=raw_row + 1,
        skipped_other_models=skipped,
        block_occurrences=int(block_offset),
        max_block_id=max_block_id,
        empty_requests=empty,
        raw_ids_sha256=block_hash.hexdigest(),
    )
    return Requests(block_counts, session_codes, arrival_times, raw_prefix, details)


# 3. Sort whole requests by arrival time and save the simulator arrays.
def write_sorted_requests(requests, output_directory, reference_prefix, started_at):
    """Sort requests by time and write their blocks and metadata in that order."""
    block_counts = requests.block_counts
    session_codes = requests.session_codes
    arrival_seconds = requests.arrival_seconds

    # Equal timestamps keep their original file order.
    arrival_order = np.argsort(arrival_seconds, kind="stable").astype("<i8")
    assert np.array_equal(np.sort(arrival_order), np.arange(len(arrival_order)))
    ordered_times = arrival_seconds[arrival_order]
    assert np.all(ordered_times[1:] >= ordered_times[:-1])
    tied_times = ordered_times[1:] == ordered_times[:-1]
    assert np.all(arrival_order[1:][tied_times] > arrival_order[:-1][tied_times])
    arrival_microseconds = np.rint(ordered_times * 1e6).astype("<i8")
    test_requests = (
        (arrival_microseconds >= TEST_START_US) & (arrival_microseconds < TEST_END_US)
    ).astype("u1")
    arrival_order.tofile(output_directory / "perm_time.i64")
    test_requests.tofile(output_directory / "eligible_test.u8")

    # Request r occupies blocks[offsets[r]:offsets[r + 1]] in the raw file.
    offsets = np.empty(len(block_counts) + 1, dtype="<i8")
    offsets[0] = 0
    np.cumsum(block_counts, dtype=np.int64, out=offsets[1:])
    raw_blocks = np.memmap(str(requests.raw_prefix) + "_ids.i32", mode="r", dtype="<i4")
    ordered_hash = hashlib.sha256()
    if reference_prefix:
        ordered_prefix = reference_prefix.parent / "dst"
        assert np.array_equal(
            np.fromfile(str(ordered_prefix) + "_nb.i32", "<i4"),
            block_counts[arrival_order],
        )
        assert np.array_equal(
            np.fromfile(str(ordered_prefix) + "_sess.i32", "<i4"),
            session_codes[arrival_order],
        )
        assert np.array_equal(
            np.fromfile(str(ordered_prefix) + "_ts.i64", "<i8"), arrival_microseconds
        )
        block_file = None
    else:
        ordered_prefix = output_directory / "dst"
        block_file = (output_directory / "dst_ids.i32").open(
            "xb", buffering=8 * 1024**2
        )
    try:
        for position, request_index in enumerate(arrival_order):
            block_bytes = raw_blocks[
                offsets[request_index] : offsets[request_index + 1]
            ].tobytes()
            ordered_hash.update(block_bytes)
            if block_file:
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
    finally:
        if block_file:
            block_file.close()

    if reference_prefix:
        assert (
            file_fingerprint(str(ordered_prefix) + "_ids.i32")
            == ordered_hash.hexdigest()
        ), "Existing arrival stream differs"
    else:
        block_counts[arrival_order].tofile(output_directory / "dst_nb.i32")
        session_codes[arrival_order].tofile(output_directory / "dst_sess.i32")
        arrival_microseconds.tofile(output_directory / "dst_ts.i64")

    hashes = {str(ordered_prefix) + "_ids.i32": ordered_hash.hexdigest()}
    for suffix in ["_nb.i32", "_sess.i32", "_ts.i64"]:
        hashes[str(ordered_prefix) + suffix] = file_fingerprint(
            str(ordered_prefix) + suffix
        )
    hashes[str(output_directory / "eligible_test.u8")] = file_fingerprint(
        output_directory / "eligible_test.u8"
    )
    return dict(
        arrival_prefix=str(ordered_prefix),
        data_sha256=hashes,
        arrival_min=float(ordered_times[0]),
        arrival_max=float(ordered_times[-1]),
        test_start_us=TEST_START_US,
        test_end_us=TEST_END_US,
        test_rows=int(test_requests.sum()),
        pre_window_rows=int(np.count_nonzero(arrival_microseconds < TEST_START_US)),
        post_window_rows=int(np.count_nonzero(arrival_microseconds >= TEST_END_US)),
    )


# 4. Connect the steps above; this is the entry point used by experiment scripts.
def prepare_trace(trace_path, output_directory, existing=None):
    """Read one trace, sort its requests, and save checked simulator inputs.

    With existing set, compare against an earlier conversion of the same trace.
    """
    started_at = time.time()
    output_directory.mkdir(exist_ok=False)
    before = trace_path.stat()
    write_json(
        output_directory / "status.json",
        dict(
            status="running",
            source=str(trace_path),
            started_unix=started_at,
        ),
    )

    # 1. Read the selected model's requests and give each session an integer code.
    requests = read_requests(trace_path, output_directory, existing, started_at)
    after = trace_path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), "Source changed while reading"

    # 2. Sort whole requests by arrival time and save the simulator's arrays.
    ordered = write_sorted_requests(requests, output_directory, existing, started_at)

    # 3. Record counts and file hashes so later stages can check their inputs.
    result = dict(
        status="PASS: input preparation only",
        model_filter=MODEL,
        merging="none",
        source=str(trace_path),
        source_bytes=before.st_size,
        source_mtime_ns=before.st_mtime_ns,
        source_unchanged=True,
        **requests.details,
        raw_prefix=str(requests.raw_prefix),
        **ordered,
        stable_permutation_verified=True,
        duplicates_preserved=True,
        matched_existing_week1_fully=bool(existing),
        seconds=time.time() - started_at,
        caveat="No capacity, HBM eviction stream, objective vector, policy or upper bound has been generated.",
    )
    write_json(output_directory / "data_manifest.json", result)
    write_json(
        output_directory / "status.json",
        dict(status="complete", seconds=time.time() - started_at),
    )
    return result


# File helpers: check file contents and write progress/summary records.
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


# Optional self-test on a small artificial trace.
def test_trace_reader():
    """Check ordering, duplicate blocks, test boundaries, and invalid inputs."""
    test_directory = BASE / "selftest"
    test_directory.mkdir(exist_ok=False)
    rows = [
        dict(
            model_id=MODEL,
            session_id="A",
            timestamp_arrival=259200.5,
            hash_ids=[2, 1, 2],
        ),
        dict(model_id="different", session_id="X", timestamp_arrival=1, hash_ids=[999]),
        dict(model_id=MODEL, session_id="B", timestamp_arrival=-1.5, hash_ids=[0]),
        dict(model_id=MODEL, session_id="A", timestamp_arrival=259200.5, hash_ids=[1]),
        dict(model_id=MODEL, session_id="C", timestamp_arrival=604800, hash_ids=[]),
    ]
    trace_path = test_directory / "fixture.jsonl"
    trace_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = prepare_trace(trace_path, test_directory / "output")
    assert (
        manifest["requests"] == 4
        and manifest["block_occurrences"] == 5
        and manifest["test_rows"] == 2
    )
    assert np.array_equal(
        np.fromfile(test_directory / "output/dst_ids.i32", "<i4"), [0, 2, 1, 2, 1]
    )
    assert np.array_equal(
        np.fromfile(test_directory / "output/perm_time.i64", "<i8"), [1, 0, 2, 3]
    )
    for invalid_block_ids in [[-1], [2**31], [1.5]]:
        row = dict(rows[0], hash_ids=invalid_block_ids)
        try:
            parse_request(row)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed IDs accepted")
    write_json(
        test_directory / "certificate.json",
        dict(
            status="PASS",
            tests=[
                "exact filter",
                "JSON whitespace",
                "stable ties",
                "duplicate preservation",
                "window edges",
                "invalid ID rejection",
            ],
        ),
    )
    print("SELFTEST PASS", flush=True)
