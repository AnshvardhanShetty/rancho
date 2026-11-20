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
LO = 3 * 86400 * 1_000_000  # Start of day 4, in microseconds.
HI = 7 * 86400 * 1_000_000  # End of day 7, excluded.
I32MAX = 2**31 - 1


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
            or block_array.max() > I32MAX
        ):
            raise ValueError(
                "Block IDs do not fit nonnegative int32: change schemas explicitly"
            )
        block_array = block_array.astype("<i4", copy=False)
    else:
        block_array = np.empty(0, dtype="<i4")
    if len(block_array) > I32MAX:
        raise ValueError("Request length overflow")
    return session_id, float(arrival_seconds), block_array


# 2. Read the trace, number sessions, and save the input-order block lists.
def read_requests(source, out, existing, started):
    """Save block lists and collect each request's length, session, and time."""
    sessions = {}
    block_counts, session_codes, arrival_times = [], [], []
    block_offset = 0
    max_block_id = -1
    block_hash = hashlib.sha256()
    source_hash = hashlib.sha256()
    skipped = empty = 0

    # Compare with an existing conversion, or create a new block-ID file.
    if existing:
        reference_counts = np.fromfile(str(existing) + "_nb.i32", "<i4")
        reference_sessions = np.fromfile(str(existing) + "_sess.i32", "<i4")
        reference_times = np.fromfile(str(existing) + "_ts.f64", "<f8")
        reference_blocks = np.memmap(str(existing) + "_ids.i32", mode="r", dtype="<i4")
        assert len(reference_counts) == len(reference_sessions) == len(reference_times)
        block_file = None
    else:
        block_file = (out / "raw_ids.i32").open("xb", buffering=8 * 1024**2)

    # Parse each JSON line, keep the selected model, and append its request data.
    try:
        with source.open("rb") as trace:
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
                if session_code > I32MAX:
                    raise ValueError("Session index overflow")
                if request_index > I32MAX:
                    raise ValueError("Request indices require wider schema")

                if existing:
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
                        out / "progress.json",
                        dict(
                            phase="parse",
                            requests=len(block_counts),
                            raw_rows=raw_row + 1,
                            accesses=block_offset,
                            seconds=time.time() - started,
                        ),
                    )
    finally:
        if block_file:
            block_file.close()

    block_counts = np.asarray(block_counts, dtype="<i4")
    session_codes = np.asarray(session_codes, dtype="<i4")
    arrival_times = np.asarray(arrival_times, dtype="<f8")
    assert len(block_counts) > 0
    if existing:
        assert len(block_counts) == len(reference_counts) and block_offset == len(
            reference_blocks
        ), "Existing stream length mismatch"
        raw_prefix = existing
    else:
        raw_prefix = out / "raw"
        block_counts.tofile(out / "raw_nb.i32")
        session_codes.tofile(out / "raw_sess.i32")
        arrival_times.tofile(out / "raw_ts.f64")
        saved_blocks = np.memmap(out / "raw_ids.i32", mode="r", dtype="<i4")
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
def write_sorted_requests(requests, out, existing, started):
    """Sort requests by time and write their blocks and metadata in that order."""
    block_counts = requests.block_counts
    session_codes = requests.session_codes
    arrival_seconds = requests.arrival_seconds

    # Equal timestamps keep their original file order.
    order = np.argsort(arrival_seconds, kind="stable").astype("<i8")
    assert np.array_equal(np.sort(order), np.arange(len(order)))
    ordered_times = arrival_seconds[order]
    assert np.all(ordered_times[1:] >= ordered_times[:-1])
    tied_times = ordered_times[1:] == ordered_times[:-1]
    assert np.all(order[1:][tied_times] > order[:-1][tied_times])
    arrival_microseconds = np.rint(ordered_times * 1e6).astype("<i8")
    test_requests = ((arrival_microseconds >= LO) & (arrival_microseconds < HI)).astype(
        "u1"
    )
    order.tofile(out / "perm_time.i64")
    test_requests.tofile(out / "eligible_test.u8")

    # Request r occupies blocks[offsets[r]:offsets[r + 1]] in the raw file.
    offsets = np.empty(len(block_counts) + 1, dtype="<i8")
    offsets[0] = 0
    np.cumsum(block_counts, dtype=np.int64, out=offsets[1:])
    raw_blocks = np.memmap(str(requests.raw_prefix) + "_ids.i32", mode="r", dtype="<i4")
    ordered_hash = hashlib.sha256()
    if existing:
        ordered_prefix = existing.parent / "dst"
        assert np.array_equal(
            np.fromfile(str(ordered_prefix) + "_nb.i32", "<i4"), block_counts[order]
        )
        assert np.array_equal(
            np.fromfile(str(ordered_prefix) + "_sess.i32", "<i4"), session_codes[order]
        )
        assert np.array_equal(
            np.fromfile(str(ordered_prefix) + "_ts.i64", "<i8"), arrival_microseconds
        )
        block_file = None
    else:
        ordered_prefix = out / "dst"
        block_file = (out / "dst_ids.i32").open("xb", buffering=8 * 1024**2)
    try:
        for position, request_index in enumerate(order):
            block_bytes = raw_blocks[
                offsets[request_index] : offsets[request_index + 1]
            ].tobytes()
            ordered_hash.update(block_bytes)
            if block_file:
                block_file.write(block_bytes)
            if (position + 1) % 200000 == 0:
                write_json(
                    out / "progress.json",
                    dict(
                        phase="stable_arrival_order",
                        requests=position + 1,
                        seconds=time.time() - started,
                    ),
                )
    finally:
        if block_file:
            block_file.close()

    if existing:
        assert (
            file_fingerprint(str(ordered_prefix) + "_ids.i32")
            == ordered_hash.hexdigest()
        ), "Existing arrival stream differs"
    else:
        block_counts[order].tofile(out / "dst_nb.i32")
        session_codes[order].tofile(out / "dst_sess.i32")
        arrival_microseconds.tofile(out / "dst_ts.i64")

    hashes = {str(ordered_prefix) + "_ids.i32": ordered_hash.hexdigest()}
    for suffix in ["_nb.i32", "_sess.i32", "_ts.i64"]:
        hashes[str(ordered_prefix) + suffix] = file_fingerprint(
            str(ordered_prefix) + suffix
        )
    hashes[str(out / "eligible_test.u8")] = file_fingerprint(out / "eligible_test.u8")
    return dict(
        arrival_prefix=str(ordered_prefix),
        data_sha256=hashes,
        arrival_min=float(ordered_times[0]),
        arrival_max=float(ordered_times[-1]),
        test_start_us=LO,
        test_end_us=HI,
        test_rows=int(test_requests.sum()),
        pre_window_rows=int(np.count_nonzero(arrival_microseconds < LO)),
        post_window_rows=int(np.count_nonzero(arrival_microseconds >= HI)),
    )


# 4. Connect the steps above; this is the entry point used by experiment scripts.
def prepare_trace(source, out, existing=None):
    """Read one trace, sort its requests, and save checked simulator inputs.

    With existing set, compare against an earlier conversion of the same trace.
    """
    started = time.time()
    out.mkdir(exist_ok=False)
    before = source.stat()
    write_json(
        out / "status.json",
        dict(
            status="running",
            source=str(source),
            started_unix=started,
        ),
    )

    # 1. Read the selected model's requests and give each session an integer code.
    requests = read_requests(source, out, existing, started)
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), "Source changed while reading"

    # 2. Sort whole requests by arrival time and save the simulator's arrays.
    ordered = write_sorted_requests(requests, out, existing, started)

    # 3. Record counts and file hashes so later stages can check their inputs.
    result = dict(
        status="PASS: input preparation only",
        model_filter=MODEL,
        merging="none",
        source=str(source),
        source_bytes=before.st_size,
        source_mtime_ns=before.st_mtime_ns,
        source_unchanged=True,
        **requests.details,
        raw_prefix=str(requests.raw_prefix),
        **ordered,
        stable_permutation_verified=True,
        duplicates_preserved=True,
        matched_existing_week1_fully=bool(existing),
        seconds=time.time() - started,
        caveat="No capacity, HBM eviction stream, objective vector, policy or upper bound has been generated.",
    )
    write_json(out / "data_manifest.json", result)
    write_json(
        out / "status.json", dict(status="complete", seconds=time.time() - started)
    )
    return result


# File helpers: check file contents and write progress/summary records.
def file_fingerprint(path):
    """Calculate a file hash for later input checks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    """Write a JSON record."""
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


# Optional self-test on a small artificial trace.
def test_trace_reader():
    """Check ordering, duplicate blocks, test boundaries, and invalid inputs."""
    p = BASE / "selftest"
    p.mkdir(exist_ok=False)
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
    src = p / "fixture.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    z = prepare_trace(src, p / "output")
    assert z["requests"] == 4 and z["block_occurrences"] == 5 and z["test_rows"] == 2
    assert np.array_equal(np.fromfile(p / "output/dst_ids.i32", "<i4"), [0, 2, 1, 2, 1])
    assert np.array_equal(np.fromfile(p / "output/perm_time.i64", "<i8"), [1, 0, 2, 3])
    for bad in [[-1], [2**31], [1.5]]:
        row = dict(rows[0], hash_ids=bad)
        try:
            parse_request(row)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed IDs accepted")
    write_json(
        p / "certificate.json",
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
