"""Count eligible reads restored from SSD, after the GPU-only readable prefix.

Assumes the verified same-session bit index and per-request rows supplied by the
caller. Counts are block positions, not measured GPU time. No private bits ship here.
"""

from pathlib import Path
import numpy as np


def credit_and_overlap(path, reference, credit_directory):
    """Return eligible reads, SSD savings, and gains/losses against a reference."""
    credit_path = Path(credit_directory)
    rows = np.memmap(path, "<i8", "r").reshape(-1, 13)
    reference_rows = np.memmap(reference, "<i8", "r").reshape(-1, 13)
    assert np.array_equal(rows[:, 0], reference_rows[:, 0]) and np.array_equal(
        rows[:, 3], reference_rows[:, 3]
    )
    request_offsets = np.memmap(credit_path / "offsets.i64", "<u8", "r")[
        rows[:, 0]
    ].astype(np.int64)
    eligible_bits = np.memmap(credit_path / "credit.bits", "<u8", "r")
    eligible_prefix_counts = np.memmap(credit_path / "credit.rank", "<u8", "r")
    byte_popcounts = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

    def eligible_before(prefix_positions):
        positions = (request_offsets + prefix_positions).astype("uint64")
        word_indices, bit_offsets = np.divmod(positions, np.uint64(64))
        partial_word = eligible_bits[word_indices] & (
            np.uint64(1) << bit_offsets
        ) - np.uint64(1)
        return (
            eligible_prefix_counts[word_indices]
            + byte_popcounts[partial_word.view("u1").reshape(-1, 8)].sum(axis=1)
        ).astype(np.int64)

    hbm_credit = eligible_before(rows[:, 3])
    combined_credit = eligible_before(rows[:, 4])
    reference_credit = eligible_before(reference_rows[:, 4])
    return (
        int((eligible_before(rows[:, 2]) - hbm_credit).sum()),
        int((combined_credit - hbm_credit).sum()),
        int(np.maximum(combined_credit - reference_credit, 0).sum()),
        int(np.maximum(reference_credit - combined_credit, 0).sum()),
    )
