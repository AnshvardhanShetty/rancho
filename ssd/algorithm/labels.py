"""Turn reference-replay label events into counts available at each hourly cutoff.

Assumes events and sample IDs were independently checked against the reference
layout. The function preserves strict-before-cutoff counting from the experiment.
"""

from pathlib import Path
import numpy as np


def build_observed_counts(
    timestamps, sample_indices, evictions, event_paths, destination
):
    """Accumulate sampled marginal gains only after their read has occurred."""
    ts = np.memmap(timestamps, "<i8", "r")
    sample = np.load(sample_indices, mmap_mode="r", allow_pickle=False)
    ev = np.memmap(evictions, "<i4", "r").reshape(-1, 6)
    row_of_sample = ev[sample, 0]
    total = np.zeros(len(sample), dtype=np.uint64)
    streams = []
    for path in event_paths:
        events = np.memmap(path, "<i4", "r").reshape(-1, 3)
        if not np.all(events[1:, 0] >= events[:-1, 0]) or events[:, 2].min() <= 0:
            raise ValueError("Expected chronological positive marginal gains")
        streams.append(events)
    times = [ts[events[:, 0]] for events in streams]
    cursor = [0] * len(streams)
    # Row h contains only gains observed strictly before hour h starts.
    with Path(destination).open("xb") as out:
        for cut in np.arange(169, dtype=np.int64) * 3600000000:
            for i, (events, observed_at) in enumerate(zip(streams, times)):
                end = int(observed_at.searchsorted(cut, side="left"))
                batch = events[cursor[i] : end]
                cursor[i] = end
                if not np.all(row_of_sample[batch[:, 1]] < batch[:, 0]):
                    raise ValueError("A label read must follow its offer")
                np.add.at(total, batch[:, 1], batch[:, 2].astype(np.uint64))
            if total.max() > np.iinfo(np.uint32).max:
                raise ValueError("Observed count exceeds its file format")
            total.astype("<u4").tofile(out)
    if not all(n == len(events) for n, events in zip(cursor, streams)):
        raise ValueError("Unconsumed events after the final cutoff")
    return dict(shape=[169, len(sample)], sum_observed=int(total.sum()))
