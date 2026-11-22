"""Turn reference-replay label events into counts available at each hourly cutoff.

Assumes events and sample IDs were independently checked against the reference
layout. The function preserves strict-before-cutoff counting from the experiment.
"""

from pathlib import Path
import numpy as np


def build_hourly_saved_read_counts(
    timestamps, sample_indices, evictions, event_paths, destination
):
    """Accumulate sampled marginal gains only after their read has occurred."""
    arrival_times_us = np.memmap(timestamps, "<i8", "r")
    sampled_eviction_ids = np.load(sample_indices, mmap_mode="r", allow_pickle=False)
    eviction_records = np.memmap(evictions, "<i4", "r").reshape(-1, 6)
    row_of_sample = eviction_records[sampled_eviction_ids, 0]
    reads_unlocked = np.zeros(len(sampled_eviction_ids), dtype=np.uint64)
    streams = []
    for path in event_paths:
        events = np.memmap(path, "<i4", "r").reshape(-1, 3)
        if not np.all(events[1:, 0] >= events[:-1, 0]) or events[:, 2].min() <= 0:
            raise ValueError("Expected chronological positive marginal gains")
        streams.append(events)
    event_times_us = [arrival_times_us[events[:, 0]] for events in streams]
    next_event_by_stream = [0] * len(streams)
    # Row h contains only gains observed strictly before hour h starts.
    with Path(destination).open("xb") as label_file:
        for cutoff_us in np.arange(169, dtype=np.int64) * 3600000000:
            for stream_index, (events, observed_at) in enumerate(zip(streams, event_times_us)):
                stop_event = int(observed_at.searchsorted(cutoff_us, side="left"))
                batch = events[next_event_by_stream[stream_index] : stop_event]
                next_event_by_stream[stream_index] = stop_event
                if not np.all(row_of_sample[batch[:, 1]] < batch[:, 0]):
                    raise ValueError("A label read must follow its offer")
                np.add.at(reads_unlocked, batch[:, 1], batch[:, 2].astype(np.uint64))
            if reads_unlocked.max() > np.iinfo(np.uint32).max:
                raise ValueError("Observed count exceeds its file format")
            reads_unlocked.astype("<u4").tofile(label_file)
    if not all(
        events_consumed == len(events)
        for events_consumed, events in zip(next_event_by_stream, streams)
    ):
        raise ValueError("Unconsumed events after the final cutoff")
    return dict(shape=[169, len(sampled_eviction_ids)], sum_observed=int(reads_unlocked.sum()))
