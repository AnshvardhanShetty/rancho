"""Prepare the eviction examples used to fit each hourly model.

TrainingData loads checked files and keeps their rows aligned.
find_first_eviction_at_or_after() finds the requests belonging to an hourly fit interval.
training_examples() selects past evictions and their gains observed by the cutoff.
feature_table() builds the eight inputs; categorical_columns() identifies user IDs.

Assumes chronological prepared arrays and checked reference-label counts.
Private arrays are not distributed. Later label rows cannot enter an earlier fit.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from .settings import HOUR, TRAINING

EVICTION_RECORD_DTYPE = np.dtype(
    [(x, "<i4") for x in ("row", "block", "last", "freq", "nev", "born")]
)


def resolve_and_verify_input_file(record, parent):
    """Resolve a caller-supplied file and check its complete contents."""
    path = Path(record["path"])
    if not path.is_absolute():
        path = parent / path
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != record["sha256"]:
            raise ValueError("Input digest differs: " + str(path))
    return path


def verify_file_hash(path, digest):
    resolve_and_verify_input_file(dict(path=str(path), sha256=digest), Path("."))


def validation_start_time(cutoff, cohort_lag_hours, validation_hours=6):
    """Leave a real future validation interval, plus label-maturation delay."""
    return cutoff - (cohort_lag_hours + validation_hours) * HOUR


def select_validation_rows(decision_times, cutoff, inner_cutoff, cohort_lag_hours):
    times = np.asarray(decision_times)
    return (times >= inner_cutoff) & (times + cohort_lag_hours * HOUR <= cutoff)


class TrainingData:
    """Keep training samples, request features and observed labels aligned."""

    def __init__(self, manifest_path, cold=False):
        """Load the eviction records and reuse counts used by the hourly models"""
        manifest_path = Path(manifest_path).resolve()
        manifest_bytes = manifest_path.read_bytes()
        self.identity = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = json.loads(manifest_bytes)
        manifest = self.manifest
        self.schema = manifest["feature_schema"]
        files = manifest["files"]
        paths = {
            key: resolve_and_verify_input_file(files[key], manifest_path.parent)
            for key in ("timestamps", "evictions", "features", "sample_indices")
        }
        self.arrival_times_us = np.memmap(paths["timestamps"], "<i8", "r")
        self.eviction_records = np.memmap(
            paths["evictions"], EVICTION_RECORD_DTYPE, "r"
        )
        if len(self.eviction_records) != manifest["eviction_count"]:
            raise ValueError("Eviction count differs")
        if cold:
            self.eviction_records = self.eviction_records[manifest["ntrain"] :]
        self.eviction_count = len(self.eviction_records)
        self.labels = {}
        with np.load(paths["features"], allow_pickle=False) as archive:
            self.features = {name: archive[name] for name in archive.files}
        self.sampled_eviction_ids = np.load(paths["sample_indices"], allow_pickle=False)
        if not cold:
            self.sample_times = self.arrival_times_us[
                self.eviction_records["row"][self.sampled_eviction_ids]
            ]
            if np.any(np.diff(self.sampled_eviction_ids) <= 0) or np.any(
                np.diff(self.sample_times) < 0
            ):
                raise ValueError("Samples must be distinct and chronological")
        labels = dict(manifest["cumulative_labels"])
        labels["path"] = str(
            resolve_and_verify_input_file(labels, manifest_path.parent)
        )
        self.manifest = dict(manifest, cumulative_labels=labels)

    def find_first_eviction_at_or_after(self, cutoff):
        """Find the first eviction whose request arrived at or after this time."""
        request_row = int(np.searchsorted(self.arrival_times_us, cutoff, side="left"))
        return int(
            np.searchsorted(self.eviction_records["row"], request_row, side="left")
        )

    def decision_times(self, indices):
        return np.asarray(self.arrival_times_us[self.eviction_records["row"][indices]])

    def user_ids(self, indices):
        return np.asarray(self.features["user"][self.eviction_records["last"][indices]])

    def training_examples(self, cutoff, config):
        """Return earlier eviction IDs, observed marginal gains and sample weights."""
        profile = next(
            (
                training_profile
                for training_profile in TRAINING
                if training_profile["id"] == config["training"]
            )
        )
        if config["target_mode"] != "cumulative":
            raise ValueError("Expected recorded cumulative marginal labels")
        first_sample = int(
            np.searchsorted(
                self.sample_times,
                (
                    max(0, cutoff - profile["window_hours"] * HOUR)
                    if profile["window_hours"]
                    else 0
                ),
                side="left",
            )
        )
        stop_sample = max(
            first_sample, int(np.searchsorted(self.sample_times, cutoff, side="left"))
        )
        first_sample = max(first_sample, stop_sample - config["sample_cap"])
        eviction_ids = np.asarray(
            self.sampled_eviction_ids[first_sample:stop_sample], dtype=np.int64
        )
        if not len(eviction_ids):
            return (eviction_ids, np.empty(0), np.empty(0))
        if cutoff % HOUR or cutoff < 0:
            raise ValueError("Exact hourly observed cutoff required")
        label_file = self.manifest["cumulative_labels"]
        if not hasattr(self, "observed_matrix"):
            verify_file_hash(label_file["path"], label_file["sha256"])
            self.observed_matrix = np.memmap(
                label_file["path"],
                "<u4",
                "r",
                shape=(169, len(self.sampled_eviction_ids)),
            )
        observed_gains = np.asarray(
            self.observed_matrix[cutoff // HOUR, first_sample:stop_sample],
            dtype=np.float64,
        )
        if np.any(self.sample_times[first_sample:stop_sample] >= cutoff):
            raise ValueError("Nonpast cumulative example")
        weights = np.ones(len(eviction_ids), dtype=np.float64)
        if profile["sample_weight"] == "recency24h":
            weights = np.exp2(
                -(cutoff - self.sample_times[first_sample:stop_sample]) / (24 * HOUR)
            )
            weights /= weights.mean()
        return (eviction_ids, observed_gains, weights)

    def categorical_columns(self, config):
        """Tell the tree which columns represent identities rather than quantities."""
        return [
            i
            for i, name in enumerate(config["features"])
            if self.schema[name]["encoding"] == "categorical"
        ]

    def feature_table(self, indices, config):
        """Build request inputs from the last touch and block inputs from the eviction."""
        eviction_ids = np.asarray(indices, dtype=np.int64)
        if np.any(eviction_ids < 0) or np.any(eviction_ids >= self.eviction_count):
            raise ValueError("Scoring request references invalid event IDs")
        eviction_records = self.eviction_records[eviction_ids]
        last_touch_rows, eviction_times_us = (
            eviction_records["last"],
            self.arrival_times_us[eviction_records["row"]],
        )
        if np.any(last_touch_rows > eviction_records["row"]) or np.any(
            eviction_records["born"] > eviction_records["row"]
        ):
            raise ValueError("Noncausal eviction metadata")
        feature_columns = []
        for name in config["features"]:
            if name == "prior_touch_count":
                values = eviction_records["freq"]
            elif name == "seconds_since_last_touch":
                values = (
                    eviction_times_us - self.arrival_times_us[last_touch_rows]
                ) / 1000000.0
            elif name == "seconds_in_HBM":
                values = (
                    eviction_times_us - self.arrival_times_us[eviction_records["born"]]
                ) / 1000000.0
            elif name == "prior_eviction_count":
                values = eviction_records["nev"]
            else:
                values = self.features[name][last_touch_rows]
            feature_columns.append(values)
        return (
            np.column_stack(feature_columns).astype(np.float64)
            if feature_columns
            else np.empty((len(eviction_ids), 0))
        )
