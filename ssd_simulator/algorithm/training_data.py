"""Prepare the eviction examples used to fit each hourly model.

TrainingData loads checked files and keeps their rows aligned.
first_eviction_at() finds the requests belonging to an hourly fit interval.
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

EDT = np.dtype([(x, "<i4") for x in ("row", "block", "last", "freq", "nev", "born")])


def pinned(record, parent):
    """Resolve a caller-supplied file and check its complete contents."""
    path = Path(record["path"])
    if not path.is_absolute():
        path = parent / path
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != record["sha256"]:
            raise ValueError("Input digest differs: " + str(path))
    return path


def verify_digest(path, digest):
    pinned(dict(path=str(path), sha256=digest), Path("."))


def chronological_inner_cut(cutoff, cohort_lag_hours, validation_hours=6):
    """Leave a real future validation interval, plus label-maturation delay."""
    return cutoff - (cohort_lag_hours + validation_hours) * HOUR


def validation_mask(decision_times, cutoff, inner_cutoff, cohort_lag_hours):
    times = np.asarray(decision_times)
    return (times >= inner_cutoff) & (times + cohort_lag_hours * HOUR <= cutoff)


class TrainingData:
    """Keep training samples, request features and observed labels aligned."""

    def __init__(self, manifest_path, cold=False):
        manifest_path = Path(manifest_path).resolve()
        raw = manifest_path.read_bytes()
        self.identity = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        manifest = self.manifest
        self.schema = manifest["feature_schema"]
        files = manifest["files"]
        paths = {
            key: pinned(files[key], manifest_path.parent)
            for key in ("timestamps", "evictions", "features", "sample_indices")
        }
        # ts holds request arrival times; es has one record per GPU eviction.
        self.ts = np.memmap(paths["timestamps"], "<i8", "r")
        self.es = np.memmap(paths["evictions"], EDT, "r")
        if len(self.es) != manifest["eviction_count"]:
            raise ValueError("Eviction count differs")
        if cold:
            self.es = self.es[manifest["ntrain"] :]
        self.n = len(self.es)
        self.labels = {}
        with np.load(paths["features"], allow_pickle=False) as archive:
            self.features = {name: archive[name] for name in archive.files}
        self.sample = np.load(paths["sample_indices"], allow_pickle=False)
        if not cold:
            self.sample_times = self.ts[self.es["row"][self.sample]]
            if np.any(np.diff(self.sample) <= 0) or np.any(
                np.diff(self.sample_times) < 0
            ):
                raise ValueError("Samples must be distinct and chronological")
        labels = dict(manifest["cumulative_labels"])
        labels["path"] = str(pinned(labels, manifest_path.parent))
        self.manifest = dict(manifest, cumulative_labels=labels)

    def first_eviction_at(self, cutoff):
        """Find the first eviction whose request arrived at or after this time."""
        row = int(np.searchsorted(self.ts, cutoff, side="left"))
        return int(np.searchsorted(self.es["row"], row, side="left"))

    def decision_times(self, indices):
        return np.asarray(self.ts[self.es["row"][indices]])

    def user_ids(self, indices):
        return np.asarray(self.features["user"][self.es["last"][indices]])

    def training_examples(self, cutoff, config):
        """Return earlier eviction IDs, observed marginal gains and sample weights."""
        profile = next((x for x in TRAINING if x["id"] == config["training"]))
        if config["target_mode"] != "cumulative":
            raise ValueError("Expected recorded cumulative marginal labels")
        a = int(
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
        b = max(a, int(np.searchsorted(self.sample_times, cutoff, side="left")))
        a = max(a, b - config["sample_cap"])
        ids = np.asarray(self.sample[a:b], dtype=np.int64)
        if not len(ids):
            return (ids, np.empty(0), np.empty(0))
        if cutoff % HOUR or cutoff < 0:
            raise ValueError("Exact hourly observed cutoff required")
        q = self.manifest["cumulative_labels"]
        if not hasattr(self, "observed_matrix"):
            verify_digest(q["path"], q["sha256"])
            self.observed_matrix = np.memmap(
                q["path"], "<u4", "r", shape=(169, len(self.sample))
            )
        y = np.asarray(self.observed_matrix[cutoff // HOUR, a:b], dtype=np.float64)
        if np.any(self.sample_times[a:b] >= cutoff):
            raise ValueError("Nonpast cumulative example")
        weights = np.ones(len(ids), dtype=np.float64)
        if profile["sample_weight"] == "recency24h":
            weights = np.exp2(-(cutoff - self.sample_times[a:b]) / (24 * HOUR))
            weights /= weights.mean()
        return (ids, y, weights)

    def categorical_columns(self, config):
        """Tell the tree which columns represent identities rather than quantities."""
        return [
            i
            for i, name in enumerate(config["features"])
            if self.schema[name]["encoding"] == "categorical"
        ]

    def feature_table(self, indices, config):
        """Build request inputs from the last touch and block inputs from the eviction."""
        ids = np.asarray(indices, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.n):
            raise ValueError("Scoring request references invalid event IDs")
        e = self.es[ids]
        rows, now = (e["last"], self.ts[e["row"]])
        if np.any(rows > e["row"]) or np.any(e["born"] > e["row"]):
            raise ValueError("Noncausal eviction metadata")
        cols = []
        for name in config["features"]:
            if name == "prior_touch_count":
                values = e["freq"]
            elif name == "seconds_since_last_touch":
                values = (now - self.ts[rows]) / 1000000.0
            elif name == "seconds_in_HBM":
                values = (now - self.ts[e["born"]]) / 1000000.0
            elif name == "prior_eviction_count":
                values = e["nev"]
            else:
                values = self.features[name][rows]
            cols.append(values)
        return (
            np.column_stack(cols).astype(np.float64)
            if cols
            else np.empty((len(ids), 0))
        )
