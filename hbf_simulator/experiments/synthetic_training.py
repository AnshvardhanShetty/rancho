"""Build reuse labels and train RANCHO on earlier synthetic requests.

An initial hourly predictor supplies scores for a replay that records offered
suffixes. The final model learns from later accesses to those suffixes.
Fitting and label cutoffs use model_training.py."""

from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hbf_simulator/model"))
import model_training


class SyntheticTrainingData(model_training.RequestData):
    """Supply synthetic arrays to the research training and label-selection code."""

    def __init__(
        self, request_blocks, arrival_times_us, sessions, features, names, offered=None
    ):
        self.arrival_times_us = np.asarray(arrival_times_us, dtype=np.int64)
        self.request_count = len(arrival_times_us)
        self.features = {
            name: features[:, request_index] for request_index, name in enumerate(names)
        }
        self.feature_schema = {
            name: {"encoding": "categorical" if request_index == 0 else "numeric"}
            for request_index, name in enumerate(names)
        }
        self.offer_block_counts = np.zeros(self.request_count, dtype=np.int32)
        self.hourly_read_counts = np.zeros((168, self.request_count), dtype=np.uint64)
        self.counted_until_hour = 0
        self.observed_read_totals = np.zeros(self.request_count, dtype=np.int64)
        self.label_kind = (
            "offered_suffix" if offered is not None else "original_increment"
        )
        if (
            len(request_blocks) != self.request_count
            or len(sessions) != self.request_count
            or len(features) != self.request_count
        ):
            raise ValueError("Training arrays describe different request counts")
        if np.any(self.arrival_times_us < 0) or np.any(
            self.arrival_times_us[1:] < self.arrival_times_us[:-1]
        ):
            raise ValueError("Training requests must be chronological and nonnegative")
        last_session = {}
        earlier_offers_by_block = {}
        introducing_request = {}
        for request_index, blocks in enumerate(request_blocks):
            session = sessions[request_index]
            hour = int(self.arrival_times_us[request_index] // model_training.H)
            if hour >= len(self.hourly_read_counts):
                raise ValueError("Request is beyond the supported training window")
            distinct_request_blocks = list(dict.fromkeys(blocks))
            if offered is None:
                # Same ownership rule as initial_reuse_counts.c: a change of
                # session starts a new introducing request for that block.
                for block in distinct_request_blocks:
                    if last_session.get(block) != session:
                        introducing_request[block] = request_index
                        self.offer_block_counts[request_index] += 1
                    else:
                        self.hourly_read_counts[hour, introducing_request[block]] += 1
                    last_session[block] = session
                continue
            for block in distinct_request_blocks:
                if last_session.get(block) == session:
                    for offer_request_index in earlier_offers_by_block.get(block, []):
                        self.hourly_read_counts[hour, offer_request_index] += 1
                last_session[block] = session
            distinct_offered_blocks = list(dict.fromkeys(offered[request_index]))
            if not set(distinct_offered_blocks).issubset(distinct_request_blocks):
                raise ValueError("Training offer contains blocks outside its request")
            self.offer_block_counts[request_index] = len(distinct_offered_blocks)
            for block in distinct_offered_blocks:
                earlier_offers_by_block.setdefault(block, []).append(request_index)


def fit_research_model(data, cut, settings):
    """Call the research fit, silencing only LightGBM's renamed-argument notice."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The argument 'eval_set' is deprecated.*",
            category=FutureWarning,
            module="lightgbm.sklearn",
        )
        return model_training.fit_tree_model(data, cut, settings)


def initial_predictions(data, settings):
    """Fit using completed hours, then predict requests in the following hour."""
    predictions = np.zeros(data.request_count, dtype=float)
    hours = np.unique(data.arrival_times_us // model_training.H)
    for hour in hours:
        cutoff_us = int(hour * model_training.H)
        earlier_requests, _ = data.training_examples(cutoff_us)
        if not len(earlier_requests):
            continue
        model, _ = fit_research_model(data, cutoff_us, settings)
        requests_this_hour = np.flatnonzero(
            data.arrival_times_us // model_training.H == hour
        )
        features = data.feature_table(requests_this_hour, settings["features"])
        predictions[requests_this_hour] = model_training.predict_reuse(model, features)
    return predictions
