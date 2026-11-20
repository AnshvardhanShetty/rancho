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

    def __init__(self, rows, times, sessions, features, names, offered=None):
        self.ts = np.asarray(times, dtype=np.int64)
        self.n = len(times)
        self.features = {name: features[:, i] for i, name in enumerate(names)}
        self.schema = {
            name: {"encoding": "categorical" if i == 0 else "numeric"}
            for i, name in enumerate(names)
        }
        self.c = np.zeros(self.n, dtype=np.int32)
        self.Y = np.zeros((168, self.n), dtype=np.uint64)
        self.latest = 0
        self.observed = np.zeros(self.n, dtype=np.int64)
        self.label_kind = (
            "offered_suffix" if offered is not None else "original_increment"
        )
        if len(rows) != self.n or len(sessions) != self.n or len(features) != self.n:
            raise ValueError("Training arrays describe different request counts")
        if np.any(self.ts < 0) or np.any(self.ts[1:] < self.ts[:-1]):
            raise ValueError("Training requests must be chronological and nonnegative")
        last_session = {}
        owners = {}
        introducing_request = {}
        for i, blocks in enumerate(rows):
            session = sessions[i]
            hour = int(self.ts[i] // model_training.H)
            if hour >= len(self.Y):
                raise ValueError("Request is beyond the supported training window")
            distinct = list(dict.fromkeys(blocks))
            if offered is None:
                # Same ownership rule as initial_reuse_counts.c: a change of
                # session starts a new introducing request for that block.
                for block in distinct:
                    if last_session.get(block) != session:
                        introducing_request[block] = i
                        self.c[i] += 1
                    else:
                        self.Y[hour, introducing_request[block]] += 1
                    last_session[block] = session
                continue
            for block in distinct:
                if last_session.get(block) == session:
                    for earlier in owners.get(block, []):
                        self.Y[hour, earlier] += 1
                last_session[block] = session
            candidate = list(dict.fromkeys(offered[i]))
            if not set(candidate).issubset(distinct):
                raise ValueError("Training offer contains blocks outside its request")
            self.c[i] = len(candidate)
            for block in candidate:
                owners.setdefault(block, []).append(i)


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
    predictions = np.zeros(data.n, dtype=float)
    hours = np.unique(data.ts // model_training.H)
    for hour in hours:
        cut = int(hour * model_training.H)
        earlier, _ = data.training_examples(cut)
        if not len(earlier):
            continue
        model, _ = fit_research_model(data, cut, settings)
        selected = np.flatnonzero(data.ts // model_training.H == hour)
        features = data.feature_table(selected, settings["features"])
        predictions[selected] = model_training.predict_reuse(model, features)
    return predictions
