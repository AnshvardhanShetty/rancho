"""Apply the training-time user category codes before tree prediction."""

import numpy as np


def encode(X, vocab):
    """Use the model's saved user codes for incoming requests"""
    z = np.where(np.isfinite(X) & (X >= 0), X, np.nan)
    for i, values in enumerate(vocab):
        if values is None:
            continue
        z[:, i] = -1
        if len(values):
            at = np.searchsorted(values, X[:, i])
            good = at < len(values)
            good &= values[np.minimum(at, len(values) - 1)] == X[:, i]
            z[good, i] = at[good]
    return z
