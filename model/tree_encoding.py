"""Encode tree inputs using categories seen during training."""

import numpy as np


# Collect user IDs seen during fitting; their positions become category codes.
# Numeric columns do not need a category list. Save this mapping with the model.
def build_category_values(feature_table, cats):
    return [
        (
            np.unique(
                feature_table[
                    np.isfinite(feature_table[:, i]) & (feature_table[:, i] >= 0), i
                ]
            )
            if i in cats
            else None
        )
        for i in range(feature_table.shape[1])
    ]


def encode_tree_features(feature_table, vocab):
    """Map known categories to training IDs and give unseen categories the code -1."""
    # Keep numeric inputs; mark invalid values as missing.
    z = np.where(
        np.isfinite(feature_table) & (feature_table >= 0), feature_table, np.nan
    )
    # Replace categorical values with their saved codes; unseen users receive -1.
    for i, v in enumerate(vocab):
        if v is None:
            continue
        z[:, i] = -1
        if len(v):
            at = np.searchsorted(v, feature_table[:, i])
            ok = at < len(v)
            ok &= v[np.minimum(at, len(v) - 1)] == feature_table[:, i]
            z[ok, i] = at[ok]
    return z
