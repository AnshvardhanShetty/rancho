"""Apply the training-time user category codes before tree prediction."""

import numpy as np


def encode_tree_features(feature_table, categories_by_column):
    """Use the model's saved user codes for incoming requests"""
    encoded_features = np.where(
        np.isfinite(feature_table) & (feature_table >= 0), feature_table, np.nan
    )
    for column, known_categories in enumerate(categories_by_column):
        if known_categories is None:
            continue
        encoded_features[:, column] = -1
        if len(known_categories):
            category_indices = np.searchsorted(
                known_categories, feature_table[:, column]
            )
            known_category_mask = category_indices < len(known_categories)
            known_category_mask &= (
                known_categories[
                    np.minimum(category_indices, len(known_categories) - 1)
                ]
                == feature_table[:, column]
            )
            encoded_features[known_category_mask, column] = category_indices[
                known_category_mask
            ]
    return encoded_features
