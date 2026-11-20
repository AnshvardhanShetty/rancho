"""Encode tree inputs using categories seen during training."""

import numpy as np


# Collect user IDs seen during fitting; their positions become category codes.
# Numeric columns do not need a category list. Save this mapping with the model.
def build_category_values(feature_table, categorical_columns):
    """Save the user categories seen in this fit"""
    return [
        (
            np.unique(
                feature_table[
                    np.isfinite(feature_table[:, column])
                    & (feature_table[:, column] >= 0),
                    column,
                ]
            )
            if column in categorical_columns
            else None
        )
        for column in range(feature_table.shape[1])
    ]


def encode_tree_features(feature_table, categories_by_column):
    """Map known categories to training IDs and give unseen categories the code -1."""
    # Keep numeric inputs; mark invalid values as missing.
    encoded_features = np.where(
        np.isfinite(feature_table) & (feature_table >= 0), feature_table, np.nan
    )
    # Replace categorical values with their saved codes; unseen users receive -1.
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
