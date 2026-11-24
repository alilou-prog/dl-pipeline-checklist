"""Leak-free preprocessing utilities for tabular classification pipelines."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clips numeric columns to quantile bounds learned strictly on the training set."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        """Initializes lower and upper quantile bounds in [0, 1]."""
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        """Learns per-column lower and upper quantile bounds from X."""
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.quantile(arr, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        """Clips X element-wise to the learned quantile bounds."""
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        """Preserves input feature names for scikit-learn ColumnTransformer compatibility."""
        return np.asarray(input_features)


def stratified_split_3way(
    X,
    y,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """Splits (X, y) into stratified train, validation, and test subsets."""
    temp_size = val_size + test_size
    if not (0.0 < temp_size < 1.0):
        raise ValueError("val_size + test_size must be strictly between 0 and 1.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=random_state
    )
    rel_test = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
