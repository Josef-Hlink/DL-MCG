from typing import Tuple

import numpy as np


def train_test_split(
    X: np.ndarray,
    test_size: float,
    y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray]:
    """ Splits the data up into a train and test set, labels are optional """

    return_y = True
    if y is None:
        y = np.zeros(X.shape[0])
        return_y = False
    
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
    
    # randomize
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]

    # split
    split = int(X.shape[0] * test_size)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if return_y:
        return X_train, X_test, y_train, y_test
    return X_train, X_test
