import os
from warnings import warn

import numpy as np


def get_dirs(parent_file: str):
    """
    Returns a dictionary of directories to be used in the program. 
    Also builds these directories if they are not already there.
    ```
    base
    ├── data
    └── results
        ├── csv
        ├── plots
        ├── figs
        └── models
    ```
    """

    src = os.path.dirname(os.path.abspath(parent_file))
    root = os.sep.join(src.split(os.sep)[:-1])
    data = os.path.join(root, 'data')
    results = os.path.join(root, 'results')
    csv = os.path.join(results, 'csv')
    plots = os.path.join(results, 'plots')
    figs = os.path.join(results, 'figs')
    models = os.path.join(results, 'models')
    
    dirs: dict[str, str] = {}
    for directory in (data, results, csv, plots, figs, models):
        basename = os.path.basename(directory)
        if not os.path.exists(directory):
            os.mkdir(directory)
            print()
            warn(f'Created empty {basename} directory at "{directory}".')
            print()
        dirs[basename] = directory + os.sep
    return dirs


def train_test_split(X: np.ndarray, test_size: float, y: np.ndarray = None):
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
