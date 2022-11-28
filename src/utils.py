import os
from time import perf_counter
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


def train_test_split(X: np.ndarray, y: np.ndarray = None, test_size: float = 0.2):
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
    X_train, X_test = X[split:], X[:split]
    y_train, y_test = y[split:], y[:split]

    if return_y:
        return X_train, X_test, y_train, y_test
    return X_train, X_test


class ProgressBar:
    done_char = '\033[32m' + '\033[1m' + '━' + '\033[0m'   # green bold ━, reset after
    todo_char = '\033[31m' + '\033[2m' + '─' + '\033[0m'   # red faint ─, reset after

    def __init__(self, n_iters: int, p_id: str) -> None:
        self.n_iters = n_iters
        self.len_n_iters = len(str(n_iters))
        print(p_id)
        print('\r' + 50 * self.todo_char + ' 0%', end='')
        self.start_ts = perf_counter()

    def __call__(self, iteration: int) -> None:
        """Updates and displays a progress bar on the command line."""
        percentage = 100 * (iteration+1) // self.n_iters            # floored percentage
        if percentage == 100 * iteration // self.n_iters: return    # prevent printing same line multiple times
        steps = 50 * (iteration+1) // self.n_iters                  # chars representing progress

        bar = (steps)*self.done_char + (50-steps)*self.todo_char    # the actual bar
        
        runtime = perf_counter() - self.start_ts
        if iteration+1 == self.n_iters:             # flush last suffix with spaces and place carriage at newline
            suffix = ' completed in ' + f'{runtime:.2f} sec'  + ' ' * 50 + '\n'
        else:                                       # print iteration number
            percentage_float = (100 * (iteration+1) / self.n_iters)
            eta = (100-percentage_float) / percentage_float * runtime
            suffix = f' {percentage}% (ETA {eta:.1f} sec) '
        
        print('\r' + bar + suffix, end='')
        return
