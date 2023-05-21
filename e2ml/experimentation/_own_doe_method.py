import numpy as np
import math


def own_doe_method(levels, step_size=1):
    """
    Implements a custom design of experiments method for given factor-wise levels.

    Parameters
    ----------
    levels : array-like of shape (n_factors,)
        Integer array indicating the number of levels of each input design factor (variable).
    step_size : int, optional
        Step size for the design matrix. Default is 1.

    Returns
    -------
    X : np.ndarray of shape (n_combs, n_factors)
        Design matrix with coded levels 0 to k-1 for a k-level factor, each one at a time.
    """
    levels = np.asarray(levels)

    # Create a matrix with all possible combinations of levels for each factor
    # skipping every step_size'th level to reduce the number of combinations
    # and transform it into a matrix with one combination per row
    # (e.g. for levels = [3, 5] and step_size = 2, the matrix would be:
    # [[0, 0], [0, 2], [0, 4], [2, 0], [2, 2], [2, 4]])
    X = np.array(np.meshgrid(*[np.arange(0, level, step_size) for level in levels])).T.reshape(-1, len(levels))

    return X.astype(int)
