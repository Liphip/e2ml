import numpy as np
from sklearn.utils import column_or_1d, check_scalar


def one_factor_at_a_time(levels):
    """
    Implements the standard one-factor-at-a-time approach for given factor-wise levels.

    Parameters
    ----------
    levels : array-like of shape (n_factors,)
        Integer array indicating the number of levels of each input design factor (variable).

    Returns
    -------
    X : np.ndarray of shape (n_combs, n_factors)
        Design matrix with coded levels 0 to k-1 for a k-level factor, each one at a time.
    """
    levels = np.asarray(levels)
    X = np.zeros((levels.sum() + 1 - levels.shape[0], levels.shape[0]))

    summed_levels = 1
    for i, level in enumerate(levels):
        for j, lev in enumerate(range(1, level)):
            X[summed_levels + j, i] = lev
        summed_levels += level - 1

    """ Example solution:
    for i, level in enumerate(levels):
        v_min = levels[:i].sum() -1 + (i > 0)
        values = np.arange(i > 0, level)
        v_max = v_min + len(values)
        X[v_min:v_max, i] = values
    """

    return X.astype(int)
