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
    # BEGIN SOLUTION
    levels = column_or_1d(levels, dtype=int)
    check_scalar(levels.min(), min_val=1, name="minimum level per factor", target_type=np.int64)
    X = np.zeros((levels.sum() + 1 - len(levels), len(levels)))
    for l, level in enumerate(levels):
        v_min = levels[:l].sum() - l + (l > 0)
        values = np.arange(l > 0, level)
        v_max = v_min + len(values)
        X[v_min:v_max, l] = values
    return X.astype(int)
    # END SOLUTION
