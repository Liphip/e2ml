import numpy as np
import itertools

from sklearn.utils.validation import check_array
from scipy import stats

from ._one_sample_tests import t_test_one_sample


def t_test_paired(sample_data_1, sample_data_2=None, mu_0=0, test_type="two-sided"):
    """Perform a paired t-test.

    Parameters
    ----------
    sample_data_1 : array-like of shape (n_samples,)
        Sample data drawn from a population 1. If no sample data is given, `sample_data_1` is assumed to consist of
        differences.
    sample_data_2 : array-like of shape (n_samples,), optional (default=None)
        Sample data drawn from a population 2.
    mu_0 : float or int
        Population mean assumed by the null hypothesis.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.

    Returns
    -------
    t_statistic : float
        Observed t-transformed test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data_1 = check_array(sample_data_1, ensure_2d=False)
    if sample_data_2 is not None:
        sample_data_2 = check_array(sample_data_2, ensure_2d=False)

    """
    # Compute differences.
    if sample_data_2 is None:
        sample_data_2 = np.zeros_like(sample_data_1)
    sample_data_diff = sample_data_1 - sample_data_2

    # Perform one-sample t-test.
    t_statistic, p = t_test_one_sample(sample_data_diff, mu_0=mu_0, test_type=test_type)

    return t_statistic, p
    """

    sample_data_diff = sample_data_1 if sample_data_2 is None else sample_data_1 - sample_data_2

    return t_test_one_sample(sample_data_diff, mu_0=mu_0, test_type=test_type)


def wilcoxon_signed_rank_test(sample_data_1, sample_data_2=None, test_type="two-sided"):
    """Perform a Wilcoxon signed-rank test.

    Parameters
    ----------
    sample_data_1 : array-like of shape (n_samples,)
        Sample data drawn from a population 1. If no sample data is given, `sample_data_1` is assumed to consist of
        differences.
    sample_data_2 : array-like of shape (n_samples,), optional (default=None)
        Sample data drawn from a population 2.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.

    Returns
    -------
    w_statistic : float
        Observed positive rank sum as test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data_1 = check_array(sample_data_1, ensure_2d=False)
    if sample_data_2 is not None:
        sample_data_2 = check_array(sample_data_2, ensure_2d=False)
    if test_type not in ["two-sided", "left-tail", "right-tail"]:
        raise ValueError("`test_type` must be in `['two-sided', 'left-tail', 'right-tail']`")

    sample_data_diff = sample_data_1 if sample_data_2 is None else sample_data_1 - sample_data_2

    # remove zero differences
    sample_data_diff = sample_data_diff[sample_data_diff != 0]

    # remember pos/neg diffs
    pos_diffs = np.where(sample_data_diff > 0)[0]
    neg_diffs = np.where(sample_data_diff < 0)[0]

    # compute rank of absolute differences
    ranks = stats.rankdata(np.abs(sample_data_diff))

    # test statistic: (marel: pos rank sum, franz: min between pos and neg rank sum)
    pos_rank_sum = np.sum(ranks[pos_diffs])
    neg_rank_sum = np.sum(ranks[neg_diffs])

    w_statistic = pos_rank_sum
    #w_statistic = np.min([pos_rank_sum, neg_rank_sum])

    # set up sampling distribution
    sample_size = len(sample_data_diff)

    p_left = 0
    p_right = 0

    if sample_size > 30: # assume normal distribution
        # compute mu and sigma
        mu = sample_size * (sample_size + 1) / 4
        sigma = sample_size * (sample_size + 1) * (2 * sample_size + 1) / 24

        # compute z_statistic
        z_statistic = (w_statistic - mu) / np.sqrt(sigma)

        # determine p_left and p_right
        p_left = stats.norm.cdf(z_statistic)
        p_right = 1 - p_left
    elif sample_size <= 30: # use sampling distribution
        # compute all possible combinations
        w_dict = {}
        for comb in itertools.product([0, 1], repeat=sample_size):
            w_statistic_ = np.sum(np.array(comb) * ranks[None, :])
            w_dict[w_statistic_] = w_dict.get(w_statistic_, 0) + 1
        w_stat_arr = np.array(list(w_dict.keys()))
        p_arr = np.array(list(w_dict.values())) / 2 ** sample_size

        # determine p_left and p_right using the above distribution
        p_left = p_arr[w_stat_arr <= w_statistic].sum()
        p_right = p_arr[w_stat_arr >= w_statistic].sum()

    # determine p-value based on test_type
    if test_type == "two-sided":
        p = 2 * np.min([p_left, p_right])
    elif test_type == "left-tail":
        p = p_left
    elif test_type == "right-tail":
        p = p_right

    return w_statistic, p

