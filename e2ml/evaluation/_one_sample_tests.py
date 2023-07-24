import numpy as np

from sklearn.utils.validation import check_array, check_scalar
from scipy import stats


def z_test_one_sample(sample_data, mu_0, sigma, test_type="two-sided"):
    """Perform a one-sample z-test.

    Parameters
    ----------
    sample_data : array-like of shape (n_samples,)
        Sample data drawn from a population.
    mu_0 : float or int
        Population mean assumed by the null hypothesis.
    sigma: float
        True population standard deviation.
    test_type : {'right-tail', 'left-tail', 'two-sided'}
        Specifies the type of test for computing the p-value.

    Returns
    -------
    z_statistic : float
        Observed z-transformed test statistic.
    p : float
        p-value for the observed sample data.
    """
    # Check parameters.
    sample_data = check_array(sample_data, ensure_2d=False)
    mu_0 = check_scalar(mu_0, name="mu_0", target_type=(int, float))
    sigma = check_scalar(sigma, name="sigma", target_type=(int, float), min_val=0, include_boundaries="neither")
    if test_type not in ["two-sided", "left-tail", "right-tail"]:
        raise ValueError("`test_type` must be in `['two-sided', 'left-tail', 'right-tail']`")

    # BEGIN SOLUTION

    # Determine degrees of freedom.
    sample_size = len(sample_data)

    # Compute empirical mean.
    mu_empirical = np.mean(sample_data)

    # Compute z-test statistic.
    z_statistic = (mu_empirical - mu_0) / (sigma / np.sqrt(sample_size))

    # Compute p-value.
    p_left = stats.norm.cdf(z_statistic)
    p_right = 1 - p_left
    if test_type == "two-sided":
        p = 2 * np.min((p_left, p_right))
    elif test_type == "left-tail":
        p = p_left
    else:
        p = p_right

    return z_statistic, p

    # END SOLUTION


def t_test_one_sample(sample_data, mu_0, test_type="two-sided"):
    """Perform a one-sample t-test.

    Parameters
    ----------
    sample_data : array-like of shape (n_samples,)
        Sample data drawn from a population.
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
    sample_data = check_array(sample_data, ensure_2d=False)
    mu_0 = check_scalar(mu_0, name="mu_0", target_type=(int, float))
    if test_type not in ["two-sided", "left-tail", "right-tail"]:
        raise ValueError("`test_type` must be in `['two-sided', 'left-tail', 'right-tail']`")

    # BEGIN SOLUTION

    # Determine degrees of freedom.
    sample_size = len(sample_data)
    nu = sample_size - 1

    # Compute empirical mean.
    mu_empirical = np.mean(sample_data)

    # Compute empirical standard deviation.
    sigma_empirical = np.std(sample_data, ddof=1)

    # Compute t-test statistic.
    t_statistic = (mu_empirical - mu_0) / (sigma_empirical / np.sqrt(sample_size))

    # Compute p-value.
    p_left = stats.t.cdf(t_statistic, df=nu)
    p_right = 1 - p_left
    if test_type == "two-sided":
        p = 2 * np.min((p_left, p_right))
    elif test_type == "left-tail":
        p = p_left
    else:
        p = p_right

    return t_statistic, p

    # END SOLUTION

