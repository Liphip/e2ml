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

    # TODO 


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

    # TODO 

