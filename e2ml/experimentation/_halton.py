import numpy as np

from sklearn.utils import check_scalar, check_array


def van_der_corput_sequence(n_max, base=2):
    """Generate van der Corput sequence for 1 <= n <= n_max and given base.

    Parameters
    ----------
    n_max : int
        Number of elements of the sequence.
    base : int
        Base of the sequence.

    Returns
    -------
    sequence : numpy.ndarray of shape (n_max,)
        Generate van der Corput sequence for 1 <= n <= n_max and given base.
    """
    # Check parameters.
    check_scalar(n_max, name="n_max", target_type=int, min_val=1)
    check_scalar(base, name="base", target_type=int, min_val=2)

    # BEGIN SOLUTION
    sequence = []
    for i in range(1, n_max + 1):
        n_th_number, denom = 0.0, 1.0
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return np.array(sequence)
    # END SOLUTION


def primes_from_2_to(n_max):
    """Generate prime numbers from 2 to n_max.

    Parameters
    ----------
    n_max : int
        Maximum prime number to be generated.

    Returns
    -------
    prime_numbers : numpy.ndarray of shape (n_prime_numbers)
        Array of all prime numbers from 2 to n_max.
    """
    # Check parameters.
    check_scalar(n_max, name="n_max", target_type=int, min_val=2)

    # BEGIN SOLUTION
    sieve = np.ones(n_max // 3 + (n_max % 6 == 2), dtype=bool)
    for i in range(1, int(n_max**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    prime_numbers = np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]
    return prime_numbers
    # END SOLUTION


def halton_unit(n_samples, n_dimensions):
    """Generate a specified number of samples according to a Halton sequence in the unit hypercube.

    Parameters
    ----------
    n_samples : int
        Number of samples to be generated.
    n_dimensions : int
        Dimensionality of the generated samples.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples, n_dimensions)
        Generated samples.
    """
    # Check parameters.
    check_scalar(n_samples, name="n_samples", target_type=int, min_val=1)
    check_scalar(n_dimensions, name="n_dimensions", target_type=int, min_val=1)

    # BEGIN SOLUTION
    big_number = 10
    while "Not enough prime numbers":
        base = primes_from_2_to(big_number)[:n_dimensions]
        if len(base) == n_dimensions:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    X = [van_der_corput_sequence(n_samples, int(dim)) for dim in base]
    X = np.stack(X, axis=-1)

    return X
    # END SOLUTION


def halton(n_samples, n_dimensions, bounds=None):
    """Generate a specified number of samples according to a Halton sequence in a user-specified hypercube.

    Parameters
    ----------
    n_samples : int
       Number of samples to be generated.
    n_dimensions : int
       Dimensionality of the generated samples.
    bounds : None or array-like of shape (n_dimensions, 2)
       `bounds[d, 0]` is the minimum and `bounds[d, 1]` the maximum
       value for dimension `d`.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples, n_dimensions)
       Generated samples.
    """
    # Check parameters.
    check_scalar(n_samples, name="n_samples", target_type=int, min_val=1)
    check_scalar(n_dimensions, name="n_dimensions", target_type=int, min_val=1)
    if bounds is not None:
        bounds = check_array(bounds)
        if bounds.shape[0] != n_dimensions or bounds.shape[1] != 2:
            raise ValueError("`bounds` must have shape `(n_dimensions, 2)`.")
    else:
        bounds = np.zeros((n_dimensions, 2))
        bounds[:, 1] = 1

    # BEGIN SOLUTION
    X = halton_unit(n_samples, n_dimensions)
    x_min = bounds[:, 0]
    x_max = bounds[:, 1]
    X = X * (x_max - x_min) + x_min
    return X
    # END SOLUTION
