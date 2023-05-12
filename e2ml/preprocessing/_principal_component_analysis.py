import numpy as np

from sklearn.base import BaseEstimator


class PrincipalComponentAnalysis(BaseEstimator):
    """PrincipalComponentAnalysis

    This class implements the principal component analysis for n_samples >=
    n_features.

    Parameters
    ----------
    n_components : {int, float}
        If `n_components` is an integer, the number of dimension will be
        reduced from `n_features` to `n_components`. If `0 < n_components < 1`,
        select the number of components such that the amount of variance that
        needs to be explained is greater or equal than the percentage specified
        by `n_components`.

    Attributes
    ----------
    n_components_ : int,
        Number of selected principal components.
    mu_ : numpy.narray, shape (n_features)
        Means of features where mu_[i] is the mean of the i-th feature.
    lmbdas_ : numpy.ndarray, shape (n_features)
        Eigenvalues, where `lambdas_[i]` is the eigenvalue of the i-th
        eigenvector.
    U_ : numpy.ndarray, shape (n_features, n_features)
        Sorted eigenvector matrix where `U_[:, i]` is the i-th eigenvector
        with the i-th highest eigenvalue.
    """
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        """
        Determine required parameters of the PCA.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        self : PrincipalComponentAnalysis
            The fitted PrincipalComponentAnalysis object.
        """
        # Transform to numpy.ndarray.
        X = np.array(X)

        # Number of samples.
        # TODO 

        # Compute mean `self.mu_` of each feature, which is zero if samples have been
        # standardized.
        # TODO 

        # Compute DxD covariance matrix `S` (take mean into account).
        # TODO 

        # Compute eigenvalues `self.lmbdas_` and eigenvectors `self.U_`.
        # TODO 

        # Sort eigenvalues and eigenvectors in decreasing order.
        # TODO 

        # Determine number of selected components.
        self._determine_M()

        return self

    def transform(self, X):
        """
        Transforms samples from the D-dimensional input space into
        the M-dimensional projection space.

        Parameters
        ----------
        X : numpy.ndarray, sahpe (n_samples, n_features)
            Samples in the input space.

        Returns
        -------
        Z : array-like, shape (n_samples, n_components_)
            Transformed samples in the projection space.
        """
        # TODO 

    def inverse_transform(self, Z):
        """
        Retransforms samples from the M-dimensional projection space into
        the D-dimensional input space.

        Parameters
        ----------
        Z : array-like, shape (n_samples, n_components_)
            Samples in the projection space.

        Returns
        -------
        X : numpy.ndarray, sahpe (n_samples, n_features)
            Re-transformed samples in the input space.
        """
        # TODO 

    def _determine_M(self):
        """
        Determine number of finally selected components.
        """
        if self.n_components >= 1:
            # If `n_components` is an integer, the number `self.n_components_` of selected dimension will
            # be reduced from `n_features` to `n_components`.
            # TODO 
        elif 0 < self.n_components < 1:
            # If `0 < n_components < 1`,  select the number `self.n_components_` of components such
            # that the amount of variance that needs to be explained is greater
            # or equal than the percentage specified by `n_components`.
            # TODO 
        else:
            raise ValueError('Invalid `n_components` parameter.')