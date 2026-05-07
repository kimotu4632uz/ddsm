from __future__ import annotations
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.preprocessing import PolynomialFeatures
from ..base._dicts import BaseDict

class IdentityDict(BaseDict):
    """
    Identity dictionary with a bias (constant) term.

    Lifts input $X$ to $[1, X]$.
    """
    dim_: int

    def __init__(self) -> None:
        """Initialize the IdentityDict instance."""
        super(IdentityDict, self).__init__()

    def __len__(self) -> int:
        """
        Return the number of dictionary elements (dim + 1).

        Returns
        -------
        int
            Dimension of the lifted space.
        """
        return self.dim_ + 1 if hasattr(self, 'dim_') else 0

    def lift(self, X: np.ndarray) -> np.ndarray:
        """
        Lift input data by appending a bias term.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            State data.

        Returns
        -------
        ndarray of shape (n_samples, n_features + 1)
            Lifted data $[1, X]$.
        """
        X = check_array(X)
        self.fit(X)
        PsiX = np.hstack((np.ones((X.shape[0], 1)), X))
        return PsiX

    def reconstruct(self, PsiX: np.ndarray) -> np.ndarray:
        """
        Extract state features from the lifted representation.

        Parameters
        ----------
        PsiX : ndarray
            Lifted data.

        Returns
        -------
        ndarray
            State data.
        """
        PsiX = check_array(PsiX)
        X = PsiX[:, 1:]
        return X

    def diff(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of the identity lifting.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, dim+1, dim)
            Jacobian matrix.
        """
        X = check_array(X)
        self.fit(X)
        J = np.zeros((X.shape[0], self.dim_ + 1, self.dim_))
        J[:, 1:, :] = np.eye(self.dim_)
        return J

    def diff2(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the identity lifting (returns zeros).

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, dim+1, dim, dim)
            Hessian matrix.
        """
        X = check_array(X)
        self.fit(X)
        H = np.zeros((X.shape[0], self.dim_ + 1, self.dim_, self.dim_))
        return H

    def fit(self, X: np.ndarray, **kwargs) -> IdentityDict:
        """
        Set the dimension based on input data.

        Parameters
        ----------
        X : ndarray
            Input data.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        IdentityDict
            The fitted instance.
        """
        X = check_array(X)
        if not hasattr(self, 'dim_') or X.shape[1] != self.dim_:
            self.dim_ = X.shape[1]
        return self

class MonomialsDict(BaseDict):
    """
    Polynomial dictionary consisting of monomials up to a specified degree.

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of the polynomial features.
    """
    degree: int
    dim_: int
    degrees_: np.ndarray

    def __init__(self, degree: int = 2) -> None:
        """Initialize the MonomialsDict instance."""
        super(MonomialsDict, self).__init__()
        self.degree = degree

    def __len__(self) -> int:
        """
        Number of monomial features.

        Returns
        -------
        int
            Count of monomials.
        """
        return len(self.degrees_) if hasattr(self, 'degrees_') else 0

    def lift(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data into the monomial feature space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray
            Evaluated monomials.
        """
        X = check_array(X)
        self.fit(X)
        return np.prod(X[:, np.newaxis, :] ** self.degrees_[np.newaxis, :, :], axis=2)

    def reconstruct(self, PsiX: np.ndarray) -> np.ndarray:
        """
        Extract first-order terms from the monomial features.

        Parameters
        ----------
        PsiX : ndarray
            Monomial features.

        Returns
        -------
        ndarray
            Reconstructed state features.
        """
        PsiX = check_array(PsiX)
        indices = np.where((self.degrees_.sum(axis=1) == 1) & (self.degrees_.max(axis=1) == 1))[0]
        return PsiX[:, indices]

    def diff(self, X: np.ndarray) -> np.ndarray:
        """
        Compute first-order derivatives of monomials.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_monomials, dim)
            Jacobian of the monomials.
        """
        X = check_array(X)
        self.fit(X)
        P_diff = self.degrees_[:, np.newaxis, :] - np.eye(self.dim_)[np.newaxis, :, :]
        P_diff = np.maximum(P_diff, 0)
        terms = X[:, np.newaxis, np.newaxis, :] ** np.maximum(P_diff[np.newaxis, ...], 0)
        diff_monomials = np.prod(terms, axis=-1)
        coeffs = self.degrees_[np.newaxis, :, :]
        J = coeffs * diff_monomials * (coeffs > 0)
        return J

    def diff2(self, X: np.ndarray) -> np.ndarray:
        """
        Compute second-order derivatives of monomials.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_monomials, dim, dim)
            Hessian of the monomials.
        """
        X = check_array(X)
        self.fit(X)
        I = np.eye(self.dim_)
        P_diff2 = self.degrees_[:, np.newaxis, np.newaxis, :] - I[np.newaxis, :, np.newaxis, :] - I[np.newaxis, np.newaxis, :, :]
        P_diff2 = np.maximum(P_diff2, 0)
        terms = X[:, np.newaxis, np.newaxis, np.newaxis, :] ** P_diff2[np.newaxis, ...]
        diff2_monomials = np.prod(terms, axis=-1)
        coeffs = (self.degrees_[:, :, np.newaxis] * (self.degrees_[:, np.newaxis, :] - I[np.newaxis, :, :]))[np.newaxis, ...]
        H = coeffs * diff2_monomials * (coeffs > 0)
        return H

    def fit(self, X: np.ndarray, **kwargs) -> MonomialsDict:
        """
        Generate monomial powers based on input dimension and degree.

        Parameters
        ----------
        X : ndarray
            Training data.
        **kwargs : dict
            Optional 'degree' override.

        Returns
        -------
        MonomialsDict
            The fitted instance.
        """
        X = check_array(X)
        is_regenerate = False
        if not hasattr(self, 'dim_') or X.shape[1] != self.dim_:
            self.dim_ = X.shape[1]
            is_regenerate = True
        if kwargs.get('degree', self.degree) != self.degree:
            self.degree = kwargs['degree']
            is_regenerate = True
        if is_regenerate:
            self.degrees_ = self._generate_degrees()
        return self

    def _generate_degrees(self) -> np.ndarray:
        """Internal helper to generate polynomial exponent matrix."""
        poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        poly.fit(np.zeros((1, self.dim_)))
        return poly.powers_

    @property
    def degrees(self) -> np.ndarray:
        """
        Exponent matrix of the monomials.

        Returns
        -------
        ndarray
            Shape (n_monomials, dim).
        """
        return self.degrees_ if hasattr(self, 'degrees_') else np.array([])