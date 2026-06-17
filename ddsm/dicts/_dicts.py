from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_array

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
