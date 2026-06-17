from __future__ import annotations

import abc

import numpy as np


class BaseDict(abc.ABC):
    """
    Abstract base class for dictionary functions.

    Attributes
    ----------
    dim_ : int
        The dimension of the state space.
    """
    dim_: int

    def __init__(self) -> None:
        """Initialize the BaseDict instance."""
        super(BaseDict, self).__init__()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for the lift method.

        Parameters
        ----------
        X : ndarray
            Input data in the state space.

        Returns
        -------
        ndarray
            Lifted representation of the input.
        """
        return self.lift(X)

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Return the number of dictionary elements.

        Returns
        -------
        int
            The dictionary size.
        """
        return -1

    @abc.abstractmethod
    def lift(self, X: np.ndarray) -> np.ndarray:
        """
        Map input data to the lifted feature space.

        Parameters
        ----------
        X : ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Lifted data of shape (n_samples, dim_).
        """
        pass

    @abc.abstractmethod
    def reconstruct(self, PsiX: np.ndarray) -> np.ndarray:
        """
        Reconstruct state data from the lifted space.

        Parameters
        ----------
        PsiX : ndarray
            Lifted data of shape (n_samples, dim_).

        Returns
        -------
        ndarray
            Reconstructed data in the state space.
        """
        pass

    @abc.abstractmethod
    def diff(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the first-order derivative of the lifting functions.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            First derivative values.
        """
        pass

    @abc.abstractmethod
    def diff2(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the second-order derivative of the lifting functions.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            Second derivative values.
        """
        pass

    @abc.abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> BaseDict:
        """
        Fit the dictionary parameters to the provided data.

        Parameters
        ----------
        X : ndarray
            Training data.
        **kwargs : dict
            Additional hyperparameters for fitting.

        Returns
        -------
        BaseDict
            The fitted instance.
        """
        pass