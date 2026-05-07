from __future__ import annotations
import warnings
from typing import Any
import numpy as np
from scipy import linalg
from sklearn.utils.validation import validate_data, check_is_fitted
from ..base._dicts import BaseDict
from ..base._estimators import DDSMBaseEstimator
from ..dicts._dicts import IdentityDict

class SINDy(DDSMBaseEstimator):
    """
    Sparse Identification of Non-linear Dynamics (SINDy) estimator.

    Uses Sequentially Thresholded Least Squares (STLS) to find a sparse
    mapping between a feature library and time derivatives.

    Parameters
    ----------
    psi_cls : type[BaseDict], default=IdentityDict
        Class type for the candidate function library (dictionary).
    psi_kwargs : dict, optional
        Arguments for initializing `psi_cls`.
    threshold : float, default=0.1
        Sparsity threshold. Coefficients with absolute values smaller
        than this are set to zero.
    max_iter : int, default=20
        Maximum number of STLS iterations.

    Attributes
    ----------
    psi_ : BaseDict
        Fitted candidate library instance.
    L_ : ndarray
        The estimated sparse coefficient matrix.
    n_iter_ : int
        Actual number of iterations performed during fitting.
    """
    psi_: BaseDict
    L_: np.ndarray
    _psi_cls: type[BaseDict]
    _psi_kwargs: dict[str, Any]
    _threshold: float
    _max_iter: int
    _is_psi_fitted: bool
    _y_ndim_1d: bool
    _y_features: int

    def __init__(
        self,
        psi_cls: type[BaseDict] = IdentityDict,
        psi_kwargs: dict[str, Any] | None = None,
        threshold: float = 0.1,
        max_iter: int = 20
    ) -> None:
        """Initialize the SINDy estimator."""
        super(SINDy, self).__init__()
        self.psi_cls = psi_cls
        self.psi_kwargs = psi_kwargs
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> SINDy:
        """
        Fit the SINDy model using STLS.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            State at current time steps.
        y : ndarray of shape (n_samples, n_features)
            Time derivatives of the state (dot{X}).

        Returns
        -------
        SINDy
            The fitted instance.

        Raises
        ------
        ValueError
            If y contains non-numeric data.
        """
        X, y = validate_data(self, X, y, reset=True, multi_output=True)
        if y.dtype == object:
            raise ValueError('Unknown label type: SINDy does not support object dtype for y. Ensure that y is a numeric array.')
        self._y_ndim_1d = (y.ndim == 1)
        if self._y_ndim_1d:
            y = y[:, np.newaxis]

        self._y_features = y.shape[1]
        if X.shape[1] != y.shape[1]:
            warnings.warn(f'Number of features in X ({X.shape[1]}) does not match number of features in y ({y.shape[1]}). This may indicate that the data is not properly aligned. Ensure that each row of X corresponds to the same time step as the corresponding row of y.')
            _y = np.zeros((y.shape[0], X.shape[1]), dtype=y.dtype)
            col = min(X.shape[1], y.shape[1])
            _y[:, :col] = y[:, :col]
            y = _y
        self._fit_psi(X)
        theta = self.psi_.lift(X)
        xi = linalg.lstsq(theta, y)[0].T
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            xi[np.abs(xi) < self.threshold] = 0.0
            for i in range(xi.shape[0]):
                indices = np.where(xi[i, :] != 0)[0]
                if 0 < len(indices):
                    xi[i, indices] = linalg.lstsq(theta[:, indices], y[:, i])[0].flatten()
            self.n_iter_ += 1
        self.L_ = xi.T
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict time derivatives dot{X} using the sparse model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Current state.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Predicted time derivatives.
        """
        check_is_fitted(self, 'L_')
        X = validate_data(self, X, reset=False)
        PsiX = self.psi_.lift(X)
        y = PsiX @ self.L_
        if hasattr(self, '_y_features') and y.shape[1] != self._y_features:
            y = y[:, :self._y_features]
        if hasattr(self, '_y_ndim_1d') and self._y_ndim_1d:
            y = y.ravel()
        return y

    def _fit_psi(self, X: np.ndarray) -> None:
        """
        Initialize and fit the library dictionary (Psi).

        Parameters
        ----------
        X : ndarray
            Training data used to determine library parameters.
        """
        if not hasattr(self, 'psi_') or not self._is_psi_fitted:
            self.psi_ = self.psi_cls(**self.psi_kwargs.copy() if self.psi_kwargs is not None else {})
            self.psi_.fit(X)
            self._is_psi_fitted = True

    @property
    def right_L(self) -> np.ndarray:
        """The Koopman generator matrix acting on the right."""
        check_is_fitted(self, 'L_')
        return self.L_

    @property
    def left_L(self) -> np.ndarray:
        """The Koopman generator matrix acting on the left (adjoint)."""
        return self.right_L.conj().T

    @property
    def psi_cls(self) -> type[BaseDict]:
        """Class type for the lifting dictionary. Resets fit state on change."""
        return self._psi_cls

    @psi_cls.setter
    def psi_cls(self, value: type[BaseDict]) -> None:
        self._psi_cls = value
        self._is_psi_fitted = False

    @property
    def psi_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for dictionary initialization. Resets fit state on change."""
        return self._psi_kwargs

    @psi_kwargs.setter
    def psi_kwargs(self, value: dict[str, Any] | None) -> None:
        self._psi_kwargs = value
        self._is_psi_fitted = False

    @property
    def threshold(self) -> float:
        """Sparsity threshold for coefficient masking."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def max_iter(self) -> int:
        """Maximum allowed iterations for the STLS algorithm."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        self._max_iter = value

    @property
    def is_psi_fitted(self) -> bool:
        """Boolean flag indicating if the lifting dictionary is fitted."""
        return self._is_psi_fitted

    @is_psi_fitted.setter
    def is_psi_fitted(self, value: bool) -> None:
        self._is_psi_fitted = value