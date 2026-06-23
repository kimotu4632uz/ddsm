from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from scipy import linalg
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils.validation import check_is_fitted, validate_data

from ..base._dicts import BaseDict
from ..base._estimators import DDSMBaseEstimator
from ..dicts._dicts import IdentityDict


class gEDMD(DDSMBaseEstimator):
    """
    Generator Extended Dynamic Mode Decomposition (gEDMD) estimator.

    Computes the continuous-time Koopman generator directly from data pairs
    `(X, \\dot{X})` using the chain rule and dictionary derivatives.

    Parameters
    ----------
    psi_cls : type[BaseDict], default=IdentityDict
        Class type for the lifting dictionary.
    psi_kwargs : dict, optional
        Arguments for initializing `psi_cls`.
    reg : {'none', 'lasso', 'ridge'}, default='none'
        Regularization type for computing the generator matrix.
    reg_kwargs : dict, optional
        Arguments passed to the underlying Scikit-learn regressor.

    Attributes
    ----------
    psi_ : BaseDict
        Fitted lifting dictionary instance.
    L_ : ndarray
        The estimated Koopman generator matrix.
    """
    psi_: BaseDict
    L_: np.ndarray
    _psi_cls: type[BaseDict]
    _psi_kwargs: dict[str, Any]
    _reg: Literal['none', 'lasso', 'ridge']
    _reg_kwargs: dict[str, Any]
    _is_psi_fitted: bool
    _y_ndim_1d: bool
    _y_features: int

    def __init__(
        self,
        psi_cls: type[BaseDict] = IdentityDict,
        psi_kwargs: dict[str, Any] | None = None,
        reg:Literal['none', 'lasso', 'ridge'] = 'none',
        reg_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Initialize the gEDMD estimator."""
        super(gEDMD, self).__init__()
        self.psi_cls = psi_cls
        self.psi_kwargs = psi_kwargs
        self.reg = reg
        self.reg_kwargs = reg_kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> gEDMD:
        """
        Fit the gEDMD model to estimate the generator L.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            State at current time steps.
        y : ndarray of shape (n_samples, n_features)
            Time derivatives of the state (`\\dot{X}`).

        Returns
        -------
        gEDMD
            The fitted instance.
        """
        X, y = validate_data(self, X, y, reset=True, multi_output=True)
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
        PsiX = self.psi_.lift(X)
        dPsiX = np.sum(self.psi_.diff(X) * y[:, np.newaxis, :], axis=2)
        if self.reg == 'none':
            G = PsiX.T @ PsiX
            A = PsiX.T @ dPsiX
            self.L_ = linalg.pinv(G) @ A
        elif self.reg == 'lasso' or self.reg == 'ridge':
            reg_kwargs = self.reg_kwargs.copy() if self.reg_kwargs is not None else {}
            reg_kwargs.setdefault('fit_intercept', False)
            reg = Lasso(**reg_kwargs) if self.reg == 'lasso' else Ridge(**reg_kwargs)
            reg.fit(PsiX, dPsiX)
            self.L_ = np.atleast_2d(reg.coef_).T

        else:
            raise ValueError(f'Invalid regularization type: {self.reg}. Must be one of "none", "lasso", or "ridge".')
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the time derivative dot{X} given state X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Current state.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Predicted time derivative.
        """
        check_is_fitted(self, 'L_')
        X = validate_data(self, X, reset=False)
        PsiX = self.psi_.lift(X)
        dPsiX = PsiX @ self.L_
        y = self.psi_.reconstruct(dPsiX)
        if hasattr(self, '_y_features') and y.shape[1] != self._y_features:
            y = y[:, :self._y_features]
        if hasattr(self, '_y_ndim_1d') and self._y_ndim_1d:
            y = y.ravel()
        return y

    def _fit_psi(self, X: np.ndarray) -> None:
        """
        Initialize and fit the lifting dictionary (Psi).

        Parameters
        ----------
        X : ndarray
            Training data used to determine dictionary parameters.
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

    def right_K(self, dt: float) -> np.ndarray:
        """
        Compute the discrete-time Koopman operator for a step dt.

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        ndarray
            Matrix exponential of (L * dt).
        """
        return linalg.expm(self.right_L * dt)

    def left_K(self, dt: float) -> np.ndarray:
        """
        Compute the adjoint discrete-time Koopman operator.

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        ndarray
            Adjoint of the right Koopman operator.
        """
        return self.right_K(dt).conj().T

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
    def reg(self) -> Literal['none', 'lasso', 'ridge']:
        """Regularization method: 'none', 'lasso', or 'ridge'."""
        return self._reg

    @reg.setter
    def reg(self, value: Literal['none', 'lasso', 'ridge']) -> None:
        self._reg = value

    @property
    def reg_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the regularization estimator."""
        return self._reg_kwargs

    @reg_kwargs.setter
    def reg_kwargs(self, value: dict[str, Any] | None) -> None:
        self._reg_kwargs = value

    @property
    def is_psi_fitted(self) -> bool:
        """Boolean flag indicating if the lifting dictionary is fitted."""
        return self._is_psi_fitted

    @is_psi_fitted.setter
    def is_psi_fitted(self, value: bool) -> None:
        self._is_psi_fitted = value