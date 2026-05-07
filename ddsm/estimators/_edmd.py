from __future__ import annotations
from typing import Any, Literal
import numpy as np
from scipy import linalg
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils.validation import validate_data, check_is_fitted
from ..base._dicts import BaseDict
from ..base._estimators import DDSMBaseEstimator
from ..dicts._dicts import IdentityDict

class EDMD(DDSMBaseEstimator):
    psix_: BaseDict
    psiy_: BaseDict
    K_: np.ndarray
    _psix_cls: type[BaseDict]
    _psix_kwargs: dict[str, Any]
    _psiy_cls: type[BaseDict]
    _psiy_kwargs: dict[str, Any]
    _reg: Literal['none', 'lasso', 'ridge']
    _reg_kwargs: dict[str, Any]
    _is_psix_fitted: bool
    _is_psiy_fitted: bool
    _y_ndim_1d: bool
    def __init__(
        self,
        psix_cls: type[BaseDict] = IdentityDict,
        psix_kwargs: dict[str, Any] | None = None,
        psiy_cls: type[BaseDict] = IdentityDict,
        psiy_kwargs: dict[str, Any] | None = None,
        reg:Literal['none', 'lasso', 'ridge'] = 'none',
        reg_kwargs: dict[str, Any] | None = None
    ) -> None:
        super(EDMD, self).__init__()
        self.psix_cls = psix_cls
        self.psix_kwargs = psix_kwargs
        self.psiy_cls = psiy_cls
        self.psiy_kwargs = psiy_kwargs
        self.reg = reg
        self.reg_kwargs = reg_kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> EDMD:
        X, y = validate_data(self, X, y, reset=True, multi_output=True)
        self._y_ndim_1d = (y.ndim == 1)
        if self._y_ndim_1d:
            y = y[:, np.newaxis]
        self._fit_psix(X)
        self._fit_psiy(y)
        PsiX = self.psix_.lift(X)
        Psiy = self.psiy_.lift(y)
        if self.reg == 'none':
            G = PsiX.T @ PsiX
            A = PsiX.T @ Psiy
            self.K_ = linalg.pinv(G) @ A
        elif self.reg == 'lasso' or self.reg == 'ridge':
            reg_kwargs = self.reg_kwargs.copy() if self.reg_kwargs is not None else {}
            reg_kwargs.setdefault('fit_intercept', False)
            reg = Lasso(**reg_kwargs) if self.reg == 'lasso' else Ridge(**reg_kwargs)
            reg.fit(PsiX, Psiy)
            self.K_ = np.atleast_2d(reg.coef_).T

        else:
            raise ValueError(f'Invalid regularization type: {self.reg}. Must be one of "none", "lasso", or "ridge".')
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'K_')
        X = validate_data(self, X, reset=False)
        PsiX = self.psix_.lift(X)
        Psiy = PsiX @ self.K_
        y = self.psiy_.reconstruct(Psiy)
        if hasattr(self, '_y_ndim_1d') and self._y_ndim_1d:
            y = y.ravel()
        return y

    def _fit_psix(self, X: np.ndarray) -> None:
        if not hasattr(self, 'psix_') or not self._is_psix_fitted:
            self.psix_ = self.psix_cls(**self.psix_kwargs.copy() if self.psix_kwargs is not None else {})
            self.psix_.fit(X)
            self._is_psix_fitted = True

    def _fit_psiy(self, y: np.ndarray) -> None:
        if not hasattr(self, 'psiy_') or not self._is_psiy_fitted:
            self.psiy_ = self.psiy_cls(**self.psiy_kwargs.copy() if self.psiy_kwargs is not None else {})
            self.psiy_.fit(y)
            self._is_psiy_fitted = True

    @property
    def right_K(self) -> np.ndarray:
        check_is_fitted(self, 'K_')
        return self.K_

    @property
    def left_K(self) -> np.ndarray:
        return self.K_.conj().T

    def right_L(self, dt: float) -> np.ndarray:
        return linalg.logm(self.K_) / dt

    def left_L(self, dt: float) -> np.ndarray:
        return self.right_L(dt).conj().T

    @property
    def psix_cls(self) -> type[BaseDict]:
        return self._psix_cls

    @psix_cls.setter
    def psix_cls(self, value: type[BaseDict]) -> None:
        self._psix_cls = value
        self._is_psix_fitted = False

    @property
    def psix_kwargs(self) -> dict[str, Any]:
        return self._psix_kwargs

    @psix_kwargs.setter
    def psix_kwargs(self, value: dict[str, Any] | None) -> None:
        self._psix_kwargs = value
        self._is_psix_fitted = False

    @property
    def psiy_cls(self) -> type[BaseDict]:
        return self._psiy_cls

    @psiy_cls.setter
    def psiy_cls(self, value: type[BaseDict]) -> None:
        self._psiy_cls = value
        self._is_psiy_fitted = False

    @property
    def psiy_kwargs(self) -> dict[str, Any]:
        return self._psiy_kwargs

    @psiy_kwargs.setter
    def psiy_kwargs(self, value: dict[str, Any] | None) -> None:
        self._psiy_kwargs = value
        self._is_psiy_fitted = False

    @property
    def reg(self) -> Literal['none', 'lasso', 'ridge']:
        return self._reg

    @reg.setter
    def reg(self, value: Literal['none', 'lasso', 'ridge']) -> None:
        self._reg = value

    @property
    def reg_kwargs(self) -> dict[str, Any]:
        return self._reg_kwargs

    @reg_kwargs.setter
    def reg_kwargs(self, value: dict[str, Any] | None) -> None:
        self._reg_kwargs = value

    @property
    def is_psix_fitted(self) -> bool:
        return self._is_psix_fitted

    @is_psix_fitted.setter
    def is_psix_fitted(self, value: bool) -> None:
        self._is_psix_fitted = value

    @property
    def is_psiy_fitted(self) -> bool:
        return self._is_psiy_fitted

    @is_psiy_fitted.setter
    def is_psiy_fitted(self, value: bool) -> None:
        self._is_psiy_fitted = value