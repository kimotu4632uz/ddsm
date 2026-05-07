from __future__ import annotations
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.preprocessing import PolynomialFeatures
from ..base._dicts import BaseDict

class IdentityDict(BaseDict):
    dim_: int
    def __init__(self) -> None:
        super(IdentityDict, self).__init__()

    def __len__(self) -> int:
        return self.dim_ + 1 if hasattr(self, 'dim_') else 0

    def lift(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        self.fit(X)
        PsiX = np.hstack((np.ones((X.shape[0], 1)), X))
        return PsiX

    def reconstruct(self, PsiX: np.ndarray) -> np.ndarray:
        PsiX = check_array(PsiX)
        X = PsiX[:, 1:]
        return X

    def diff(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        self.fit(X)
        J = np.zeros((X.shape[0], self.dim_ + 1, self.dim_))
        J[:, 1:, :] = np.eye(self.dim_)
        return J

    def diff2(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        self.fit(X)
        H = np.zeros((X.shape[0], self.dim_ + 1, self.dim_, self.dim_))
        return H

    def fit(self, X: np.ndarray, **kwargs) -> IdentityDict:
        X = check_array(X)
        if not hasattr(self, 'dim_') or X.shape[1] != self.dim_:
            self.dim_ = X.shape[1]
        return self

class MonomialsDict(BaseDict):
    degree: int
    dim_: int
    degrees_: np.ndarray
    def __init__(self, degree: int = 2) -> None:
        super(MonomialsDict, self).__init__()
        self.degree = degree

    def __len__(self) -> int:
        return len(self.degrees_) if hasattr(self, 'degrees_') else 0

    def lift(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        self.fit(X)
        return np.prod(X[:, np.newaxis, :] ** self.degrees_[np.newaxis, :, :], axis=2)

    def reconstruct(self, PsiX: np.ndarray) -> np.ndarray:
        PsiX = check_array(PsiX)
        indices = np.where((self.degrees_.sum(axis=1) == 1) & (self.degrees_.max(axis=1) == 1))[0]
        return PsiX[:, indices]

    def diff(self, X: np.ndarray) -> np.ndarray:
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
        poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        poly.fit(np.zeros((1, self.dim_)))
        return poly.powers_

    @property
    def degrees(self) -> np.ndarray:
        return self.degrees_ if hasattr(self, 'degrees_') else np.array([])