from __future__ import annotations
import numpy as np
from sklearn.utils.validation import check_array
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