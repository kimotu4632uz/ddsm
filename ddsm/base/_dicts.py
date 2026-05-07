from __future__ import annotations
import abc
import numpy as np

class BaseDict(abc.ABC):
    dim_: int
    def __init__(self) -> None:
        super(BaseDict, self).__init__()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.lift(X)

    @abc.abstractmethod
    def __len__(self) -> int:
        return -1

    @abc.abstractmethod
    def lift(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def reconstruct(self, PsiX: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def diff(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def diff2(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> BaseDict:
        pass