from __future__ import annotations
from typing import Any
import abc
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class DDSMBaseEstimator(abc.ABC, RegressorMixin, BaseEstimator):
    def __init__(self) -> None:
        super(DDSMBaseEstimator, self).__init__()

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> DDSMBaseEstimator:
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __sklearn_tags__(self) -> dict[str, Any]:
        tags = super(DDSMBaseEstimator, self).__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags