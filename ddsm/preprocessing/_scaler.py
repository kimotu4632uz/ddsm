"""データを 10 のべき乗でスケーリングする sklearn 互換 transformer を提供するモジュール。"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


def _power_of_ten_factor(max_abs: float) -> float:
    """最大絶対値を 1 以下に収める 10 のべき乗の係数を返す。

    Parameters
    ----------
    max_abs : float
        対象データの最大絶対値。

    Returns
    -------
    float
        ``max_abs`` が 1 以下なら 1.0、そうでなければ整数部の桁数 ``n`` に対する
        ``10 ** -n``。
    """
    if max_abs <= 1.0:
        return 1.0
    n_digits = len(str(int(max_abs)))
    return float(np.power(10.0, -float(n_digits)))


class PowerOfTenScaler(TransformerMixin, BaseEstimator):
    """データを 10 のべき乗でスケーリングする transformer。

    各特徴量(``per_feature=True``)または全体(``per_feature=False``)の最大絶対値を
    1 以下に収める 10 のべき乗を係数として求め、データに掛けてスケーリングする。
    係数を 10 のべき乗に限定することで逆変換が厳密になり、Koopman/生成子行列の
    スケール戻しでも復元係数の解釈性が保たれる。

    Parameters
    ----------
    per_feature : bool, optional
        True の場合は特徴量ごとに係数を求める。False の場合は全要素から
        単一の係数を求めて全特徴量に適用する。

    Attributes
    ----------
    scale_ : np.ndarray
        各特徴量に掛けるスケーリング係数。形状は ``(n_features,)``。
    n_features_in_ : int
        ``fit`` に渡された特徴量数。
    """

    def __init__(self, *, per_feature: bool = True) -> None:
        self.per_feature = per_feature

    def fit(self, X: np.ndarray, y: object = None) -> "PowerOfTenScaler":
        """スケーリング係数を学習する。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            スケーリング係数の計算に用いるデータ。
        y : Ignored
            sklearn API との互換のために存在する。使用しない。

        Returns
        -------
        PowerOfTenScaler
            学習済みの自身。
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)

        if self.per_feature:
            max_abs = np.max(np.abs(X), axis=0)
            scale = np.array([_power_of_ten_factor(float(m)) for m in max_abs])
        else:
            factor = _power_of_ten_factor(float(np.max(np.abs(X))))
            scale = np.full(X.shape[1], factor)

        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """データにスケーリング係数を掛ける。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            スケーリングするデータ。

        Returns
        -------
        np.ndarray
            スケーリング後のデータ。

        Raises
        ------
        ValueError
            特徴量数が ``fit`` 時と異なる場合。
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return X * self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """スケーリングを元に戻す。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            スケーリング済みのデータ。

        Returns
        -------
        np.ndarray
            元のスケールに戻したデータ。

        Raises
        ------
        ValueError
            特徴量数が ``fit`` 時と異なる場合。
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return X / self.scale_
