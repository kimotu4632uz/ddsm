"""スケール座標で同定した Koopman/生成子行列を元座標へ戻すユーティリティ。"""

from __future__ import annotations

import numpy as np


def unscale_koopman(
    matrix: np.ndarray,
    scale: np.ndarray | float,
    degrees: np.ndarray,
) -> np.ndarray:
    """単項式基底の Koopman/生成子行列をスケール座標から元座標へ戻す。

    スケール変数 ``y_k = scale_k * x_k`` の下で同定した行列を、元の座標 ``x`` に
    対応する行列へ相似変換 ``D^{-1} A D`` で戻す。``D_i = prod_k scale_k ** degrees[i, k]``
    は単項式基底の座標変換に対応する対角行列であり、全次元共通の係数(スカラー)でも
    次元ごとに異なる係数(``PowerOfTenScaler.scale_``)でも正しく戻せる。

    Parameters
    ----------
    matrix : np.ndarray
        スケール座標で同定した Koopman 作用素または生成子行列。形状は ``(N, N)``。
    scale : np.ndarray or float
        データのスケーリング係数。``PowerOfTenScaler.scale_`` を渡す。
        スカラーを渡した場合は全次元共通として扱う。
    degrees : np.ndarray
        各基底関数の多重次数。形状は ``(N, dim)``。``MonomialsDict.degrees`` を渡す。

    Returns
    -------
    np.ndarray
        元座標に対応する Koopman/生成子行列。

    Raises
    ------
    ValueError
        ``matrix`` が正方行列でない、または行数が ``degrees`` の行数と一致しない場合。
    """
    matrix = np.asarray(matrix, dtype=float)
    degrees = np.asarray(degrees)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    if matrix.shape[0] != degrees.shape[0]:
        raise ValueError("matrix size must match the number of basis functions in degrees")

    scale = np.asarray(scale, dtype=float)
    if scale.ndim == 0:
        D = scale ** degrees.sum(axis=1)
    else:
        D = np.prod(scale ** degrees, axis=1)

    return matrix * (D[None, :] / D[:, None])
