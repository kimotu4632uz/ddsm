"""SVD と特異値に基づく rank 打ち切りの utility を提供するモジュール。"""

from typing import Literal

import numpy as np
from scipy import linalg

from ._common import is_close_float

SVDTruncationCriterion = Literal["none", "relative", "cumulative", "energy"]


def trunc_svd(
    matrix: np.ndarray,
    *,
    criterion: SVDTruncationCriterion = "none",
    threshold: float | None = None,
    max_rank: int | None = None,
    full_matrices: bool = False,
    overwrite_a: bool = True,
    check_finite: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """行列の SVD を計算し、指定した基準で打ち切る。

    Parameters
    ----------
    matrix : np.ndarray
        SVD を計算する 2 次元配列。
    criterion : SVDTruncationCriterion, optional
        打ち切り基準。``"relative"`` は最大特異値に対する相対値、
        ``"cumulative"`` は特異値の累積和、``"energy"`` は特異値二乗の
        累積和に基づいて rank を決める。
    threshold : float or None, optional
        打ち切り閾値。``None`` の場合は基準による打ち切りを行わない。
    max_rank : int or None, optional
        rank の上限。``None`` の場合は上限なし。
    full_matrices : bool, optional
        ``scipy.linalg.svd`` に渡す ``full_matrices``。
    overwrite_a : bool, optional
        ``scipy.linalg.svd`` に渡す ``overwrite_a``。
    check_finite : bool, optional
        ``scipy.linalg.svd`` に渡す ``check_finite``。

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        打ち切り後の ``(u, s, vt)``。``vt`` は ``scipy.linalg.svd`` と同じ向き。

    Raises
    ------
    ValueError
        入力が 2 次元でない、空行列である、基準が不正、または
        相対打ち切りで最大特異値が 0 に近い場合。
    """
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {array.shape}")

    n_rows, n_cols = array.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("SVD requires a non-empty matrix.")

    u, s, vt = linalg.svd(
        array,
        full_matrices=full_matrices,
        overwrite_a=overwrite_a,
        check_finite=check_finite,
    )
    rank = _truncation_rank(
        s,
        criterion=criterion,
        threshold=threshold,
        max_rank=max_rank,
    )
    return u[:, :rank], s[:rank], vt[:rank, :]


def truncated_pinv_values(
    singular_values: np.ndarray,
    pinv_tol: float | None,
) -> np.ndarray:
    """特異値の閾値付き逆数を返す。

    Parameters
    ----------
    singular_values : np.ndarray
        特異値の 1 次元配列。
    pinv_tol : float or None
        擬似逆で無視する特異値の閾値。``None`` の場合は
        0 でない特異値をすべて使う。

    Returns
    -------
    np.ndarray
        閾値以下の成分を 0 にした特異値の逆数。

    Raises
    ------
    ValueError
        ``singular_values`` が 1 次元でない場合、または
        ``pinv_tol`` が負の場合。
    """
    values = np.asarray(singular_values)
    if values.ndim != 1:
        raise ValueError(f"singular_values must be 1D, got shape {values.shape}")

    if pinv_tol is None:
        mask = values != 0.0
    else:
        tol = float(pinv_tol)
        if tol < 0.0:
            raise ValueError("pinv_tol must be non-negative")
        mask = values > tol

    inv = np.zeros(values.shape, dtype=np.result_type(values, np.float64))
    inv[mask] = 1.0 / values[mask]
    return inv


def _relative_truncation_rank(
    singular_values: np.ndarray,
    threshold: float | None,
) -> int:
    """最大特異値に対する相対閾値で残す rank を返す。

    Parameters
    ----------
    singular_values : np.ndarray
        特異値の 1 次元配列。
    threshold : float or None
        相対閾値。``None`` または 0 以下なら全 rank を残す。

    Returns
    -------
    int
        残す rank。

    Raises
    ------
    ValueError
        特異値が空、または ``threshold`` が正で最大特異値が
        0 に近い場合。
    """
    if singular_values.size == 0:
        raise ValueError("singular_values must be non-empty.")
    if threshold is None or threshold <= 0.0:
        return singular_values.size

    sigma0 = float(singular_values[0])
    if is_close_float(sigma0, 0.0):
        raise ValueError(
            "The largest singular value is close to zero. "
            "`threshold` truncation is not possible."
        )
    return int(np.count_nonzero((singular_values / sigma0) > threshold))


def _cumulative_truncation_rank(
    singular_values: np.ndarray,
    threshold: float | None,
    *,
    squared: bool = False,
) -> int:
    """累積寄与率に基づいて残す rank を返す。

    Parameters
    ----------
    singular_values : np.ndarray
        特異値の 1 次元配列。
    threshold : float or None
        累積寄与率の閾値。``None`` なら全 rank を残す。
    squared : bool, optional
        True の場合は特異値の二乗和で累積寄与率を計算する。

    Returns
    -------
    int
        残す rank。

    Raises
    ------
    ValueError
        特異値が空の場合。
    """
    if singular_values.size == 0:
        raise ValueError("singular_values must be non-empty.")
    if threshold is None:
        return singular_values.size

    weights = singular_values**2 if squared else singular_values
    total = weights[-1] if weights.size == 1 else np.sum(weights)
    if total == 0.0:
        return singular_values.size

    cumulative = np.cumsum(weights) / total
    indices = np.where(cumulative > threshold)[0]
    if indices.size == 0:
        return singular_values.size
    return int(indices[0] + 1)


def _truncation_rank(
    singular_values: np.ndarray,
    *,
    criterion: SVDTruncationCriterion,
    threshold: float | None,
    max_rank: int | None,
) -> int:
    """指定条件から最終的に残す rank を返す。

    Parameters
    ----------
    singular_values : np.ndarray
        特異値の 1 次元配列。
    criterion : SVDTruncationCriterion
        打ち切り基準。
    threshold : float or None
        打ち切り閾値。
    max_rank : int or None
        rank の上限。

    Returns
    -------
    int
        残す rank。

    Raises
    ------
    ValueError
        特異値が空、または打ち切り基準が不正な場合。
    """
    if singular_values.size == 0:
        raise ValueError("SVD returned no singular values.")

    if criterion == "none":
        rank = singular_values.size
    elif criterion == "relative":
        rank = _relative_truncation_rank(singular_values, threshold)
    elif criterion == "cumulative":
        rank = _cumulative_truncation_rank(singular_values, threshold, squared=False)
    elif criterion == "energy":
        rank = _cumulative_truncation_rank(singular_values, threshold, squared=True)
    else:
        raise ValueError(f"unsupported SVD truncation criterion: {criterion}")

    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return rank
