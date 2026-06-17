"""SymPy 記号配列を扱う utility を提供するモジュール。"""

import itertools

import numpy as np
import sympy as sp


def symarray(prefix: str, shape: tuple[int, ...], **kwargs: dict) -> np.ndarray:
    """指定した形状の SymPy 記号配列を生成する。

    Parameters
    ----------
    prefix : str
        記号名の接頭辞。
    shape : tuple[int, ...]
        生成する配列の形状。
    **kwargs : dict
        ``sympy.Symbol`` に渡す追加引数。

    Returns
    -------
    np.ndarray
        指定した形状に整形した SymPy 記号配列。
    """
    ranges = [range(s) for s in shape]

    flat_syms = [
        sp.Symbol(f"{prefix}_{','.join(map(str, idx))}", **kwargs)
        for idx in itertools.product(*ranges)
    ]

    return np.array(flat_syms, dtype=object).reshape(shape)
