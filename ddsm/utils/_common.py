"""汎用 utility 関数を提供するモジュール。"""

import numpy as np

FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-12


def is_close_float(a: float, b: float) -> bool:
    """浮動小数点数が許容誤差内で等しいかを判定する。

    Parameters
    ----------
    a : float
        比較する値。
    b : float
        比較する値。

    Returns
    -------
    bool
        2 つの値が許容誤差内で等しい場合は True。
    """
    return bool(np.isclose(a, b, rtol=FLOAT_RTOL, atol=FLOAT_ATOL))
