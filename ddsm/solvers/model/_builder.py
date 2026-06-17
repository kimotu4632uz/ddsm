"""記号式ベースのモデル方程式を構築するモジュール。"""

from numbers import Real

import numpy as np
import sympy as sp

from ...utils.sympy import symarray
from ._base import ODEModel, SDEModel


class EquationBuilder:
    """ODEModel, SDEModel を構築する builder。"""

    _sym_xs: np.ndarray | None
    _param_pairs: dict[sp.Symbol, int | float]

    def __init__(self) -> None:
        """builder を初期化する。"""
        self._sym_xs = None
        self._param_pairs: dict[sp.Symbol, int | float] = {}

    def create_state_symbols(self, dim: int) -> np.ndarray:
        """状態変数の symbol 配列を生成する。

        Parameters
        ----------
        dim : int
            状態空間の次元。

        Returns
        -------
        np.ndarray
            長さ ``dim`` の状態変数 symbol 配列。
        """
        if self._sym_xs is not None:
            raise ValueError("State symbols have already been created.")

        self._sym_xs = symarray("x", (dim,))
        return np.array(self._sym_xs, copy=True)

    def create_param_symbol(self, name: str, value: int | float) -> sp.Symbol:
        """スカラー値のパラメータ記号を生成する。

        Parameters
        ----------
        name : str
            パラメータ名。symbol の生成にも使われます。
        value : int | float
            パラメータ値。

        Returns
        -------
        sp.Symbol
            パラメータに対応する symbol。

        Raises
        ------
        TypeError
            パラメータが数値型でない場合。
        """
        if isinstance(value, Real):
            sym_param = sp.Symbol(name)
            self._param_pairs[sym_param] = value
            return sym_param

        raise TypeError(
            f"変数{name}は{type(value)}型ですが，param_symbolは数値型のみ対応しています。"
        )

    def create_param_symbol_list(
        self,
        name: str,
        values: list | tuple | np.ndarray,
    ) -> np.ndarray:
        """配列値のパラメータ symbol を生成する。

        Parameters
        ----------
        name : str
            パラメータ名。symbol の生成にも使われます。
        values : list | tuple | np.ndarray
            パラメータ配列。

        Returns
        -------
        np.ndarray
            パラメータ配列と同じ形状の symbol 配列。

        Raises
        ------
        TypeError
            パラメータが配列型でない場合。
        """
        if isinstance(values, (list, tuple, np.ndarray)):
            val_array = np.array(values)
            sym_array = symarray(name, val_array.shape)
            self._param_pairs.update(dict(zip(sym_array.flat, val_array.flat)))
            return sym_array

        raise TypeError(
            f"変数{name}は{type(values)}型ですが，param_symbol_listは配列型のみ対応しています。"
        )

    def build_ode(self, *, name: str, drifts: np.ndarray) -> "ODEModel":
        """ODE モデルを構築する。

        Parameters
        ----------
        name : str
            モデル名。
        drifts : np.ndarray
            ドリフト項の symbol 式配列。

        Returns
        -------
        ODEModel
            構築した ODE モデル。

        Raises
        ------
        ValueError
            状態変数の symbol 配列が未作成の場合。
        """

        if self._sym_xs is None:
            raise ValueError("`create_state_symbols` must be called before building a model.")

        return ODEModel(
            name=name,
            sym_xs=self._sym_xs,
            drifts=drifts,
            param_pairs=self._param_pairs,
        )

    def build_sde(
        self,
        *,
        name: str,
        drifts: np.ndarray,
        diffs: np.ndarray,
    ) -> "SDEModel":
        """SDE モデルを構築する。

        Parameters
        ----------
        name : str
            モデル名。
        drifts : np.ndarray
            ドリフト項の symbol 式配列。
        diffs : np.ndarray
            拡散項の symbol 式行列。

        Returns
        -------
        SDEModel
            構築した SDE モデル。

        Raises
        ------
        ValueError
            状態変数の symbol 配列が未作成の場合。
        """
        if self._sym_xs is None:
            raise ValueError("`create_state_symbols` must be called before building a model.")

        return SDEModel(
            name=name,
            sym_xs=self._sym_xs,
            drifts=drifts,
            diffs=diffs,
            param_pairs=self._param_pairs,
        )
