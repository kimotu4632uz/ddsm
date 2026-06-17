"""ODE/SDE モデルの共通基底クラスを提供するモジュール。"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import numpy as np
import sympy as sp


class BaseModel(metaclass=ABCMeta):
    """sympyで定義された力学モデルの基底クラス。

    Parameters
    ----------
    name : str
        モデル名。
    sym_xs : np.ndarray
        状態変数の symbol 配列。
    drifts : np.ndarray
        ドリフト項の symbol 式配列。
    param_pairs : dict[sp.Symbol, int | float]
        パラメータの symbol と数値の対応。

    Raises
    ------
    TypeError
        パラメータ対応の型が不正な場合。
    ValueError
        モデル式の形状が不正な場合。
    """

    name: str
    _dim: int
    _sym_xs: np.ndarray
    _drifts: np.ndarray
    _param_pairs: dict[sp.Symbol, int | float]

    def __init__(
        self,
        name: str,
        sym_xs: np.ndarray,
        drifts: np.ndarray,
        param_pairs: dict[sp.Symbol, int | float],
    ) -> None:
        """モデルを初期化する。

        Parameters
        ----------
        name : str
            モデル名。
        sym_xs : np.ndarray
            状態変数の SymPy 記号配列。
        drifts : np.ndarray
            ドリフト項の記号式配列。
        param_pairs : dict[sp.Symbol, int | float]
            パラメータ記号と数値の対応。

        Raises
        ------
        TypeError
            パラメータ対応の型が不正な場合。
        ValueError
            モデル式の形状が不正な場合。
        """
        self.name = name
        self._sym_xs = np.array(sym_xs, copy=True)
        self._drifts = np.array(drifts, copy=True)
        self._param_pairs = dict(param_pairs)
        self._dim = int(self._sym_xs.shape[0])

        if self._sym_xs.shape != (self._dim,):
            raise ValueError("`sym_xs` must have shape `(dim,)`.")
        if self._drifts.shape != (self._dim,):
            raise ValueError("`drifts` must have shape `(dim,)`.")
        if not isinstance(self._param_pairs, dict):
            raise TypeError("`param_pairs` must be a dict.")

    @property
    @abstractmethod
    def is_ode(self) -> bool:
        """モデルが ODE かどうか。"""
        ...

    @property
    def dim(self) -> int:
        """状態空間の次元。"""
        return self._dim

    @property
    def sym_xs(self) -> np.ndarray:
        """状態変数の SymPy 記号配列。"""
        return np.array(self._sym_xs, copy=True)

    @property
    def drifts(self) -> np.ndarray:
        """ドリフト項の記号式配列。"""
        return np.array(self._drifts, copy=True)

    @property
    def param_pairs(self) -> dict[sp.Symbol, int | float]:
        """パラメータ記号と数値の対応。"""
        return dict(self._param_pairs)

    @abstractmethod
    def build_adjoint_operator(self, sym_derivs: np.ndarray) -> sp.Expr:
        """Backward-Kolmogorov 方程式の作用素を構築する。

        Parameters
        ----------
        sym_derivs : np.ndarray
            各状態変数に対応する偏微分演算子の記号配列。

        Returns
        -------
        sp.Expr
            ドリフト項と拡散項から構成した随伴作用素の symbol 式。
        """
        ...

    def get_drifts_func(self) -> Callable:
        """ドリフト項を評価する NumPy 関数を返す。

        Returns
        -------
        Callable
            状態ベクトルを受け取り、パラメータ代入済みのドリフト項を返す関数。
        """
        args = ([self._sym_xs])
        sym_func = sp.Matrix(self._drifts).subs(self._param_pairs)
        return sp.lambdify(args, sym_func, "numpy")

    def print_model(self) -> None:
        """パラメータ記号を残したモデル式を LaTeX 表示する。"""
        from IPython.display import Math, display

        print(self.name, "(ODE)" if self.is_ode else "(SDE)")

        sp.init_printing()

        for d1 in range(self._dim):
            display(Math(rf'd{sp.latex(self._sym_xs[d1])}(t) = \left({sp.latex(self._drifts[d1])}\right) dt'))

    def print_subs_model(self) -> None:
        """パラメータを数値に代入したモデル式を LaTeX 表示する。"""
        from IPython.display import Math, display

        print(self.name, "(ODE)" if self.is_ode else "(SDE)")

        sp.init_printing()

        for d1 in range(self._dim):
            drift = self._drifts[d1]
            if isinstance(drift, sp.Expr):
                drift = drift.subs(self._param_pairs)

            display(Math(rf'd{sp.latex(self._sym_xs[d1])}(t) = \left({sp.latex(drift)}\right) dt'))


class ODEModel(BaseModel):
    """ODE モデルを表す基底クラス。"""

    def __init__(
        self,
        name: str,
        sym_xs: np.ndarray,
        drifts: np.ndarray,
        param_pairs: dict[sp.Symbol, int | float],
    ) -> None:
        """ODE モデルを初期化する。

        Parameters
        ----------
        name : str
            モデル名。
        sym_xs : np.ndarray
            状態変数の symbol 配列。
        drifts : np.ndarray
            ドリフト項の symbol 式配列。
        param_pairs : dict[sp.Symbol, int | float]
            パラメータの symbol と数値の対応。

        Raises
        ------
        TypeError
            パラメータ対応の型が不正な場合。
        ValueError
            モデル式の形状が不正な場合。
        """
        super().__init__(
            name=name,
            sym_xs=sym_xs,
            drifts=drifts,
            param_pairs=param_pairs,
        )

    @property
    def is_ode(self) -> bool:
        """モデルが ODE かどうか。"""
        return True

    def build_adjoint_operator(self, sym_derivs: np.ndarray) -> sp.Expr:
        """Backward-Kolmogorov 方程式の作用素を構築する。

        Parameters
        ----------
        sym_derivs : np.ndarray
            各状態変数に対応する偏微分演算子の記号配列。

        Returns
        -------
        sp.Expr
            ドリフト項から構成した随伴作用素の記号式。
        """
        drift_term = np.dot(self._drifts, sym_derivs)
        return sp.expand(drift_term)


class SDEModel(BaseModel):
    """SDE モデルを表す基底クラス。"""

    _diffs: np.ndarray

    def __init__(
        self,
        name: str,
        sym_xs: np.ndarray,
        drifts: np.ndarray,
        diffs: np.ndarray,
        param_pairs: dict[sp.Symbol, int | float],
    ) -> None:
        """SDE モデルを初期化する。

        Parameters
        ----------
        name : str
            モデル名。
        sym_xs : np.ndarray
            状態変数の symbol 配列。
        drifts : np.ndarray
            ドリフト項の symbol 式配列。
        diffs : np.ndarray
            拡散項の symbol 式行列。
        param_pairs : dict[sp.Symbol, int | float]
            パラメータの symbol と数値の対応。

        Raises
        ------
        TypeError
            パラメータ対応の型が不正な場合。
        ValueError
            モデル式の形状が不正な場合。
        """
        super().__init__(
            name=name,
            sym_xs=sym_xs,
            drifts=drifts,
            param_pairs=param_pairs,
        )
        self._diffs = np.array(diffs, copy=True)
        if self._diffs.shape != (self._dim, self._dim):
            raise ValueError("`diffs` must have shape `(dim, dim)`.")

    @property
    def is_ode(self) -> bool:
        """モデルが ODE かどうか。"""
        return False

    @property
    def diffs(self) -> np.ndarray:
        """拡散項の symbol 式行列。"""
        return np.array(self._diffs, copy=True)

    def get_diffs_func(self) -> Callable:
        """拡散項を評価する NumPy 関数を返す。"""
        args = ([self._sym_xs])
        sym_func = sp.Matrix(self._diffs).subs(self._param_pairs)
        return sp.lambdify(args, sym_func, "numpy")

    def build_adjoint_operator(self, sym_derivs: np.ndarray) -> sp.Expr:
        """Backward-Kolmogorov 方程式の作用素を構築する。

        Parameters
        ----------
        sym_derivs : np.ndarray
            各状態変数に対応する偏微分演算子の記号配列。

        Returns
        -------
        sp.Expr
            ドリフト項と拡散項から構成した随伴作用素の記号式。
        """
        drift_term = np.dot(self._drifts, sym_derivs)
        B = sp.Rational(1, 2) * (self._diffs @ self._diffs.T)
        diff_term = sym_derivs @ B @ sym_derivs
        return sp.expand(drift_term + diff_term)

    def print_model(self) -> None:
        """パラメータ記号を残したモデル式を LaTeX 表示する。"""
        from IPython.display import Math, display

        print(self.name, "(SDE)")

        sp.init_printing()

        for d1 in range(self._dim):
            latex_txt = rf"d{sp.latex(self._sym_xs[d1])}(t) = \left({sp.latex(self._drifts[d1])}\right) dt"
            for d2 in range(self._dim):
                if self._diffs[d1][d2] != 0:
                    latex_txt += rf"+ \left({sp.latex(self._diffs[d1][d2])}\right) dW_{{{d2+1}}}"
            display(Math(latex_txt))

    def print_subs_model(self) -> None:
        """パラメータを数値に代入したモデル式を LaTeX 表示する。"""
        from IPython.display import Math, display

        print(self.name, "(SDE)")

        sp.init_printing()

        for d1 in range(self._dim):
            drift = self._drifts[d1]
            if isinstance(drift, sp.Expr):
                drift = drift.subs(self._param_pairs)

            latex_txt = rf"d{sp.latex(self._sym_xs[d1])}(t) = \left({sp.latex(drift)}\right) dt"
            for d2 in range(self._dim):
                diff = self._diffs[d1][d2]
                if isinstance(diff, sp.Expr):
                    diff = diff.subs(self._param_pairs)
                if diff != 0:
                    latex_txt += rf"+ \left({sp.latex(diff)}\right) dW_{{{d2+1}}}"
            display(Math(latex_txt))
