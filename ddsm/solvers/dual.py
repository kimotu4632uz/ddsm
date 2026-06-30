"""双対 Koopman 方程式に基づく solver を提供するモジュール。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import linalg

from ..dicts.monomials import BaseMonomials, KronMonomialsDict, MonomialBasisKind
from .model import BaseModel
from .time_grid import TimeGrid


@dataclass(slots=True, eq=False)
class StatsTrajectory:
    """時刻ごとの統計量と内部状態列を表す。

    Attributes
    ----------
    time_grid : TimeGrid
        時間グリッド。
    values : np.ndarray
        観測時刻ごとの統計量。
    states : Sequence[np.ndarray]
        観測時刻ごとの内部状態列。

    Raises
    ------
    ValueError
        統計量または状態列の形状が不正な場合。
    """

    time_grid: TimeGrid
    values: np.ndarray
    states: Sequence[np.ndarray]

    def __post_init__(self) -> None:
        """統計量と内部状態列の形状を検証する。

        Raises
        ------
        ValueError
            統計量または状態列の長さが観測時刻列と一致しない場合。
        """
        self.values = np.asarray(self.values).copy()
        self.states = tuple(value.copy() for value in self.states)

        if self.values.ndim != 1:
            raise ValueError("stats must be a 1D array")
        if self.values.shape[0] != len(self.time_grid.obs_times) - 1:
            raise ValueError("values length must match len(time_grid.obs_times) - 1")
        if len(self.states) != len(self.time_grid.obs_times) - 1:
            raise ValueError("states length must match len(time_grid.obs_times) - 1")

    @property
    def obs_times(self) -> np.ndarray:
        """観測時刻列。"""
        return self.time_grid.obs_times


def _iter_operator_terms(
    model: BaseModel,
    sym_derivs: np.ndarray,
) -> Sequence[tuple[float, np.ndarray, np.ndarray]]:
    """随伴作用素の各単項式項を係数、状態次数、微分次数に分解する。

    Parameters
    ----------
    model : BaseModel
        対象の SDE モデル。
    sym_derivs : np.ndarray
        各状態変数に対応する偏微分演算子の記号配列。

    Returns
    -------
    Sequence[tuple[float, np.ndarray, np.ndarray]]
        ``(係数, 状態変数の次数, 微分演算子の次数)`` の列。

    Raises
    ------
    ValueError
        随伴作用素が多項式として扱えない場合。
    """
    L = model.build_adjoint_operator(sym_derivs)

    variables: list[sp.Symbol] = []
    variables.extend(model.sym_xs.flat)
    variables.extend(sym_derivs.flat)

    terms: list[tuple[float, np.ndarray, np.ndarray]] = []
    expand = sp.expand(L.subs(model.param_pairs))
    for arg in sp.Add.make_args(expand):
        if arg == 0:
            continue

        poly = sp.Poly(arg, *variables)
        poly_terms = poly.terms()
        if len(poly_terms) != 1:
            raise ValueError("adjoint operator must be polynomial in state and derivative symbols")

        powers, coeff = poly_terms[0]
        degree_list = np.array(powers, dtype=int)
        terms.append(
            (
                float(coeff),
                degree_list[:model.dim],
                degree_list[model.dim:],
            )
        )

    return tuple(terms)


def _create_generator_kron(model: BaseModel, psi: KronMonomialsDict) -> np.ndarray:
    """テンソル積単項式辞書用の生成行列をクロネッカー積で作成する。

    Parameters
    ----------
    model : BaseModel
        対象の SDE モデル。
    psi : KronMonomialsDict
        テンソル積単項式辞書。

    Returns
    -------
    np.ndarray
        生成行列。
    """
    sym_derivs = sp.symarray(r'\partial', (model.dim,))

    n_trunc = psi.degree + 1
    A = np.zeros((len(psi), len(psi)), dtype='float64')

    for coeff, state_power, state_rate in _iter_operator_terms(model, sym_derivs):
        state_change = state_power - state_rate # 状態変化のベクトルを得る
        # 微分演算子の回数に応じて処理を変える（定数項、ドリフト項、拡散項）
        op = 0.0
        for d in range(model.dim):
            if state_rate[d] == 0: # 定数項
                comp = np.eye(n_trunc,k=-state_change[d])
            elif state_rate[d] == 1: # ドリフト項
                comp = np.eye(n_trunc,k=-state_change[d])@np.diag(np.arange(n_trunc))
            else: # 拡散項
                comp = np.eye(n_trunc,k=-state_change[d])@np.diag(np.arange(n_trunc)*(np.arange(n_trunc)-1))
            if d == 0: # 最初は自分自身のみ
                op = coeff*comp
            else: # 2次元目以降はクロネッカー積
                if psi.basis_kind is MonomialBasisKind.KRON_RIGHT:
                    op = np.kron(comp, op)
                elif psi.basis_kind is MonomialBasisKind.KRON_LEFT:
                    op = np.kron(op, comp)
                else:
                    raise ValueError("create_generator requires a tensor-product monomial basis")
        A = A + op

    A = np.array(A, dtype='float64')
    return A


def _create_generator_projected(model: BaseModel, psi: BaseMonomials) -> np.ndarray:
    """単項式辞書へ射影した生成行列を次数ベクトルから作成する。

    Parameters
    ----------
    model : BaseModel
        対象の SDE モデル。
    psi : BaseMonomials
        単項式辞書。

    Returns
    -------
    np.ndarray
        生成行列。
    """
    sym_derivs = sp.symarray(r'\partial', (model.dim,))
    n_eqs = len(psi)
    A = np.zeros((n_eqs, n_eqs), dtype='float64')
    degree_to_index = {
        tuple(psi.i2s(i).astype(int).tolist()): i
        for i in range(n_eqs)
    }

    for coeff, state_power, state_rate in _iter_operator_terms(model, sym_derivs):
        for source_index in range(n_eqs):
            source_degree = psi.i2s(source_index).astype(int)
            if np.any(source_degree < state_rate):
                continue

            factor = coeff
            for d in range(model.dim):
                for i in range(int(state_rate[d])):
                    factor *= int(source_degree[d]) - i

            target_degree = source_degree - state_rate + state_power
            target_index = degree_to_index.get(tuple(target_degree.tolist()))
            if target_index is None:
                continue
            A[target_index, source_index] += factor

    return A


def create_generator(model: BaseModel, psi: BaseMonomials) -> np.ndarray:
    """SDE の随伴作用素から有限次元の生成行列を作成する。

    Parameters
    ----------
    model : BaseModel
        対象の SDE モデル。
    psi : BaseMonomials
        単項式辞書。

    Returns
    -------
    np.ndarray
        生成行列。

    Raises
    ------
    ValueError
        ``psi`` の次元がモデルと一致しない場合。
    """
    if psi.dim_ != model.dim:
        raise ValueError("Dual.create_generator: psi dimension does not match model dimension")

    if isinstance(psi, KronMonomialsDict):
        return _create_generator_kron(model, psi)
    else:
        return _create_generator_projected(model, psi)


def solve_dual_ode(
    *,
    model: BaseModel,
    psi: BaseMonomials,
    time_grid: TimeGrid,
    comp_index: list[int]
) -> Sequence[np.ndarray]:
    """指定した単項式成分の双対方程式を解く。

    Parameters
    ----------
    model : BaseModel
        対象の SDE モデル。
    psi : BaseMonomials
        単項式辞書。
    time_grid : TimeGrid
        時間グリッド。
    comp_index : list[int]
        初期条件として 1 を置く単項式の次数ベクトル。

    Returns
    -------
    Sequence[np.ndarray]
        観測時刻ごとの係数ベクトル。
    """
    A = create_generator(
        model = model,
        psi=psi
    )
    n_eqs = A.shape[0]

    p_ini = np.zeros(n_eqs)
    p_ini[psi.s2i(comp_index)] = 1.0

    p_list: list[np.ndarray] = []
    sol = p_ini
    dt = time_grid.dt
    for step in time_grid.iter_steps():
        leftA = np.eye(n_eqs) - 0.5*dt*A
        right_b = (np.eye(n_eqs) + 0.5*dt*A)@sol
        sol = linalg.solve(leftA, right_b)

        if step.should_save:
            p_list.append(sol.copy())

    return p_list


def solve_dual(
    *,
    model: BaseModel,
    psi: BaseMonomials,
    x0: np.ndarray,
    time_grid: TimeGrid,
    comp_index: list[int]
) -> StatsTrajectory:
    """初期状態に対する指定成分の統計量を計算する。

    Parameters
    ----------
    model : BaseModel
        対象の SDE モデル。
    psi : BaseMonomials
        単項式辞書。
    x0 : np.ndarray
        初期状態。
    time_grid : TimeGrid
        時間グリッド。
    comp_index : list[int]
        評価する単項式の次数ベクトル。

    Returns
    -------
    StatsTrajectory
        観測時刻ごとの統計量と係数ベクトル列。
    """
    p_list = solve_dual_ode(model=model, psi=psi, time_grid=time_grid, comp_index=comp_index)

    lifted_x0 = psi.lift(x0[np.newaxis, :]).reshape(-1)
    stats = [np.dot(p, lifted_x0) for p in p_list]

    return StatsTrajectory(time_grid=time_grid, values=np.asarray(stats), states=p_list)


def create_koopman(*, model: BaseModel, psi: BaseMonomials, time_grid: TimeGrid) -> np.ndarray:
    """Koopman 行列を計算する。

    Parameters
    ----------
    model : BaseModel
        対象の SDE モデル。
    psi : BaseMonomials
        単項式辞書。
    time_grid : TimeGrid
        時間グリッド。

    Returns
    -------
    np.ndarray
        観測終了時刻までの時間発展に対応する Koopman 行列。
    """
    K = np.zeros((len(psi), len(psi)), dtype='float64')
    for i in range(len(psi)):
        result = solve_dual_ode(model=model, psi=psi, time_grid=time_grid, comp_index=psi.i2s(i).tolist())
        K[i, :] = result[-1]

    return K
