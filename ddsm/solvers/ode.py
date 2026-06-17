"""ODE の時間発展 solver と軌道データを提供するモジュール。"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.integrate import solve_ivp

from .model import ODEModel
from .time_grid import TimeGrid


@dataclass(slots=True, eq=False)
class Trajectory:
    """時間発展の軌道を表す。

    Attributes
    ----------
    time_grid : TimeGrid
        時間グリッド。
    initial_state : np.ndarray
        初期状態。
    values : np.ndarray
        観測時刻ごとの状態。形状は ``(dim, n_obs)``。

    Raises
    ------
    ValueError
        ``initial_state`` または ``values`` の形状が不正な場合。
    """

    time_grid: TimeGrid
    initial_state: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        """軌道データの形状を検証する。

        Raises
        ------
        ValueError
            初期状態または状態列の形状が不正な場合。
        """
        if self.initial_state.ndim != 1:
            raise ValueError('initial_state must be a 1D array')
        if self.values.ndim != 2:
            raise ValueError('values must be a 2D array')
        if self.values.shape[0] != self.initial_state.size:
            raise ValueError('value dimension must match initial_state')
        if self.values.shape[1] != len(self.time_grid.obs_times):
            raise ValueError('values sample count must match len(time_grid.obs_times)')

    @property
    def obs_times(self) -> np.ndarray:
        """観測時刻列。"""
        return self.time_grid.obs_times

    @property
    def dim(self) -> int:
        """状態空間の次元。"""
        return self.initial_state.size

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            values=self.values,
            initial_state=self.initial_state,
            dt=self.time_grid.dt,
            dt_obs=self.time_grid.dt_obs,
            time_steps=self.time_grid.obs_times.shape[0] - 1,
        )

    @classmethod
    def load(cls, path: str) -> Self:
        data = np.load(path)
        time_grid = TimeGrid(
            dt=float(data['dt']),
            dt_obs=float(data['dt_obs']),
            time_steps=int(data['time_steps']),
        )
        return cls(
            time_grid=time_grid,
            initial_state=data['initial_state'],
            values=data['values'],
        )


def solve_ode(
    *,
    rhs: ODEModel | Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    time_grid: TimeGrid,
    method: str = 'RK45',
    rtol: float = 1e-9,
    atol: float = 1e-11,
    **kwargs: dict
) -> Trajectory:
    """モデルのドリフト項に従って ODE を解く。

    Parameters
    ----------
    rhs : ODEModel or Callable[[float, np.ndarray], np.ndarray]
        ODE の右辺関数、型は ``(t, x) -> dx/dt``。
    x0 : np.ndarray
        初期状態。
    time_grid : TimeGrid
        時間グリッド。
    method : str, optional
        ``scipy.integrate.solve_ivp`` で使用する解法。
    rtol : float, optional
        相対許容誤差。
    atol : float, optional
        絶対許容誤差。

    Returns
    -------
    Trajectory
        観測時刻ごとの状態を ``(dim, n_obs)`` で持つ軌道。

    Raises
    ------
    RuntimeError
        ODE solver が失敗した場合。
    """
    if isinstance(rhs, ODEModel):
        drifts_func = rhs.get_drifts_func()
        ode_func = lambda t, x: drifts_func(x).flatten()
    else:
        ode_func = rhs

    initial_state = np.asarray(x0).reshape(-1)
    obs = time_grid.obs_times
    result = solve_ivp(ode_func, t_span=(obs[0], obs[-1]), y0=initial_state, t_eval=obs, method=method, atol=atol, rtol=rtol, **kwargs)

    if not result.success:
        raise RuntimeError('ODE solver failed')

    return Trajectory(time_grid=time_grid, initial_state=initial_state, values=result.y)
