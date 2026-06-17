"""時間刻みと観測時刻を管理するクラスを提供するモジュール。"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..utils import is_close_float


def _count_steps(span: float, step: float, span_name: str, step_name: str) -> int:
    """区間長を刻み幅で割ったステップ数を返す。

    Parameters
    ----------
    span : float
        区間長。
    step : float
        刻み幅。
    span_name : str
        エラーメッセージに使う区間長の名前。
    step_name : str
        エラーメッセージに使う刻み幅の名前。

    Returns
    -------
    int
        整数化したステップ数。

    Raises
    ------
    ValueError
        ``span`` が ``step`` の整数倍でない場合。
    """
    count_float = span / step
    count = int(round(count_float))
    if not is_close_float(count_float, float(count)):
        raise ValueError(f'{span_name} is not multiple of {step_name}')
    return count


@dataclass(frozen=True, slots=True)
class TimeStep:
    """時間発展の 1 ステップを表す。

    Attributes
    ----------
    index : int
        1 始まりのステップ番号。
    time : float
        ステップに対応する時刻。
    should_save : bool
        このステップを観測値として保存するかどうか。
    """

    index: int
    time: float
    should_save: bool


class TimeGrid:
    """時間発展と観測時刻のグリッドを管理する。

    Parameters
    ----------
    dt_obs : float
        観測時刻の間隔。
    dt : float or None, optional
        数値積分の時間刻み。省略時は ``dt_obs`` の 1/10 を用いる。
    t_f : float or None, optional
        最終観測時刻。
    time_steps : int or None, optional
        観測ステップ数。``t_f`` とどちらか一方のみを指定する。

    Raises
    ------
    ValueError
        時間刻みや時間長が不正、または割り切れない場合。
    """

    _dt: float
    _dt_obs: float
    _total_steps: int
    _save_step_indices: set[int]
    _obs_times: np.ndarray

    def __init__(
        self,
        *,
        dt_obs: float,
        dt: float | None = None,
        t_f: float | None = None,
        time_steps: int | None = None
    ) -> None:
        """時間グリッドを初期化する。

        Parameters
        ----------
        dt_obs : float
            観測時刻の間隔。
        dt : float or None, optional
            数値積分の時間刻み。省略時は ``dt_obs`` の 1/10 を用いる。
        t_f : float or None, optional
            最終観測時刻。
        time_steps : int or None, optional
            観測ステップ数。``t_f`` とどちらか一方のみを指定する。

        Raises
        ------
        ValueError
            時間刻みや時間長が不正、または割り切れない場合。
        """
        if dt is not None and dt < 0:
            raise ValueError('dt must be non-negative')
        if dt_obs < 0:
            raise ValueError('dt_obs must be non-negative')

        if (t_f is None) == (time_steps is None):
            raise ValueError('Exactly one of t_f or time_steps must be specified')

        if t_f is not None and t_f < 0:
            raise ValueError('t_f must be non-negative')
        if time_steps is not None and time_steps <= 0:
            raise ValueError('time_steps must be positive')

        if dt is None:
            dt = dt_obs * 0.1
            dt_steps = 10
        else:
            dt_steps = _count_steps(dt_obs, dt, 'dt_obs', 'dt')

        if t_f is not None:
            obs_steps = _count_steps(t_f, dt_obs, 't_f', 'dt_obs')
        else:
            assert time_steps is not None
            obs_steps = time_steps

        self._dt = dt
        self._dt_obs = dt_obs
        self._total_steps = obs_steps * dt_steps

        self._save_step_indices = {
            obs_index * dt_steps
            for obs_index in range(obs_steps + 1)
        }
        self._obs_times = np.arange(obs_steps + 1, dtype=float) * dt_obs

    @property
    def dt(self) -> float:
        """数値積分の時間刻み。"""
        return self._dt

    @property
    def dt_obs(self) -> float:
        """観測時刻の間隔。"""
        return self._dt_obs

    @property
    def obs_times(self) -> np.ndarray:
        """観測時刻列。

        Returns
        -------
        np.ndarray
            観測時刻列のコピー。
            ndarrayの中身は``[0, dt_obs, 2*dt_obs, ..., t_f]``であり、len(obs_times) = time_steps + 1 である。
        """
        return self._obs_times.copy()

    def iter_steps(self) -> Iterator[TimeStep]:
        """時間発展の各ステップを順に返す。

        Yields
        ------
        TimeStep
            ステップ番号、時刻、保存対象かどうかを持つ ``TimeStep``。
        """
        for step_index in range(1, self._total_steps + 1):
            yield TimeStep(
                index=step_index,
                time=float(step_index * self._dt),
                should_save=step_index in self._save_step_indices,
            )
