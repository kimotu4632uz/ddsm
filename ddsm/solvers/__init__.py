"""Koopman 作用素推定と時間発展 solver を提供するパッケージ。"""
from .dual import (
    StatsTrajectory,
    create_generator,
    create_koopman,
    solve_dual,
    solve_dual_ode,
)
from .ode import (
    Trajectory,
    solve_ode,
)
from .time_grid import TimeGrid
