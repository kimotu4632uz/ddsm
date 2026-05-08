from __future__ import annotations
import numpy as np
import sympy
from ..dicts._dicts import MonomialsDict

class Equation(object):
    """
    A container for the symbolic drift and diffusion terms of a system.

    Attributes
    ----------
    drift : sympy.Matrix
        The symbolic matrix representing the drift terms (deterministic part).
    diff : sympy.Matrix or None
        The symbolic matrix representing the diffusion terms (stochastic part).
        Returns None if the system is purely deterministic.
    """
    drift: sympy.Matrix
    diff: sympy.Matrix | None

    def __init__(self, drift: sympy.Matrix, diff: sympy.Matrix | None) -> None:
        """Initialize the Equation object with symbolic drift and diffusion matrices."""
        self.drift = drift
        self.diff = diff

def generator_to_eq(right_L: np.ndarray, psi: MonomialsDict, threshold_drift: float | None = 1.0e-2, threshold_diff: float | None = 1.0e-2) -> Equation:
    """
    Reconstruct symbolic drift and diffusion terms from a generator matrix.

    This function maps the entries of a generator matrix $L$ back to the
    corresponding coefficients of a Stochastic Differential Equation (SDE)
    or Ordinary Differential Equation (ODE) using a monomial basis.

    Parameters
    ----------
    right_L : np.ndarray
        The right generator matrix. Shape must be (N, N) for stochastic systems
        or (N, D) for deterministic systems, where N is the number of basis
        functions and D is the dimension of the state space.
    psi : MonomialsDict
        Object containing information about the monomial basis functions,
        including their degrees and the dimension of the state space.
    threshold_drift : float, optional
        Minimum absolute value for a drift coefficient to be included.
        Coefficients smaller than this are set to zero. Default is 1.0e-2.
    threshold_diff : float, optional
        Minimum absolute value for a diffusion coefficient to be included.
        Default is 1.0e-2.

    Returns
    -------
    Equation
        An Equation object containing the symbolic drift and diffusion matrices.

    Raises
    ------
    ValueError
        If the shape of `right_L` is inconsistent with `psi.dim_` or the
        number of basis functions.

    Notes
    -----
    The reconstruction relies on the properties of the infinitesimal generator $L$.
    For a state vector $x$, the drift $f$ and diffusion $a$ are identified as:

    * Drift: $f_i(x) = L(x_i)$
    * Diffusion: $a_{ij}(x) = L(x_i x_j) - x_i L(x_j) - x_j L(x_i)$

    The function iterates through the basis functions to identify linear (degree 1)
    and quadratic (degree 2) terms to perform these calculations symbolically.
    """
    row, col = right_L.shape
    if col != row and col != psi.dim_:
        raise ValueError('The number of columns in right_L must be equal to the number of rows, or must match the dimension of psi.')
    if row != len(psi):
        raise ValueError('The number of rows in right_L must match the number of basis functions in psi.')
    xi = sympy.symbols(f'x1:{psi.dim_ + 1}')
    basis_functions = []
    indices_degree1 = np.zeros(psi.dim_, dtype=int)
    indices_degree2 = np.zeros((psi.dim_, psi.dim_), dtype=int)
    for i in range(len(psi)):
        basis_function = 1
        total_degree = 0
        for j in range(psi.dim_):
            basis_function *= xi[j] ** psi.degrees[i, j]
            total_degree += psi.degrees[i, j]
        basis_functions.append(basis_function)
        if total_degree == 1 :
            idx = np.where(psi.degrees[i] == 1)[0][0]
            indices_degree1[idx] = i
        elif total_degree == 2:
            idx = np.where(psi.degrees[i] == 2)[0]
            if len(idx) == 0:
                idx = np.where(psi.degrees[i] == 1)[0]
                indices_degree2[idx[0], idx[1]] = i
                indices_degree2[idx[1], idx[0]] = i
            else:
                idx = idx[0]
                indices_degree2[idx, idx] = i
    if col == row:
        drift_exprs = []
        for i in range(psi.dim_):
            drift_expr = 0
            l = right_L[:, indices_degree1[i]]
            for j in range(len(psi)):
                if threshold_drift is not None and abs(l[j]) < threshold_drift:
                    continue
                drift_expr += l[j] * basis_functions[j]
            drift_expr = drift_expr.expand()
            drift_exprs.append(drift_expr)

        diff_exprs = np.zeros((psi.dim_, psi.dim_), dtype=object)
        for i in range(psi.dim_):
            for j in range(i, psi.dim_):
                diff_expr = 0
                l = right_L[:, indices_degree2[i, j]]
                for k in range(len(psi)):
                    if threshold_diff is not None and abs(l[k]) < threshold_diff:
                        continue
                    diff_expr += l[k] * basis_functions[k]
                if i == j:
                    diff_expr -= (2 * drift_exprs[i] * xi[i]).expand()
                    diff_expr = diff_expr.expand()
                    diff_exprs[i, j] = diff_expr
                else:
                    diff_expr -= (drift_exprs[i] * xi[j] + drift_exprs[j] * xi[i]).expand()
                    diff_expr = diff_expr.expand()
                    diff_exprs[i, j] = diff_expr
                    diff_exprs[j, i] = diff_expr
        return Equation(sympy.Matrix(drift_exprs), sympy.Matrix(diff_exprs))
    else:
        drift_exprs = []
        idx_constant = np.where(np.sum(psi.degrees, axis=1) == 0)[0][0]
        for i in range(psi.dim_):
            drift_expr = 0
            idx = indices_degree1[i] - 1 if idx_constant < indices_degree1[i] else indices_degree1[i]
            l = right_L[:, idx]
            for j in range(len(psi)):
                if threshold_drift is not None and abs(l[j]) < threshold_drift:
                    continue
                drift_expr += l[j] * basis_functions[j]
            drift_expr = drift_expr.expand()
            drift_exprs.append(drift_expr)
        return Equation(sympy.Matrix(drift_exprs), None)