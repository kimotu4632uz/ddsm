from __future__ import annotations

import abc
from enum import Enum
from itertools import product
from typing import Self

import numpy as np
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array

from ..base._dicts import BaseDict


class MonomialBasisKind(str, Enum):
    """Enumeration of the construction methods for monomial bases."""

    TOTAL_DEGREE = "total_degree"
    KRON_LEFT = "kron_left"
    KRON_RIGHT = "kron_right"


class BaseMonomials(BaseDict):
    """
    Polynomial dictionary consisting of monomials up to a specified degree.

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of the polynomial features.
    """
    degree: int
    basis_kind: MonomialBasisKind
    dim_: int
    degree_list_: np.ndarray
    _degree_to_index: dict[tuple[int, ...], int]

    def __init__(self, degree: int = 2) -> None:
        """Initialize the MonomialsDict instance."""
        super().__init__()
        self.degree = degree

    def __len__(self) -> int:
        """
        Number of monomial features.

        Returns
        -------
        int
            Count of monomials.
        """
        return len(self.degree_list_) if hasattr(self, 'degree_list_') else 0

    def lift(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data into the monomial feature space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray
            Evaluated monomials.
        """
        X = check_array(X)
        self.fit(X)
        return np.prod(X[:, np.newaxis, :] ** self.degree_list_[np.newaxis, :, :], axis=2)

    def reconstruct(self, PsiX: np.ndarray) -> np.ndarray:
        """
        Extract first-order terms from the monomial features.

        Parameters
        ----------
        PsiX : ndarray of shape (n_samples, n_monomials)
            Monomial features.

        Returns
        -------
        ndarray
            Reconstructed state features.
        """
        PsiX = check_array(PsiX)
        indices = np.where((self.degree_list_.sum(axis=1) == 1) & (self.degree_list_.max(axis=1) == 1))[0]
        return PsiX[:, indices]

    def diff(self, X: np.ndarray) -> np.ndarray:
        """
        Compute first-order derivatives of monomials.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_monomials, dim)
            Jacobian of the monomials.
        """
        X = check_array(X)
        self.fit(X)
        P_diff = self.degree_list_[:, np.newaxis, :] - np.eye(self.dim_)[np.newaxis, :, :]
        P_diff = np.maximum(P_diff, 0)
        terms = X[:, np.newaxis, np.newaxis, :] ** np.maximum(P_diff[np.newaxis, ...], 0)
        diff_monomials = np.prod(terms, axis=-1)
        coeffs = self.degree_list_.sum(axis=1)
        J = coeffs * diff_monomials * (coeffs > 0)
        return J

    def diff2(self, X: np.ndarray) -> np.ndarray:
        """
        Compute second-order derivatives of monomials.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_monomials, dim, dim)
            Hessian of the monomials.
        """
        X = check_array(X)
        self.fit(X)
        I = np.eye(self.dim_)
        P_diff2 = self.degree_list_[:, np.newaxis, np.newaxis, :] - I[np.newaxis, :, np.newaxis, :] - I[np.newaxis, np.newaxis, :, :]
        P_diff2 = np.maximum(P_diff2, 0)
        terms = X[:, np.newaxis, np.newaxis, np.newaxis, :] ** P_diff2[np.newaxis, ...]
        diff2_monomials = np.prod(terms, axis=-1)
        coeffs = (self.degree_list_[:, :, np.newaxis] * (self.degree_list_[:, np.newaxis, :] - I[np.newaxis, :, :]))[np.newaxis, ...]
        H = coeffs * diff2_monomials * (coeffs > 0)
        return H

    def fit(self, X: np.ndarray, **kwargs) -> Self:
        """
        Generate monomial powers based on input dimension and degree.

        Parameters
        ----------
        X : ndarray
            Training data.
        **kwargs : dict
            Optional 'degree' override.

        Returns
        -------
        Self
            The fitted instance.
        """
        X = check_array(X)
        is_regenerate = False
        if not hasattr(self, 'dim_') or X.shape[1] != self.dim_:
            self.dim_ = X.shape[1]
            is_regenerate = True
        if kwargs.get('degree', self.degree) != self.degree:
            self.degree = kwargs['degree']
            is_regenerate = True
        if is_regenerate:
            self.degree_list_ = self._generate_degrees()
            self._degree_to_index = self._generate_degree_to_index()
        return self

    @abc.abstractmethod
    def _generate_degrees(self) -> np.ndarray:
        """Internal helper to generate polynomial exponent matrix."""
        ...

    @property
    def degrees(self) -> np.ndarray:
        """
        Exponent matrix of the monomials.

        Returns
        -------
        ndarray
            Shape (n_monomials, dim).
        """
        return self.degree_list_ if hasattr(self, 'degree_list_') else np.array([])

    def _generate_degree_to_index(self) -> dict[tuple[int, ...], int]:
        """
        Generate the mapping from degree vectors to dictionary indices.

        Returns
        -------
        dict
            Dictionary whose keys are degree vectors and whose values are
            dictionary indices.
        """
        return {
            tuple(self.degree_list_[i, :].astype(int).tolist()): i
            for i in range(len(self))
        }

    def s2i(self, state: list[int]) -> int:
        """
        Convert a degree vector to its index in the monomial dictionary.

        Parameters
        ----------
        state : list[int]
            List of length ``dim`` holding the degree of each variable.

        Returns
        -------
        int
            Index within the monomial dictionary.

        Raises
        ------
        ValueError
            If the length of ``state`` differs from ``dim``, or if ``state``
            is not contained in this monomial dictionary.
        """
        state_array = np.asarray(state, dtype=int)
        if state_array.shape != (self.dim_,):
            raise ValueError(f"state must have shape ({self.dim_},)")

        key = tuple(state_array.tolist())
        if key not in self._degree_to_index:
            raise ValueError(f"state {state} is not included in this monomial dictionary")

        return self._degree_to_index[key]

    def i2s(self, idx: int) -> np.ndarray:
        """
        Convert an index in the monomial dictionary to a degree vector.

        Parameters
        ----------
        idx : int
            Index within the monomial dictionary.

        Returns
        -------
        ndarray
            Vector holding the degree of each variable.
        """
        return self.degree_list_[idx, :] if hasattr(self, 'degree_list_') else np.array([])

    def to_sym(self) -> list[sp.Expr]:
        """
        Convert the monomial dictionary into a list of SymPy expressions.

        Returns
        -------
        list[sympy.Expr]
            List of SymPy expressions following the order of the monomial
            dictionary.
        """
        if not hasattr(self, 'dim_'):
            return []

        exprs: list[sp.Expr] = []
        sym_xs = sp.symbols(f'x_1:{self.dim_+1}')

        for i in range(len(self)):
            e = self.i2s(i)
            sym_expr = sp.Integer(1)
            for j, x in enumerate(sym_xs):
                sym_expr *= x**e[j]
            exprs.append(sym_expr)
        return exprs


class MonomialsDict(BaseMonomials):
    """
    Parameters
    ----------
    degree : int, default=2
        Maximum degree of the polynomial features.
    """
    basis_kind = MonomialBasisKind.TOTAL_DEGREE

    def _generate_degrees(self) -> np.ndarray:
        """Internal helper to generate polynomial exponent matrix."""
        poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        poly.fit(np.zeros((1, self.dim_)))
        return poly.powers_


class KronMonomialsDict(BaseMonomials):
    """
    Monomial dictionary constructed from a tensor product.

    Parameters
    ----------
    degree : int
        Maximum degree along each dimension.
    basis_kind : MonomialBasisKind, default=MonomialBasisKind.KRON_RIGHT
        Ordering of the tensor product basis. The default matches the
        current implementation, ``MonomialBasisKind.KRON_RIGHT``.
    """
    basis_kind: MonomialBasisKind

    def __init__(
        self,
        degree: int,
        basis_kind: MonomialBasisKind | str = MonomialBasisKind.KRON_RIGHT,
    ) -> None:
        """
        Initialize the tensor product monomial dictionary.

        Parameters
        ----------
        degree : int
            Maximum degree along each dimension.
        basis_kind : MonomialBasisKind or str, \
default=MonomialBasisKind.KRON_RIGHT
            Ordering of the tensor product basis.

        Raises
        ------
        ValueError
            If ``basis_kind`` is not a tensor product basis kind.
        """
        basis_kind = MonomialBasisKind(basis_kind)
        if basis_kind not in (
            MonomialBasisKind.KRON_LEFT,
            MonomialBasisKind.KRON_RIGHT,
        ):
            raise ValueError("KronMonomials requires KRON_LEFT or KRON_RIGHT")
        self.basis_kind = basis_kind
        super().__init__(degree=degree)

    def _generate_degrees(self) -> np.ndarray:
        """Internal helper to generate polynomial exponent matrix."""
        values = range(self.degree + 1)
        states = np.array(list(product(values, repeat=self.dim_)), dtype=int)
        if self.basis_kind is MonomialBasisKind.KRON_RIGHT:
            states = states[:, ::-1]
        return states
