"""Exact, bounded-work recovery for an otherwise unclassified continuous LP.

This helper is called only after the unchanged native retry sequence fails.
Explicit nonnegativity makes supported polyhedra pointed. Every finite input
is retained as its exact binary64 Fraction; no constraint or region is omitted.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math
from numbers import Real
from typing import TYPE_CHECKING, Literal

from .models.linear_optimization_result import LinearOptimizationResult
if TYPE_CHECKING:
    from .linear_constraints import LinearConstraintSystem

MAX_EXACT_LP_VARIABLES = 5
MAX_EXACT_ACTIVE_SET_COMBINATIONS = 20_000


@dataclass(frozen=True)
class ExactLPCertificate:
    status: Literal["optimal", "infeasible", "unbounded"]
    point: tuple[Fraction, ...] | None
    recession_ray: tuple[Fraction, ...] | None
    exact_objective: Fraction | None
    variable_count: int
    affine_dimension: int
    combinations_required: int
    point_sets_tested: int
    recession_sets_tested: int
    point_vertex_count: int
    recession_vertex_count: int
    inconsistent_equalities: bool = False

    def to_linear_result(self) -> LinearOptimizationResult:
        value = float(self.exact_objective) if self.exact_objective is not None else None
        if self.exact_objective is not None and (not math.isfinite(value) or (self.exact_objective != 0 and value == 0.0)):
            raise ValueError("exact optimum is not representable as a finite nonzero binary64 value")
        return LinearOptimizationResult(
            status=self.status, objective_sense="min",
            solver_status_code={"optimal": 0, "infeasible": 2, "unbounded": 3}[self.status],
            solver_message=(
                "exact binary active-set recovery; " + self.status + "; "
                f"checked {self.point_sets_tested} point sets and "
                f"{self.recession_sets_tested} recession sets"
            ),
            optimal_value=value,
        )


@dataclass(frozen=True)
class ExactLPRecoveryAttempt:
    certificate: ExactLPCertificate | None = None
    rejection_reason: str | None = None


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _fraction(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("inputs must be finite real binary64 coefficients")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("inputs must be finite real binary64 coefficients")
    return Fraction.from_float(number)


def _rref(matrix):
    rows = [list(row) for row in matrix]
    pivots = []
    if not rows:
        return rows, pivots
    position = 0
    for column in range(len(rows[0])):
        pivot = next((i for i in range(position, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[position], rows[pivot] = rows[pivot], rows[position]
        factor = rows[position][column]
        rows[position] = [v / factor for v in rows[position]]
        for i in range(len(rows)):
            if i != position:
                factor = rows[i][column]
                if factor:
                    rows[i] = [v - factor * p for v, p in zip(rows[i], rows[position], strict=True)]
        pivots.append(column)
        position += 1
        if position == len(rows):
            break
    return rows, pivots


def _equality_basis(matrix, right, n):
    rows, _ = _rref([list(row) + [rhs] for row, rhs in zip(matrix, right, strict=True)])
    if any(not any(row[:n]) and row[n] for row in rows):
        return (), (), True
    active = [row for row in rows if any(row[:n])]
    return tuple(tuple(row[:n]) for row in active), tuple(row[n] for row in active), False


def _solve_unique(matrix, right, n):
    rows, pivots = _rref([list(row) + [rhs] for row, rhs in zip(matrix, right, strict=True)])
    if any(not any(row[:n]) and row[n] for row in rows):
        return None
    if len([p for p in pivots if p < n]) != n:
        return None
    point = [Fraction(0)] * n
    for row, column in zip(rows, pivots, strict=False):
        if column < n:
            point[column] = row[n]
    return tuple(point)


def _valid_point(point, inequalities, right, equalities, equality_right):
    return (
        all(_dot(row, point) <= rhs for row, rhs in zip(inequalities, right, strict=True))
        and all(_dot(row, point) == rhs for row, rhs in zip(equalities, equality_right, strict=True))
    )


def _vertices(inequalities, right, original_equalities, original_right, basis, basis_right, n):
    points = set()
    tested = 0
    for active in combinations(range(len(inequalities)), n - len(basis)):
        tested += 1
        point = _solve_unique(
            list(basis) + [inequalities[i] for i in active],
            list(basis_right) + [right[i] for i in active], n,
        )
        if point is not None and _valid_point(
            point, inequalities, right, original_equalities, original_right,
        ):
            points.add(point)
    return points, tested


def recover_exact_linear_minimum(
    *, system: LinearConstraintSystem, objective: list[float],
) -> ExactLPRecoveryAttempt:
    """Certify a small pointed LP, or explicitly decline without truncation.

    Complete vertex enumeration certifies finite minima or infeasibility.
    The compact recession slice certifies all improving directions. An
    unbounded result requires both an exact feasible point and improving ray.
    """
    n = len(objective)
    if not 1 <= n <= MAX_EXACT_LP_VARIABLES:
        return ExactLPRecoveryAttempt(rejection_reason="ambient dimension must be between one and five")
    if system.variable_count != n:
        return ExactLPRecoveryAttempt(rejection_reason="objective and constraint dimensions differ")
    try:
        c = tuple(_fraction(v) for v in objective)
        inequalities = tuple(tuple(_fraction(v) for v in row) for row in system.inequalities_left_side)
        right = tuple(_fraction(v) for v in system.inequalities_right_side)
        equalities = tuple(tuple(_fraction(v) for v in row) for row in system.equalities_left_side)
        equality_right = tuple(_fraction(v) for v in system.equalities_right_side)
    except (ValueError, TypeError, OverflowError) as error:
        return ExactLPRecoveryAttempt(rejection_reason=str(error))
    if (
        len(inequalities) != len(right) or len(equalities) != len(equality_right)
        or any(len(row) != n for row in inequalities + equalities)
    ):
        return ExactLPRecoveryAttempt(rejection_reason="invalid constraint dimensions")
    for i in range(n):
        if not any(
            rhs == 0 and row[i] < 0 and all(v == 0 for j, v in enumerate(row) if j != i)
            for row, rhs in zip(inequalities, right, strict=True)
        ):
            return ExactLPRecoveryAttempt(rejection_reason="every variable requires an explicit homogeneous nonnegativity constraint")
    basis, basis_right, inconsistent = _equality_basis(equalities, equality_right, n)
    if inconsistent:
        return ExactLPRecoveryAttempt(certificate=ExactLPCertificate(
            "infeasible", None, None, None, n, -1, 0, 0, 0, 0, 0, True,
        ))
    recession_equalities = equalities + (tuple(Fraction(1) for _ in range(n)),)
    recession_right = tuple(Fraction(0) for _ in equalities) + (Fraction(1),)
    ray_basis, ray_basis_right, ray_inconsistent = _equality_basis(recession_equalities, recession_right, n)
    required_points = math.comb(len(inequalities), n - len(basis))
    required_rays = 0 if ray_inconsistent else math.comb(len(inequalities), n - len(ray_basis))
    required = required_points + required_rays
    if required > MAX_EXACT_ACTIVE_SET_COMBINATIONS:
        return ExactLPRecoveryAttempt(rejection_reason=(
            f"complete enumeration needs {required} active sets; "
            f"limit is {MAX_EXACT_ACTIVE_SET_COMBINATIONS}; none enumerated"
        ))
    points, point_sets = _vertices(
        inequalities, right, equalities, equality_right, basis, basis_right, n,
    )
    if not points:
        # A nonempty pointed polyhedron always has a vertex. Nonnegativity
        # establishes pointedness; the complete active-set scan found none.
        return ExactLPRecoveryAttempt(certificate=ExactLPCertificate(
            "infeasible", None, None, None, n, n - len(basis), required,
            point_sets, 0, 0, 0,
        ))
    rays, ray_sets = (set(), 0) if ray_inconsistent else _vertices(
        inequalities, tuple(Fraction(0) for _ in right),
        recession_equalities, recession_right, ray_basis, ray_basis_right, n,
    )
    point = min(points, key=lambda x: (_dot(c, x), x))
    improving = [r for r in rays if _dot(c, r) < 0]
    ray = min(improving, key=lambda r: (_dot(c, r), r)) if improving else None
    if ray is not None:
        if not (
            _valid_point(point, inequalities, right, equalities, equality_right)
            and _valid_point(ray, inequalities, tuple(Fraction(0) for _ in right),
                             recession_equalities, recession_right)
            and all(v >= 0 for v in ray) and _dot(c, ray) < 0
        ):
            raise RuntimeError("invalid exact LP unboundedness certificate")
        status, exact_value = "unbounded", None
    else:
        status, exact_value = "optimal", _dot(c, point)
        try:
            rounded_value = float(exact_value)
            representable = math.isfinite(rounded_value) and (exact_value == 0 or rounded_value != 0.0)
        except OverflowError:
            representable = False
        if not representable:
            return ExactLPRecoveryAttempt(rejection_reason="exact finite optimum is outside representable finite nonzero binary64 output range")
    return ExactLPRecoveryAttempt(certificate=ExactLPCertificate(
        status, point, ray, exact_value, n, n - len(basis), required,
        point_sets, ray_sets, len(points), len(rays),
    ))
