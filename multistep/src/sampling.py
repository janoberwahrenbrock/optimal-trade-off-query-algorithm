from __future__ import annotations

import math

import numpy as np
from scipy.linalg import null_space

from .linear_constraints import LinearConstraintSystem


MAX_DIRECTION_RETRIES = 100


def sample_points_from_constraint_system(
    system: LinearConstraintSystem,
    num_samples: int,
    burn_in: int = 200,
    thinning: int = 5,
    seed: int | None = None,
    tol: float = 1e-10,
) -> list[list[float]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if burn_in < 0:
        raise ValueError("burn_in must not be negative")

    if thinning <= 0:
        raise ValueError("thinning must be positive")

    if tol <= 0:
        raise ValueError("tol must be positive")

    if system.variable_count <= 0:
        raise ValueError("system has no variables")

    current_point = np.array(system.find_feasible_point(), dtype=float)
    equality_nullspace_basis = _compute_equality_nullspace_basis(system)

    if equality_nullspace_basis.shape[1] == 0:
        return [current_point.tolist() for _ in range(num_samples)]

    rng = np.random.default_rng(seed)
    sampled_points: list[list[float]] = []
    total_steps = burn_in + num_samples * thinning

    for step_index in range(total_steps):
        for attempt_index in range(MAX_DIRECTION_RETRIES):
            direction = _sample_direction_in_nullspace(
                equality_nullspace_basis=equality_nullspace_basis,
                rng=rng,
                tol=tol,
            )
            try:
                lambda_min, lambda_max = _compute_feasible_lambda_interval(
                    system=system,
                    current_point=current_point,
                    direction=direction,
                    tol=tol,
                )
                break
            except RuntimeError as exc:
                if not _is_retryable_lambda_interval_error(exc):
                    raise
                if attempt_index == MAX_DIRECTION_RETRIES - 1:
                    current_point = np.array(system.find_feasible_point(), dtype=float)
                    lambda_min, lambda_max = _compute_feasible_lambda_interval(
                        system=system,
                        current_point=current_point,
                        direction=direction,
                        tol=tol,
                    )
                    break

        sampled_lambda = rng.uniform(lambda_min, lambda_max)
        current_point = current_point + sampled_lambda * direction

        if step_index >= burn_in and (step_index - burn_in) % thinning == 0:
            sampled_points.append(current_point.astype(float).tolist())

    return sampled_points


def _is_retryable_lambda_interval_error(exc: RuntimeError) -> bool:
    return str(exc) in {
        "current_point is numerically outside the feasible region",
        "no feasible lambda interval found for the sampled direction",
    }


def _compute_equality_nullspace_basis(system: LinearConstraintSystem) -> np.ndarray:
    variable_count = system.variable_count

    if not system.equalities_left_side:
        return np.eye(variable_count, dtype=float)

    equality_matrix = np.array(system.equalities_left_side, dtype=float)
    return null_space(equality_matrix)


def _sample_direction_in_nullspace(
    equality_nullspace_basis: np.ndarray,
    rng: np.random.Generator,
    tol: float,
) -> np.ndarray:
    nullspace_dimension = equality_nullspace_basis.shape[1]

    for _ in range(100):
        coefficients = rng.normal(size=nullspace_dimension)
        direction = equality_nullspace_basis @ coefficients
        direction_norm = np.linalg.norm(direction)
        if direction_norm > tol:
            return direction / direction_norm

    raise RuntimeError("failed to sample a non-zero direction in the equality nullspace")


def _compute_feasible_lambda_interval(
    system: LinearConstraintSystem,
    current_point: np.ndarray,
    direction: np.ndarray,
    tol: float,
) -> tuple[float, float]:
    lambda_min = -math.inf
    lambda_max = math.inf

    for left_side, right_side in zip(
        system.inequalities_left_side,
        system.inequalities_right_side,
    ):
        inequality_row = np.array(left_side, dtype=float)
        numerator = float(right_side - inequality_row @ current_point)
        denominator = float(inequality_row @ direction)

        if abs(denominator) <= tol:
            if numerator < -tol:
                raise RuntimeError(
                    "current_point is numerically outside the feasible region"
                )
            continue

        candidate_lambda = numerator / denominator
        if denominator > 0:
            lambda_max = min(lambda_max, candidate_lambda)
        else:
            lambda_min = max(lambda_min, candidate_lambda)

    if not math.isfinite(lambda_min) or not math.isfinite(lambda_max):
        raise ValueError(
            "sampling requires a bounded feasible region along every sampled direction"
        )

    if lambda_min > lambda_max + tol:
        raise RuntimeError("no feasible lambda interval found for the sampled direction")

    if lambda_min > lambda_max:
        midpoint = 0.5 * (lambda_min + lambda_max)
        return midpoint, midpoint

    return lambda_min, lambda_max
