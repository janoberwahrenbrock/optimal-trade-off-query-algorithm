from __future__ import annotations

import math

import numpy as np
from scipy.linalg import null_space

from .io_models import AlternativenMatrix
from .ungleichungssysteme import Ungleichungssystem


MAX_DIRECTION_RETRIES = 100


def sample_points_from_ungleichungssystem(
    system: Ungleichungssystem,
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

    if system.anzahl_variablen <= 0:
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


def estimate_optimality_shares(
    alternativen_matrix: AlternativenMatrix,
    samples: list[list[float]],
    remaining_candidates: list[int],
    utility_tol: float = 1e-9,
) -> dict[int, float]:
    if not remaining_candidates:
        raise ValueError("remaining_candidates must not be empty")

    if not samples:
        raise ValueError("samples must not be empty")

    if utility_tol < 0:
        raise ValueError("utility_tol must not be negative")

    utilities = alternativen_matrix.entries
    optimality_counts = {candidate_index: 0 for candidate_index in remaining_candidates}

    for sample in samples:
        utility_values = {
            candidate_index: sum(
                utility_value * weight
                for utility_value, weight in zip(utilities[candidate_index], sample)
            )
            for candidate_index in remaining_candidates
        }
        max_utility_value = max(utility_values.values())

        for candidate_index, utility_value in utility_values.items():
            if abs(utility_value - max_utility_value) <= utility_tol:
                optimality_counts[candidate_index] += 1

    anzahl_samples = len(samples)
    return {
        candidate_index: optimality_counts[candidate_index] / anzahl_samples
        for candidate_index in remaining_candidates
    }


def _compute_equality_nullspace_basis(system: Ungleichungssystem) -> np.ndarray:
    n_variablen = system.anzahl_variablen

    if not system.gleichungen_linke_seite:
        return np.eye(n_variablen, dtype=float)

    equality_matrix = np.array(system.gleichungen_linke_seite, dtype=float)
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
    system: Ungleichungssystem,
    current_point: np.ndarray,
    direction: np.ndarray,
    tol: float,
) -> tuple[float, float]:
    lambda_min = -math.inf
    lambda_max = math.inf

    for linke_seite, rechte_seite in zip(
        system.ungleichungen_linke_seite,
        system.ungleichungen_rechte_seite,
    ):
        inequality_row = np.array(linke_seite, dtype=float)
        numerator = float(rechte_seite - inequality_row @ current_point)
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
