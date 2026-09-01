from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

from .linear_constraints import LinearConstraintSystem
from .linear_programming import LINPROG_OPTIONS, run_linprog_with_retries


MAX_DIRECTION_RETRIES = 100


@dataclass(frozen=True)
class SamplingDiagnostics:
    """Basic diagnostics for a set of Hit-and-Run chains."""

    chain_count: int
    sample_count: int
    unique_sample_count: int
    relative_interior_radius: float
    minimum_effective_sample_size: float
    median_effective_sample_size: float
    maximum_split_r_hat: float | None


def sample_points_from_constraint_system(
    system: LinearConstraintSystem,
    num_samples: int,
    burn_in: int = 200,
    thinning: int = 5,
    seed: int | None = None,
    tol: float = 1e-10,
    chain_count: int = 1,
) -> list[list[float]]:
    samples, _ = sample_points_with_diagnostics(
        system=system,
        num_samples=num_samples,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed,
        tol=tol,
        chain_count=chain_count,
    )
    return samples


def sample_points_with_diagnostics(
    system: LinearConstraintSystem,
    num_samples: int,
    burn_in: int = 200,
    thinning: int = 5,
    seed: int | None = None,
    tol: float = 1e-10,
    chain_count: int = 1,
) -> tuple[list[list[float]], SamplingDiagnostics]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if burn_in < 0:
        raise ValueError("burn_in must not be negative")

    if thinning <= 0:
        raise ValueError("thinning must be positive")

    if tol <= 0:
        raise ValueError("tol must be positive")

    if chain_count <= 0:
        raise ValueError("chain_count must be positive")

    if chain_count > num_samples:
        raise ValueError("chain_count must not exceed num_samples")

    if system.variable_count <= 0:
        raise ValueError("system has no variables")

    equality_nullspace_basis = _compute_equality_nullspace_basis(system)
    current_point, relative_interior_radius = find_relative_interior_point(
        system=system,
        equality_nullspace_basis=equality_nullspace_basis,
        tol=tol,
    )

    if equality_nullspace_basis.shape[1] == 0:
        samples = [current_point.tolist() for _ in range(num_samples)]
        return samples, SamplingDiagnostics(
            chain_count=1,
            sample_count=num_samples,
            unique_sample_count=1,
            relative_interior_radius=relative_interior_radius,
            minimum_effective_sample_size=float(num_samples),
            median_effective_sample_size=float(num_samples),
            maximum_split_r_hat=None,
        )

    inequality_matrix, inequality_right_side, _, _ = system.get_solver_matrices()
    if inequality_matrix is None or inequality_right_side is None:
        inequality_matrix = np.empty((0, system.variable_count), dtype=float)
        inequality_right_side = np.empty((0,), dtype=float)

    samples_per_chain = [num_samples // chain_count] * chain_count
    for chain_index in range(num_samples % chain_count):
        samples_per_chain[chain_index] += 1
    seed_sequences = np.random.SeedSequence(seed).spawn(chain_count)
    chains = [
        _sample_hit_and_run_chain(
            system=system,
            initial_point=current_point,
            equality_nullspace_basis=equality_nullspace_basis,
            inequality_matrix=inequality_matrix,
            inequality_right_side=inequality_right_side,
            num_samples=chain_sample_count,
            burn_in=burn_in,
            thinning=thinning,
            rng=np.random.default_rng(chain_seed),
            tol=tol,
        )
        for chain_sample_count, chain_seed in zip(samples_per_chain, seed_sequences)
    ]
    sampled_points = [point.tolist() for chain in chains for point in chain]
    diagnostics = _compute_sampling_diagnostics(
        chains=chains,
        relative_interior_radius=relative_interior_radius,
    )
    return sampled_points, diagnostics


def find_relative_interior_point(
    system: LinearConstraintSystem,
    equality_nullspace_basis: np.ndarray | None = None,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Find a point maximally separated from inequalities in the affine hull.

    The ordinary feasibility LP commonly returns a vertex of the simplex.  A
    Hit-and-Run chain started there can remain stuck because almost every
    sampled direction initially points outside.  This relative Chebyshev
    center maximizes slack only in directions permitted by the equalities.
    """

    if system.variable_count <= 0:
        raise ValueError("system has no variables")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    basis = (
        _compute_equality_nullspace_basis(system)
        if equality_nullspace_basis is None
        else equality_nullspace_basis
    )
    fallback = np.asarray(system.find_feasible_point(), dtype=float)
    if basis.shape[1] == 0:
        return fallback, 0.0

    inequality_matrix, inequality_right_side, equality_matrix, equality_right_side = (
        system.get_solver_matrices()
    )
    if inequality_matrix is None or inequality_right_side is None:
        return fallback, math.inf

    projected_norms = np.linalg.norm(inequality_matrix @ basis, axis=1)
    augmented_inequalities = np.column_stack(
        (inequality_matrix, projected_norms)
    )
    augmented_equalities = (
        np.column_stack((equality_matrix, np.zeros(equality_matrix.shape[0])))
        if equality_matrix is not None
        else None
    )
    objective = np.zeros(system.variable_count + 1, dtype=float)
    objective[-1] = -1.0
    result = run_linprog_with_retries(
        c=objective,
        A_ub=augmented_inequalities,
        b_ub=inequality_right_side,
        A_eq=augmented_equalities,
        b_eq=equality_right_side,
        bounds=[(None, None)] * system.variable_count + [(0.0, None)],
        method="highs",
        options=LINPROG_OPTIONS,
    )
    if not result.success or result.x is None:
        return fallback, 0.0

    radius = max(0.0, float(result.x[-1]))
    point = np.asarray(result.x[:-1], dtype=float)
    if radius <= tol:
        return fallback, radius
    return point, radius


def _sample_hit_and_run_chain(
    system: LinearConstraintSystem,
    initial_point: np.ndarray,
    equality_nullspace_basis: np.ndarray,
    inequality_matrix: np.ndarray,
    inequality_right_side: np.ndarray,
    num_samples: int,
    burn_in: int,
    thinning: int,
    rng: np.random.Generator,
    tol: float,
) -> np.ndarray:
    current_point = initial_point.copy()
    sampled_points: list[np.ndarray] = []
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
                    inequality_matrix=inequality_matrix,
                    inequality_right_side=inequality_right_side,
                )
                break
            except RuntimeError as exc:
                if not _is_retryable_lambda_interval_error(exc):
                    raise
                if attempt_index == MAX_DIRECTION_RETRIES - 1:
                    current_point = initial_point.copy()
                    lambda_min, lambda_max = _compute_feasible_lambda_interval(
                        system=system,
                        current_point=current_point,
                        direction=direction,
                        tol=tol,
                        inequality_matrix=inequality_matrix,
                        inequality_right_side=inequality_right_side,
                    )
                    break

        sampled_lambda = rng.uniform(lambda_min, lambda_max)
        current_point = current_point + sampled_lambda * direction

        if step_index >= burn_in and (step_index - burn_in) % thinning == 0:
            sampled_points.append(current_point.astype(float).copy())

    return np.asarray(sampled_points, dtype=float)


def _compute_sampling_diagnostics(
    chains: list[np.ndarray],
    relative_interior_radius: float,
) -> SamplingDiagnostics:
    non_empty_chains = [chain for chain in chains if len(chain) > 0]
    combined = np.vstack(non_empty_chains)
    effective_sample_sizes = [
        sum(_estimate_effective_sample_size(chain[:, coordinate]) for chain in non_empty_chains)
        for coordinate in range(combined.shape[1])
    ]
    split_r_hat_values = _compute_split_r_hat_values(non_empty_chains)
    rounded_samples = np.round(combined, decimals=12)
    unique_sample_count = int(np.unique(rounded_samples, axis=0).shape[0])
    return SamplingDiagnostics(
        chain_count=len(non_empty_chains),
        sample_count=len(combined),
        unique_sample_count=unique_sample_count,
        relative_interior_radius=float(relative_interior_radius),
        minimum_effective_sample_size=float(min(effective_sample_sizes)),
        median_effective_sample_size=float(np.median(effective_sample_sizes)),
        maximum_split_r_hat=(
            float(max(split_r_hat_values)) if split_r_hat_values else None
        ),
    )


def _estimate_effective_sample_size(values: np.ndarray) -> float:
    sample_count = len(values)
    if sample_count < 3 or float(np.var(values)) <= 1e-30:
        return float(sample_count)

    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered) / sample_count)
    autocorrelation_sum = 0.0
    for lag in range(1, sample_count):
        covariance = float(np.dot(centered[:-lag], centered[lag:]) / (sample_count - lag))
        autocorrelation = covariance / variance
        if autocorrelation <= 0.0:
            break
        autocorrelation_sum += autocorrelation
    return max(1.0, min(float(sample_count), sample_count / (1.0 + 2.0 * autocorrelation_sum)))


def _compute_split_r_hat_values(chains: list[np.ndarray]) -> list[float]:
    if len(chains) < 2:
        return []

    half_length = min(len(chain) for chain in chains) // 2
    if half_length < 2:
        return []
    split_chains = np.asarray(
        [
            segment
            for chain in chains
            for segment in (chain[:half_length], chain[-half_length:])
        ],
        dtype=float,
    )
    within_variances = np.var(split_chains, axis=1, ddof=1)
    within = np.mean(within_variances, axis=0)
    between = half_length * np.var(np.mean(split_chains, axis=1), axis=0, ddof=1)
    estimated_variance = ((half_length - 1) * within + between) / half_length
    return [
        math.sqrt(float(estimated_variance[index] / within[index]))
        for index in range(split_chains.shape[2])
        if within[index] > 1e-30
    ]


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
    inequality_matrix: np.ndarray | None = None,
    inequality_right_side: np.ndarray | None = None,
) -> tuple[float, float]:
    if inequality_matrix is None:
        inequality_matrix = np.asarray(
            system.inequalities_left_side,
            dtype=float,
        )
    if inequality_right_side is None:
        inequality_right_side = np.asarray(
            system.inequalities_right_side,
            dtype=float,
        )

    numerators = inequality_right_side - inequality_matrix @ current_point
    denominators = inequality_matrix @ direction
    near_zero_mask = np.abs(denominators) <= tol
    if np.any(numerators[near_zero_mask] < -tol):
        raise RuntimeError("current_point is numerically outside the feasible region")

    positive_mask = denominators > tol
    negative_mask = denominators < -tol
    lambda_max = (
        float(np.min(numerators[positive_mask] / denominators[positive_mask]))
        if np.any(positive_mask)
        else math.inf
    )
    lambda_min = (
        float(np.max(numerators[negative_mask] / denominators[negative_mask]))
        if np.any(negative_mask)
        else -math.inf
    )

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
