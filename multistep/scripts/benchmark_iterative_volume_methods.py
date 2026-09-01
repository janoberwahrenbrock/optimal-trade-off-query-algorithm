from __future__ import annotations

"""Benchmark volume and sampling methods along complete query trajectories.

The benchmark generates deterministic random decision problems, simulates a
decision maker with fixed hidden weights, and measures the same weight-space
states after 0, 5, 10, and 20 answered queries.  Solver queries use the current
ratio+quantile entropy policy.  Once a problem is solved (or the policy cannot
produce another query), central geometric cuts continue the trajectory so the
late, more complex polytopes can still be compared fairly.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import null_space
from scipy.spatial import ConvexHull, Delaunay, QhullError


MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.optimized import (  # noqa: E402
    OptimizedMultistepConfig,
    OptimizedValueFunctionSession,
)
from multistep.src.candidates import compute_candidate_set  # noqa: E402
from multistep.src.linear_constraints import LinearConstraintSystem  # noqa: E402
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query  # noqa: E402
from multistep.src.optimality_region import build_optimality_region  # noqa: E402
from multistep.src.polytope_geometry import (  # noqa: E402
    PolytopeVertices,
    enumerate_polytope_vertices,
)
from multistep.src.query_probability import classify_query_answer  # noqa: E402
from multistep.src.sampling import (  # noqa: E402
    _compute_sampling_diagnostics,
    _estimate_effective_sample_size,
    sample_points_with_diagnostics,
)
from multistep.src.weight_space import build_weight_space  # noqa: E402


DEFAULT_MILESTONES = (0, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--alternatives", type=int, default=10)
    parser.add_argument("--milestones", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
    parser.add_argument("--max-solve-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--policy-samples", type=int, default=800)
    parser.add_argument("--iid-samples", type=int, default=20_000)
    parser.add_argument("--hr-step", type=int, default=400)
    parser.add_argument("--hr-max-samples", type=int, default=8_000)
    parser.add_argument("--hr-burn-in", type=int, default=200)
    parser.add_argument("--hr-thinning", type=int, default=5)
    parser.add_argument("--hr-chains", type=int, default=4)
    parser.add_argument("--hr-min-ess", type=float, default=200.0)
    parser.add_argument("--hr-max-rhat", type=float, default=1.05)
    parser.add_argument("--hr-max-ci-halfwidth", type=float, default=0.025)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.goals or any(goal_count < 3 for goal_count in args.goals):
        raise ValueError("all goal counts must be at least three")
    if args.problems <= 0 or args.alternatives <= 1:
        raise ValueError("problems must be positive and alternatives greater than one")
    if sorted(set(args.milestones)) != args.milestones or args.milestones[0] != 0:
        raise ValueError("milestones must be unique, sorted, and start at zero")
    if args.max_solve_queries < max(args.milestones):
        raise ValueError("max-solve-queries must include the final milestone")
    if args.policy_samples <= 0 or args.iid_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.hr_step <= 0 or args.hr_max_samples < args.hr_step:
        raise ValueError("invalid Hit-and-Run sample range")
    if args.hr_step % args.hr_chains or args.hr_max_samples % args.hr_chains:
        raise ValueError("Hit-and-Run counts must be divisible by the chain count")


def build_policy_config(args: argparse.Namespace, seed: int) -> OptimizedMultistepConfig:
    return OptimizedMultistepConfig(
        sample_count=int(args.policy_samples),
        burn_in=300,
        thinning=5,
        random_seed=int(seed),
        sampling_chain_count=4,
        skip_zero_probability_branches=True,
        pass_candidate_subset=True,
        use_ratio_terminal_counts=True,
        candidate_count_mode="ratio_relevant",
        depth_one_query_source_mode="ratio",
        ratio_interval_engine="geometry",
        posterior_quantile_levels=(0.25, 0.5, 0.75),
        posterior_query_objective="entropy",
        posterior_query_shortlist_size=21,
        parallelize_root=False,
    )


def generate_problem(
    rng: np.random.Generator,
    goal_count: int,
    alternative_count: int,
) -> tuple[AlternativenMatrix, list[float]]:
    entries = rng.uniform(0.0, 1.0, size=(alternative_count, goal_count))
    weights = rng.dirichlet(np.ones(goal_count, dtype=float))
    return AlternativenMatrix(entries=entries.tolist()), weights.tolist()


def _affine_projection(
    system: LinearConstraintSystem,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, equality_matrix, equality_right_side = system.get_solver_matrices()
    if equality_matrix is None:
        particular_point = np.zeros(system.variable_count, dtype=float)
        basis = np.eye(system.variable_count, dtype=float)
    else:
        assert equality_right_side is not None
        particular_point, *_ = np.linalg.lstsq(
            equality_matrix,
            equality_right_side,
            rcond=None,
        )
        basis = null_space(equality_matrix)
    coordinates = (vertices - particular_point) @ basis
    return coordinates, basis


def _intrinsic_volume(
    system: LinearConstraintSystem,
    polytope: PolytopeVertices,
) -> tuple[float, bool]:
    if polytope.status in {"infeasible", "lower_dimensional"}:
        return 0.0, False
    if polytope.status == "point":
        return 0.0, False
    if polytope.status != "full_dimensional":
        raise RuntimeError(polytope.message or f"polytope status {polytope.status}")
    coordinates, _ = _affine_projection(system, polytope.vertices)
    dimension = coordinates.shape[1]
    if dimension == 1:
        return float(np.ptp(coordinates[:, 0])), False
    try:
        return float(ConvexHull(coordinates).volume), False
    except QhullError:
        return float(ConvexHull(coordinates, qhull_options="QJ").volume), True


def exact_candidate_volumes(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
) -> dict[str, Any]:
    started = time.perf_counter()
    total_polytope = enumerate_polytope_vertices(weight_space)
    total_volume, total_joggled = _intrinsic_volume(weight_space, total_polytope)
    candidate_volumes: list[float] = []
    candidate_vertices: list[int] = []
    joggled_count = int(total_joggled)
    statuses: list[str] = []
    for alternative_index in range(alternatives.get_anzahl_zeilen()):
        region = build_optimality_region(
            alternatives=alternatives,
            weight_space=weight_space,
            alternative_index=alternative_index,
        )
        polytope = enumerate_polytope_vertices(region)
        volume, joggled = _intrinsic_volume(region, polytope)
        candidate_volumes.append(volume)
        candidate_vertices.append(int(len(polytope.vertices)))
        joggled_count += int(joggled)
        statuses.append(polytope.status)
    seconds = time.perf_counter() - started
    if total_volume <= 0.0:
        raise RuntimeError("weight space has no positive intrinsic volume")
    probabilities = np.asarray(candidate_volumes, dtype=float) / total_volume
    return {
        "success": True,
        "seconds": seconds,
        "total_volume": total_volume,
        "weight_space_vertices": int(len(total_polytope.vertices)),
        "weight_space_status": total_polytope.status,
        "candidate_volumes": candidate_volumes,
        "candidate_probabilities": probabilities.tolist(),
        "candidate_vertices": candidate_vertices,
        "candidate_statuses": statuses,
        "positive_volume_candidate_count": int(np.count_nonzero(probabilities > 1e-10)),
        "probability_sum": float(np.sum(probabilities)),
        "probability_sum_absolute_error": abs(float(np.sum(probabilities)) - 1.0),
        "joggled_hulls": joggled_count,
    }


def _build_triangulation(
    system: LinearConstraintSystem,
    polytope: PolytopeVertices,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    if polytope.status != "full_dimensional":
        raise RuntimeError(polytope.message or f"polytope status {polytope.status}")
    coordinates, _ = _affine_projection(system, polytope.vertices)
    dimension = coordinates.shape[1]
    centered_coordinates = coordinates - np.mean(coordinates, axis=0)
    _, singular_values, rotation = np.linalg.svd(
        centered_coordinates,
        full_matrices=False,
    )
    singular_tolerance = (
        np.finfo(float).eps
        * max(centered_coordinates.shape)
        * float(np.max(singular_values, initial=0.0))
    )
    if len(singular_values) != dimension or np.any(
        singular_values <= singular_tolerance
    ):
        raise RuntimeError("polytope vertices are numerically lower-dimensional")
    # Delaunay is particularly fragile for the thin polytopes produced after
    # many query cuts.  Any nonsingular affine transform preserves a valid
    # triangulation of the convex hull.  SVD whitening makes Qhull operate on
    # an isotropic point cloud; the determinant below maps simplex volumes
    # back to the original intrinsic coordinates.
    normalized_coordinates = (
        centered_coordinates @ rotation.T
    ) / singular_values
    inverse_volume_scale = float(np.prod(singular_values))
    joggled = False
    if len(normalized_coordinates) == dimension + 1:
        simplices = np.arange(dimension + 1, dtype=int).reshape(1, -1)
    else:
        try:
            simplices = np.asarray(
                Delaunay(normalized_coordinates).simplices,
                dtype=int,
            )
        except QhullError:
            simplices = np.asarray(
                Delaunay(
                    normalized_coordinates,
                    qhull_options="QJ Qbb Qc",
                ).simplices,
                dtype=int,
            )
            joggled = True
    edges = (
        normalized_coordinates[simplices[:, 1:]]
        - normalized_coordinates[simplices[:, :1]]
    )
    normalized_volumes = np.abs(np.linalg.det(edges)) / math.factorial(dimension)
    scale = max(1.0, float(np.max(normalized_volumes, initial=0.0)))
    keep = normalized_volumes > 1e-14 * scale
    simplices = simplices[keep]
    volumes = normalized_volumes[keep] * inverse_volume_scale
    if len(simplices) == 0 or float(np.sum(volumes)) <= 0.0:
        raise RuntimeError("triangulation has no positive-volume simplex")
    return polytope.vertices, simplices, volumes, joggled


def _sample_from_triangulation(
    vertices: np.ndarray,
    simplices: np.ndarray,
    volumes: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    simplex_indices = rng.choice(
        len(simplices),
        size=sample_count,
        p=volumes / np.sum(volumes),
    )
    barycentric = rng.dirichlet(np.ones(simplices.shape[1]), size=sample_count)
    selected_vertices = vertices[simplices[simplex_indices]]
    return np.einsum("ni,nij->nj", barycentric, selected_vertices)


def _winner_probabilities(
    alternatives: AlternativenMatrix,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    entries = np.asarray(alternatives.entries, dtype=float)
    winner_indices = np.argmax(samples @ entries.T, axis=1)
    probabilities = np.bincount(
        winner_indices,
        minlength=alternatives.get_anzahl_zeilen(),
    ).astype(float) / len(samples)
    return probabilities, winner_indices


def triangulation_iid_measurement(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    polytope = enumerate_polytope_vertices(weight_space)
    vertices, simplices, volumes, joggled = _build_triangulation(
        weight_space,
        polytope,
    )
    triangulation_seconds = time.perf_counter() - started
    rng = np.random.default_rng(seed)
    sampling_started = time.perf_counter()
    samples = _sample_from_triangulation(
        vertices=vertices,
        simplices=simplices,
        volumes=volumes,
        sample_count=sample_count,
        rng=rng,
    )
    probabilities, _ = _winner_probabilities(alternatives, samples)
    sampling_seconds = time.perf_counter() - sampling_started
    inequality_matrix, inequality_right_side, _, _ = weight_space.get_solver_matrices()
    max_violation = 0.0
    if inequality_matrix is not None and inequality_right_side is not None:
        max_violation = max(
            0.0,
            float(np.max(inequality_matrix @ samples.T - inequality_right_side[:, None])),
        )
    return {
        "success": True,
        "seconds": time.perf_counter() - started,
        "triangulation_seconds": triangulation_seconds,
        "sampling_and_scoring_seconds": sampling_seconds,
        "sample_count": sample_count,
        "vertex_count": int(len(vertices)),
        "simplex_count": int(len(simplices)),
        "triangulated_volume": float(np.sum(volumes)),
        "candidate_probabilities": probabilities.tolist(),
        "max_constraint_violation": max_violation,
        "joggled": joggled,
    }


def _indicator_quality(
    alternatives: AlternativenMatrix,
    chains: list[np.ndarray],
) -> tuple[float, float]:
    entries = np.asarray(alternatives.entries, dtype=float)
    winner_chains = [np.argmax(chain @ entries.T, axis=1) for chain in chains]
    combined = np.concatenate(winner_chains)
    maximum_halfwidth = 0.0
    minimum_ess = float(len(combined))
    z = 1.959963984540054
    for candidate_index in range(alternatives.get_anzahl_zeilen()):
        indicator_chains = [
            (winners == candidate_index).astype(float)
            for winners in winner_chains
        ]
        effective_count = sum(
            _estimate_effective_sample_size(indicators)
            for indicators in indicator_chains
        )
        minimum_ess = min(minimum_ess, effective_count)
        probability = float(np.mean(combined == candidate_index))
        denominator = 1.0 + z * z / effective_count
        halfwidth = (
            z
            * math.sqrt(
                probability * (1.0 - probability) / effective_count
                + z * z / (4.0 * effective_count * effective_count)
            )
            / denominator
        )
        maximum_halfwidth = max(maximum_halfwidth, halfwidth)
    return minimum_ess, maximum_halfwidth


def adaptive_hit_and_run_measurement(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    scan_started = time.perf_counter()
    maximum_samples, maximum_diagnostics = sample_points_with_diagnostics(
        system=weight_space,
        num_samples=int(args.hr_max_samples),
        burn_in=int(args.hr_burn_in),
        thinning=int(args.hr_thinning),
        seed=seed,
        chain_count=int(args.hr_chains),
    )
    scan_seconds = time.perf_counter() - scan_started
    maximum_matrix = np.asarray(maximum_samples, dtype=float)
    maximum_per_chain = int(args.hr_max_samples) // int(args.hr_chains)
    maximum_chains = [
        maximum_matrix[
            chain_index * maximum_per_chain : (chain_index + 1) * maximum_per_chain
        ]
        for chain_index in range(int(args.hr_chains))
    ]
    selected_count = int(args.hr_max_samples)
    selected_diagnostics = maximum_diagnostics
    selected_indicator_ess = 0.0
    selected_halfwidth = math.inf
    quality_achieved = False
    consecutive_passes = 0
    checks: list[dict[str, Any]] = []
    for sample_count in range(
        int(args.hr_step),
        int(args.hr_max_samples) + 1,
        int(args.hr_step),
    ):
        per_chain = sample_count // int(args.hr_chains)
        chains = [chain[:per_chain] for chain in maximum_chains]
        diagnostics = _compute_sampling_diagnostics(
            chains=chains,
            relative_interior_radius=maximum_diagnostics.relative_interior_radius,
        )
        indicator_ess, halfwidth = _indicator_quality(alternatives, chains)
        rhat_passes = (
            diagnostics.maximum_split_r_hat is None
            or diagnostics.maximum_split_r_hat <= float(args.hr_max_rhat)
        )
        passes = (
            rhat_passes
            and diagnostics.minimum_effective_sample_size >= float(args.hr_min_ess)
            and indicator_ess >= float(args.hr_min_ess)
            and halfwidth <= float(args.hr_max_ci_halfwidth)
        )
        consecutive_passes = consecutive_passes + 1 if passes else 0
        checks.append(
            {
                "sample_count": sample_count,
                "minimum_coordinate_ess": diagnostics.minimum_effective_sample_size,
                "minimum_indicator_ess": indicator_ess,
                "maximum_split_r_hat": diagnostics.maximum_split_r_hat,
                "maximum_probability_ci_halfwidth": halfwidth,
                "passes": passes,
            }
        )
        if consecutive_passes >= 2:
            selected_count = sample_count
            selected_diagnostics = diagnostics
            selected_indicator_ess = indicator_ess
            selected_halfwidth = halfwidth
            quality_achieved = True
            break
        selected_diagnostics = diagnostics
        selected_indicator_ess = indicator_ess
        selected_halfwidth = halfwidth

    timed_started = time.perf_counter()
    selected_samples, timed_diagnostics = sample_points_with_diagnostics(
        system=weight_space,
        num_samples=selected_count,
        burn_in=int(args.hr_burn_in),
        thinning=int(args.hr_thinning),
        seed=seed,
        chain_count=int(args.hr_chains),
    )
    idealized_seconds = time.perf_counter() - timed_started
    probabilities, _ = _winner_probabilities(
        alternatives,
        np.asarray(selected_samples, dtype=float),
    )
    return {
        "success": True,
        "quality_achieved": quality_achieved,
        "selected_sample_count": selected_count,
        "idealized_resumable_seconds": idealized_seconds,
        "diagnostic_scan_seconds": scan_seconds,
        "current_benchmark_wall_seconds": scan_seconds + idealized_seconds,
        "candidate_probabilities": probabilities.tolist(),
        "selected_minimum_indicator_ess": selected_indicator_ess,
        "selected_maximum_probability_ci_halfwidth": selected_halfwidth,
        "selected_diagnostics": asdict(selected_diagnostics),
        "timed_diagnostics": asdict(timed_diagnostics),
        "checks": checks,
    }


def _probability_errors(
    estimate: list[float],
    exact: list[float],
) -> dict[str, float]:
    errors = np.abs(np.asarray(estimate, dtype=float) - np.asarray(exact, dtype=float))
    return {
        "maximum_absolute_error": float(np.max(errors)),
        "mean_absolute_error": float(np.mean(errors)),
        "l1_error": float(np.sum(errors)),
    }


def benchmark_state(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=answered_queries,
    )
    exact = exact_candidate_volumes(alternatives, weight_space)
    iid = triangulation_iid_measurement(
        alternatives=alternatives,
        weight_space=weight_space,
        sample_count=int(args.iid_samples),
        seed=seed + 1,
    )
    hit_and_run = adaptive_hit_and_run_measurement(
        alternatives=alternatives,
        weight_space=weight_space,
        args=args,
        seed=seed + 2,
    )
    iid["probability_errors"] = _probability_errors(
        iid["candidate_probabilities"],
        exact["candidate_probabilities"],
    )
    hit_and_run["probability_errors"] = _probability_errors(
        hit_and_run["candidate_probabilities"],
        exact["candidate_probabilities"],
    )
    iid["triangulated_to_exact_volume_ratio"] = (
        iid["triangulated_volume"] / exact["total_volume"]
    )
    return {
        "answered_query_count": len(answered_queries),
        "constraint_count": len(weight_space.inequalities_left_side),
        "exact_volume": exact,
        "triangulation_iid": iid,
        "adaptive_hit_and_run": hit_and_run,
    }


def _central_fallback_query(
    weight_space: LinearConstraintSystem,
    query_index: int,
) -> Query:
    polytope = enumerate_polytope_vertices(weight_space)
    if polytope.status != "full_dimensional":
        raise RuntimeError(polytope.message or "cannot form a central query")
    centroid = np.mean(polytope.vertices, axis=0)
    goal_count = weight_space.variable_count
    pairs = [
        (goal_index_a, goal_index_b)
        for goal_index_a in range(goal_count)
        for goal_index_b in range(goal_index_a + 1, goal_count)
    ]
    for offset in range(len(pairs)):
        goal_index_a, goal_index_b = pairs[(query_index + offset) % len(pairs)]
        denominator = float(centroid[goal_index_b])
        if denominator > 1e-14:
            value = max(1e-12, float(centroid[goal_index_a]) / denominator)
            return Query(
                ziel_index_a=goal_index_a,
                ziel_index_b=goal_index_b,
                value=value,
            )
    raise RuntimeError("no finite central ratio query is available")


def _target_winner(
    alternatives: AlternativenMatrix,
    target_weights: list[float],
) -> int:
    entries = np.asarray(alternatives.entries, dtype=float)
    return int(np.argmax(np.asarray(target_weights, dtype=float) @ entries.T))


def _serialize_query(query: Query, answer: str, source: str) -> dict[str, Any]:
    return {
        "goal_index_a": int(query.ziel_index_a),
        "goal_index_b": int(query.ziel_index_b),
        "value": float(query.value),
        "answer": answer,
        "source": source,
    }


def _write_checkpoint(path: Path, settings: dict[str, Any], problems: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"settings": settings, "problems": problems},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    rng = np.random.default_rng(int(args.seed))
    problem_records: list[dict[str, Any]] = []
    final_milestone = max(args.milestones)
    for goal_count in args.goals:
        for problem_index in range(1, int(args.problems) + 1):
            alternatives, target_weights = generate_problem(
                rng=rng,
                goal_count=int(goal_count),
                alternative_count=int(args.alternatives),
            )
            target_winner = _target_winner(alternatives, target_weights)
            policy_seed = int(args.seed) + 100_000 * goal_count + problem_index
            config = build_policy_config(args, policy_seed)
            answered_queries: list[AnsweredQuery] = []
            query_records: list[dict[str, Any]] = []
            state_records: list[dict[str, Any]] = []
            solved_at: int | None = None
            solver_query_count = 0
            fallback_query_count = 0
            query_selection_seconds = 0.0
            with OptimizedValueFunctionSession(
                alternatives=alternatives,
                config=config,
            ) as session:
                for query_count in range(final_milestone + 1):
                    state_analysis = session.analyze_state(answered_queries)
                    if (
                        not state_analysis.is_feasible
                        or state_analysis.candidate_analysis is None
                    ):
                        raise RuntimeError("query trajectory became infeasible")
                    candidates = state_analysis.candidate_analysis.candidates
                    if target_winner not in candidates:
                        raise RuntimeError("true winner was excluded from the candidate set")
                    if len(candidates) == 1 and solved_at is None:
                        solved_at = query_count

                    if query_count in args.milestones:
                        measurement_seed = (
                            int(args.seed)
                            + 10_000_000 * goal_count
                            + 10_000 * problem_index
                            + query_count
                        )
                        state = benchmark_state(
                            alternatives=alternatives,
                            answered_queries=answered_queries,
                            args=args,
                            seed=measurement_seed,
                        )
                        state["candidate_count"] = len(candidates)
                        state_records.append(state)
                        print(
                            f"goals={goal_count} problem={problem_index:02d} "
                            f"q={query_count:02d} K={len(candidates):02d} "
                            f"vertices={state['exact_volume']['weight_space_vertices']:04d} "
                            f"vol={state['exact_volume']['seconds']:.3f}s "
                            f"tri={state['triangulation_iid']['seconds']:.3f}s "
                            f"hr={state['adaptive_hit_and_run']['idealized_resumable_seconds']:.3f}s "
                            f"hrN={state['adaptive_hit_and_run']['selected_sample_count']:04d}",
                            flush=True,
                        )
                        problem_stub = {
                            "goal_count": int(goal_count),
                            "problem_index": problem_index,
                            "target_weights": target_weights,
                            "target_winner": target_winner,
                            "alternatives": alternatives.entries,
                            "solved_at_query": solved_at,
                            "queries": query_records,
                            "states": state_records,
                        }
                        _write_checkpoint(
                            args.output_json,
                            settings,
                            problem_records + [problem_stub],
                        )

                    if query_count == final_milestone:
                        break

                    selection_started = time.perf_counter()
                    query: Query | None = None
                    source = "geometric-continuation"
                    if len(candidates) > 1:
                        result = session.compute(
                            answered_queries=answered_queries,
                            remaining_depth=1,
                        )
                        query = result.best_query
                        if query is not None:
                            source = "ratio-quantile-entropy"
                            solver_query_count += 1
                    if query is None:
                        query = _central_fallback_query(
                            weight_space=state_analysis.weight_space,
                            query_index=query_count,
                        )
                        fallback_query_count += 1
                    query_selection_seconds += time.perf_counter() - selection_started
                    answer = classify_query_answer(
                        weights=target_weights,
                        query=query,
                        equality_tol=0.0,
                    )
                    if answer == "=":
                        query = Query(
                            ziel_index_a=int(query.ziel_index_a),
                            ziel_index_b=int(query.ziel_index_b),
                            value=float(query.value) * (1.0 + 1e-10),
                        )
                        answer = classify_query_answer(
                            weights=target_weights,
                            query=query,
                            equality_tol=0.0,
                        )
                    answered_queries.append(query.answer(answer))
                    query_records.append(_serialize_query(query, answer, source))

            candidates_at_final_milestone = int(
                next(
                    state["candidate_count"]
                    for state in state_records
                    if state["answered_query_count"] == final_milestone
                )
            )
            # The benchmark states must be identical regardless of how long a
            # hard problem takes to solve.  Only after q=20 do we continue the
            # still ambiguous cases with target-independent central cuts.
            while (
                solved_at is None
                and len(answered_queries) < int(args.max_solve_queries)
            ):
                weight_space = build_weight_space(
                    goal_count=int(goal_count),
                    answered_queries=answered_queries,
                )
                candidates = compute_candidate_set(
                    alternatives=alternatives,
                    weight_space=weight_space,
                )
                if target_winner not in candidates:
                    raise RuntimeError("true winner was excluded during solve continuation")
                if len(candidates) == 1:
                    solved_at = len(answered_queries)
                    break
                selection_started = time.perf_counter()
                query = _central_fallback_query(
                    weight_space=weight_space,
                    query_index=len(answered_queries),
                )
                query_selection_seconds += time.perf_counter() - selection_started
                answer = classify_query_answer(
                    weights=target_weights,
                    query=query,
                    equality_tol=0.0,
                )
                if answer == "=":
                    query = Query(
                        ziel_index_a=int(query.ziel_index_a),
                        ziel_index_b=int(query.ziel_index_b),
                        value=float(query.value) * (1.0 + 1e-10),
                    )
                    answer = classify_query_answer(
                        weights=target_weights,
                        query=query,
                        equality_tol=0.0,
                    )
                answered_queries.append(query.answer(answer))
                query_records.append(
                    _serialize_query(query, answer, "geometric-solve-continuation")
                )
                fallback_query_count += 1

            final_candidates = compute_candidate_set(
                alternatives=alternatives,
                weight_space=build_weight_space(
                    goal_count=int(goal_count),
                    answered_queries=answered_queries,
                ),
            )
            if len(final_candidates) == 1 and solved_at is None:
                solved_at = len(answered_queries)
            record = {
                "goal_count": int(goal_count),
                "problem_index": problem_index,
                "target_weights": target_weights,
                "target_winner": target_winner,
                "alternatives": alternatives.entries,
                "solved_at_query": solved_at,
                "solved_by_final_milestone": candidates_at_final_milestone == 1,
                "fully_solved": len(final_candidates) == 1,
                "total_query_count": len(answered_queries),
                "final_candidates": final_candidates,
                "solver_query_count": solver_query_count,
                "fallback_query_count": fallback_query_count,
                "query_selection_seconds": query_selection_seconds,
                "queries": query_records,
                "states": state_records,
            }
            problem_records.append(record)
            _write_checkpoint(args.output_json, settings, problem_records)
    return problem_records


def main() -> None:
    args = parse_args()
    validate_args(args)
    records = run_benchmark(args)
    solved = sum(bool(record["solved_by_final_milestone"]) for record in records)
    print(
        f"wrote {args.output_json}; solved by q={max(args.milestones)}: "
        f"{solved}/{len(records)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
