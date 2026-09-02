from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.models import AlternativenMatrix, Query
from multistep.src.query_probability import ANSWER_OPTIONS
from multistep.src.query_value_function import (
    build_linear_query_values,
    estimate_depth_two_query_value_from_samples,
    evaluate_query_value_curve_for_goal_pair,
    filter_samples_for_query_answer,
    get_ordered_goal_pairs,
)
from multistep.src.sampling import sample_points_from_constraint_system
from multistep.src.weight_space import build_weight_space


@dataclass(frozen=True)
class DepthTwoRuntimeRow:
    index: int
    query_value: float
    elapsed_seconds: float
    expected_candidate_count: float
    branch_sample_counts: tuple[int, int, int]

    @property
    def non_empty_branch_count(self) -> int:
        return sum(1 for count in self.branch_sample_counts if count > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze runtime of the adaptive depth-2 query value curve.",
    )
    parser.add_argument("--goals", type=int, default=3)
    parser.add_argument("--alternatives", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thinning", type=int, default=4)
    parser.add_argument("--outer-steps", type=int, default=41)
    parser.add_argument("--inner-steps", type=int, default=11)
    parser.add_argument("--max-s", type=float, default=10.0)
    parser.add_argument("--goal-a", type=int, default=0)
    parser.add_argument("--goal-b", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument(
        "--skip-onestep",
        action="store_true",
        help="Skip timing the exact one-step curve.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.goals < 2:
        raise ValueError("--goals must be at least 2")

    if args.alternatives < 2:
        raise ValueError("--alternatives must be at least 2")

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    if args.burn_in < 0:
        raise ValueError("--burn-in must not be negative")

    if args.thinning <= 0:
        raise ValueError("--thinning must be positive")

    if args.outer_steps <= 0:
        raise ValueError("--outer-steps must be positive")

    if args.inner_steps <= 0:
        raise ValueError("--inner-steps must be positive")

    if args.max_s < 0.0:
        raise ValueError("--max-s must not be negative")

    if args.goal_a == args.goal_b:
        raise ValueError("--goal-a and --goal-b must be different")

    if not 0 <= args.goal_a < args.goals:
        raise ValueError("--goal-a is out of range")

    if not 0 <= args.goal_b < args.goals:
        raise ValueError("--goal-b is out of range")

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    if args.top <= 0:
        raise ValueError("--top must be positive")


def generate_random_problem(
    goal_count: int,
    alternative_count: int,
    seed: int,
) -> AlternativenMatrix:
    rng = np.random.default_rng(seed)
    entries = np.round(
        rng.uniform(0.0, 1.0, size=(alternative_count, goal_count)),
        2,
    )
    return AlternativenMatrix(entries=entries.tolist())


def time_call(function):
    start = perf_counter()
    result = function()
    return perf_counter() - start, result


def compute_branch_sample_counts(
    samples: list[list[float]],
    query: Query,
) -> tuple[int, int, int]:
    return tuple(
        len(
            filter_samples_for_query_answer(
                samples=samples,
                query=query,
                answer=answer,
            )
        )
        for answer in ANSWER_OPTIONS
    )


def run_depth_two_curve(
    alternatives: AlternativenMatrix,
    samples: list[list[float]],
    goal_pair: tuple[int, int],
    query_values: list[float],
    inner_steps: int,
    max_s: float,
) -> list[DepthTwoRuntimeRow]:
    rows: list[DepthTwoRuntimeRow] = []

    for index, query_value in enumerate(query_values):
        query = Query(
            ziel_index_a=goal_pair[0],
            ziel_index_b=goal_pair[1],
            value=float(query_value),
        )
        branch_sample_counts = compute_branch_sample_counts(
            samples=samples,
            query=query,
        )
        elapsed_seconds, evaluation = time_call(
            lambda: estimate_depth_two_query_value_from_samples(
                alternatives=alternatives,
                query=query,
                samples=samples,
                inner_query_value_steps=inner_steps,
                max_query_value=max_s,
            )
        )
        rows.append(
            DepthTwoRuntimeRow(
                index=index,
                query_value=float(query_value),
                elapsed_seconds=elapsed_seconds,
                expected_candidate_count=evaluation.expected_candidate_count,
                branch_sample_counts=branch_sample_counts,
            )
        )

    return rows


def format_seconds(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"

    return f"{seconds:.3f} s"


def print_runtime_summary(
    rows: list[DepthTwoRuntimeRow],
    total_seconds: float,
    goal_count: int,
    inner_steps: int,
    repetitions: int,
    repeated_total_seconds: list[float],
) -> None:
    goal_pair_count = len(get_ordered_goal_pairs(goal_count))
    theoretical_second_stage_queries = (
        len(rows) * len(ANSWER_OPTIONS) * goal_pair_count * inner_steps
    )
    actual_second_stage_queries = sum(
        row.non_empty_branch_count * goal_pair_count * inner_steps
        for row in rows
    )
    row_times = [row.elapsed_seconds for row in rows]
    best_row = min(rows, key=lambda row: row.expected_candidate_count)

    print("\nTiefe 2")
    print(f"  total: {format_seconds(total_seconds)}")
    if repetitions > 1:
        mean_total = statistics.mean(repeated_total_seconds)
        print(f"  total mean over repetitions: {format_seconds(mean_total)}")
        print(
            "  totals: "
            + ", ".join(format_seconds(seconds) for seconds in repeated_total_seconds)
        )
    print(f"  per outer s avg: {format_seconds(statistics.mean(row_times))}")
    print(f"  per outer s min: {format_seconds(min(row_times))}")
    print(f"  per outer s max: {format_seconds(max(row_times))}")
    print(
        "  second-stage query evaluations: "
        f"{actual_second_stage_queries} actual / "
        f"{theoretical_second_stage_queries} theoretical"
    )
    print(
        "  second-stage search per non-empty branch: "
        f"{goal_pair_count} goal pairs * {inner_steps} s2-values "
        f"= {goal_pair_count * inner_steps}"
    )
    print(
        "  best tested first s: "
        f"{best_row.query_value:.6g} with E2[K]={best_row.expected_candidate_count:.6g}"
    )


def print_slowest_rows(rows: list[DepthTwoRuntimeRow], top: int) -> None:
    slowest_rows = sorted(
        rows,
        key=lambda row: row.elapsed_seconds,
        reverse=True,
    )[:top]

    print(f"\nSlowest {len(slowest_rows)} outer s-values")
    print("  idx       s        time       E2[K]    n_<    n_=    n_>")
    for row in slowest_rows:
        less_count, equal_count, greater_count = row.branch_sample_counts
        print(
            f"  {row.index:>3}  "
            f"{row.query_value:>8.4g}  "
            f"{format_seconds(row.elapsed_seconds):>9}  "
            f"{row.expected_candidate_count:>8.4g}  "
            f"{less_count:>5}  {equal_count:>5}  {greater_count:>5}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    goal_pair = (args.goal_a, args.goal_b)
    query_values = build_linear_query_values(
        lower=0.0,
        upper=args.max_s,
        steps=args.outer_steps,
    )

    print("Configuration")
    print(f"  goals: {args.goals}")
    print(f"  alternatives: {args.alternatives}")
    print(f"  samples: {args.samples}")
    print(f"  burn-in: {args.burn_in}")
    print(f"  thinning: {args.thinning}")
    print(f"  first goal pair: {goal_pair}")
    print(f"  outer s-values: {args.outer_steps}")
    print(f"  inner s2-values: {args.inner_steps}")
    print(f"  max s: {args.max_s}")
    print(f"  seed: {args.seed}")

    problem_seconds, alternatives = time_call(
        lambda: generate_random_problem(
            goal_count=args.goals,
            alternative_count=args.alternatives,
            seed=args.seed,
        )
    )
    weight_space = build_weight_space(
        goal_count=args.goals,
        answered_queries=[],
    )
    sampling_seconds, samples = time_call(
        lambda: sample_points_from_constraint_system(
            system=weight_space,
            num_samples=args.samples,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=args.seed + 10_000,
        )
    )

    print("\nSetup")
    print(f"  problem generation: {format_seconds(problem_seconds)}")
    print(f"  sampling W(T): {format_seconds(sampling_seconds)}")

    if not args.skip_onestep:
        onestep_seconds, onestep_evaluations = time_call(
            lambda: evaluate_query_value_curve_for_goal_pair(
                alternatives=alternatives,
                answered_queries=[],
                samples=samples,
                goal_pair=goal_pair,
                query_values=query_values,
            )
        )
        best_onestep = min(
            onestep_evaluations,
            key=lambda evaluation: evaluation.expected_candidate_count,
        )
        print("\nTiefe 1")
        print(f"  total: {format_seconds(onestep_seconds)}")
        print(
            "  per outer s avg: "
            f"{format_seconds(onestep_seconds / len(query_values))}"
        )
        print(
            "  best tested s: "
            f"{best_onestep.query.value:.6g} "
            f"with E[K]={best_onestep.expected_candidate_count:.6g}"
        )

    repeated_total_seconds: list[float] = []
    rows: list[DepthTwoRuntimeRow] = []
    total_seconds = 0.0

    for _ in range(args.repetitions):
        total_seconds, rows = time_call(
            lambda: run_depth_two_curve(
                alternatives=alternatives,
                samples=samples,
                goal_pair=goal_pair,
                query_values=query_values,
                inner_steps=args.inner_steps,
                max_s=args.max_s,
            )
        )
        repeated_total_seconds.append(total_seconds)

    print_runtime_summary(
        rows=rows,
        total_seconds=total_seconds,
        goal_count=args.goals,
        inner_steps=args.inner_steps,
        repetitions=args.repetitions,
        repeated_total_seconds=repeated_total_seconds,
    )
    print_slowest_rows(rows=rows, top=args.top)


if __name__ == "__main__":
    main()
