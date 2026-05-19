from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.models import AlternativenMatrix as MultistepAlternativenMatrix
from multistep.src.models import Query as MultistepQuery
from multistep.src.value_function import MultistepConfig, compute_value_function
from onestep.src.algorithmus import AlgorithmusOutput
from onestep.src.build_ungleichungssysteme import build_W, build_optimal_region_in_W
from onestep.src.io_models import AlternativenMatrix as OnestepAlternativenMatrix
from onestep.src.io_models import Query as OnestepQuery
from onestep.src.io_models import TerminationResult
from onestep.src.query_bewertung import (
    build_zielpaar_intervalle_lookup,
    compute_query_info,
    filter_already_answered_queries,
    filter_informative_query_infos,
)
from onestep.src.query_kandidaten import compute_all_query_kandidaten
from onestep.src.ratio_intervalle import compute_all_ratio_intervals
from onestep.src.sampling import (
    estimate_optimality_shares,
    sample_points_from_ungleichungssystem,
)
from onestep.src.termination import (
    all_candidates_have_same_utility_values_in_W,
    build_no_informative_query_termination_result,
    build_one_remaining_candidate_termination_result,
    build_same_utility_termination_result,
)


ComparableQuery: TypeAlias = OnestepQuery | MultistepQuery


@dataclass(frozen=True)
class ComparisonRow:
    problem_index: int
    goal_count: int
    alternative_count: int
    onestep_output: AlgorithmusOutput
    depth_one_query: MultistepQuery | None

    @property
    def onestep_query(self) -> OnestepQuery | None:
        if isinstance(self.onestep_output, OnestepQuery):
            return self.onestep_output

        return None

    @property
    def onestep_terminated(self) -> bool:
        return isinstance(self.onestep_output, TerminationResult)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the legacy one-step algorithm with the multistep value "
            "function at remaining_depth=1 on random problems."
        ),
    )
    parser.add_argument("--problems", type=int, default=100)
    parser.add_argument("--min-goals", type=int, default=3)
    parser.add_argument("--max-goals", type=int, default=7)
    parser.add_argument("--min-alternatives", type=int, default=3)
    parser.add_argument("--max-alternatives", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thinning", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--query-epsilon", type=float, default=1e-3)
    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--rel-tol", type=float, default=1e-8)
    parser.add_argument("--show-mismatches", type=int, default=20)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.problems <= 0:
        raise ValueError("--problems must be positive")

    if args.min_goals < 2:
        raise ValueError("--min-goals must be at least 2")

    if args.min_goals > args.max_goals:
        raise ValueError("--min-goals must not be greater than --max-goals")

    if args.min_alternatives < 2:
        raise ValueError("--min-alternatives must be at least 2")

    if args.min_alternatives > args.max_alternatives:
        raise ValueError(
            "--min-alternatives must not be greater than --max-alternatives"
        )

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    if args.burn_in < 0:
        raise ValueError("--burn-in must not be negative")

    if args.thinning <= 0:
        raise ValueError("--thinning must be positive")

    if args.query_epsilon <= 0.0:
        raise ValueError("--query-epsilon must be positive")

    if args.abs_tol < 0.0:
        raise ValueError("--abs-tol must not be negative")

    if args.rel_tol < 0.0:
        raise ValueError("--rel-tol must not be negative")


def generate_problem_entries(
    goal_count: int,
    alternative_count: int,
    rng: np.random.Generator,
) -> list[list[float]]:
    return rng.uniform(
        0.0,
        1.0,
        size=(alternative_count, goal_count),
    ).astype(float).tolist()


def run_onestep_algorithmus_with_sampling_config(
    alternativen_matrix: OnestepAlternativenMatrix,
    sample_count: int,
    burn_in: int,
    thinning: int,
    seed: int | None,
    query_epsilon: float,
) -> AlgorithmusOutput:
    answered_queries = []
    goal_count = alternativen_matrix.get_anzahl_spalten()
    weight_space = build_W(goal_count, answered_queries)

    if not weight_space.is_feasible():
        raise ValueError("W is infeasible")

    candidates: list[int] = []
    for alternative_index in range(alternativen_matrix.get_anzahl_zeilen()):
        optimal_region = build_optimal_region_in_W(
            alternativen_matrix,
            weight_space,
            alternative_index,
        )
        if optimal_region.is_feasible():
            candidates.append(alternative_index)

    if not candidates:
        raise ValueError("no candidates found in W")

    if len(candidates) == 1:
        return build_one_remaining_candidate_termination_result(candidates[0])

    if all_candidates_have_same_utility_values_in_W(
        alternativen_matrix=alternativen_matrix,
        W=weight_space,
        kandidatenmenge=candidates,
    ):
        return build_same_utility_termination_result(candidates)

    goal_pair_intervals = compute_all_ratio_intervals(
        alternativen_matrix=alternativen_matrix,
        answered_queries=answered_queries,
        kandidatenmenge=candidates,
    )
    query_candidates = compute_all_query_kandidaten(
        zielpaar_intervalle_liste=goal_pair_intervals,
        epsilon=query_epsilon,
    )
    query_candidates = filter_already_answered_queries(
        query_candidates,
        answered_queries,
    )

    samples = sample_points_from_ungleichungssystem(
        weight_space,
        num_samples=sample_count,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed,
    )
    goal_pair_intervals_lookup = build_zielpaar_intervalle_lookup(
        goal_pair_intervals
    )

    query_infos = []
    for query in query_candidates:
        query_info = compute_query_info(
            query=query,
            samples=samples,
            zielpaar_intervalle_lookup=goal_pair_intervals_lookup,
        )
        if query_info is not None:
            query_infos.append(query_info)

    informative_query_infos = filter_informative_query_infos(
        query_infos=query_infos,
        kandidatenmenge=set(candidates),
    )
    if not informative_query_infos:
        optimality_shares = estimate_optimality_shares(
            alternativen_matrix=alternativen_matrix,
            samples=samples,
            remaining_candidates=candidates,
        )
        return build_no_informative_query_termination_result(
            kandidatenmenge=candidates,
            optimality_shares=optimality_shares,
        )

    best_query_info = min(
        informative_query_infos,
        key=lambda query_info: query_info.expected_kandidatenanzahl,
    )
    return best_query_info.query


def run_depth_one_value_function(
    alternatives: MultistepAlternativenMatrix,
    sample_count: int,
    burn_in: int,
    thinning: int,
    seed: int | None,
    query_epsilon: float,
) -> MultistepQuery | None:
    result = compute_value_function(
        alternatives=alternatives,
        answered_queries=[],
        remaining_depth=1,
        config=MultistepConfig(
            sample_count=sample_count,
            burn_in=burn_in,
            thinning=thinning,
            random_seed=seed,
            query_epsilon=query_epsilon,
        ),
    )
    return result.best_query


def are_equivalent_queries(
    left: ComparableQuery | None,
    right: ComparableQuery | None,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    if left is None or right is None:
        return left is None and right is None

    if (
        int(left.ziel_index_a) == int(right.ziel_index_a)
        and int(left.ziel_index_b) == int(right.ziel_index_b)
        and math.isclose(
            float(left.value),
            float(right.value),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    ):
        return True

    if float(left.value) <= 0.0 or float(right.value) <= 0.0:
        return False

    return (
        int(left.ziel_index_a) == int(right.ziel_index_b)
        and int(left.ziel_index_b) == int(right.ziel_index_a)
        and math.isclose(
            float(left.value),
            1.0 / float(right.value),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    )


def format_query(query: ComparableQuery | None) -> str:
    if query is None:
        return "<none>"

    return (
        f"({int(query.ziel_index_a)}, {int(query.ziel_index_b)}, "
        f"{float(query.value):.8g})"
    )


def format_onestep_output(output: AlgorithmusOutput) -> str:
    if isinstance(output, OnestepQuery):
        return format_query(output)

    return f"termination:{output.reason}"


def run_comparison(args: argparse.Namespace) -> list[ComparisonRow]:
    rng = np.random.default_rng(args.seed)
    rows: list[ComparisonRow] = []

    for problem_index in range(args.problems):
        goal_count = int(rng.integers(args.min_goals, args.max_goals + 1))
        alternative_count = int(
            rng.integers(args.min_alternatives, args.max_alternatives + 1)
        )
        entries = generate_problem_entries(
            goal_count=goal_count,
            alternative_count=alternative_count,
            rng=rng,
        )
        problem_seed = args.seed + problem_index
        onestep_output = run_onestep_algorithmus_with_sampling_config(
            alternativen_matrix=OnestepAlternativenMatrix(entries=entries),
            sample_count=args.samples,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=problem_seed,
            query_epsilon=args.query_epsilon,
        )
        depth_one_query = run_depth_one_value_function(
            alternatives=MultistepAlternativenMatrix(entries=entries),
            sample_count=args.samples,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=problem_seed,
            query_epsilon=args.query_epsilon,
        )
        rows.append(
            ComparisonRow(
                problem_index=problem_index,
                goal_count=goal_count,
                alternative_count=alternative_count,
                onestep_output=onestep_output,
                depth_one_query=depth_one_query,
            )
        )

    return rows


def print_summary(rows: list[ComparisonRow], args: argparse.Namespace) -> None:
    comparable_rows = [
        row
        for row in rows
        if row.onestep_query is not None and row.depth_one_query is not None
    ]
    matching_rows = [
        row
        for row in comparable_rows
        if are_equivalent_queries(
            row.onestep_query,
            row.depth_one_query,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
    ]
    mismatching_rows = [
        row
        for row in comparable_rows
        if not are_equivalent_queries(
            row.onestep_query,
            row.depth_one_query,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
    ]
    onestep_terminations = [row for row in rows if row.onestep_terminated]
    depth_one_without_query = [
        row
        for row in rows
        if row.depth_one_query is None and not row.onestep_terminated
    ]

    print("Configuration")
    print(f"  problems: {args.problems}")
    print(f"  goals: {args.min_goals}..{args.max_goals}")
    print(f"  alternatives: {args.min_alternatives}..{args.max_alternatives}")
    print(f"  samples: {args.samples}")
    print(f"  burn-in: {args.burn_in}")
    print(f"  thinning: {args.thinning}")
    print(f"  seed: {args.seed}")
    print(f"  query epsilon: {args.query_epsilon}")
    print()
    print("Summary")
    print(f"  both returned query: {len(comparable_rows)}")
    print(f"  equivalent query: {len(matching_rows)}")
    print(f"  different query: {len(mismatching_rows)}")
    print(f"  one-step terminations: {len(onestep_terminations)}")
    print(f"  depth-1 without query while one-step returned query: {len(depth_one_without_query)}")

    if comparable_rows:
        match_rate = len(matching_rows) / len(comparable_rows)
        print(f"  match rate among comparable problems: {match_rate:.2%}")

    if mismatching_rows:
        print()
        print(f"First {min(args.show_mismatches, len(mismatching_rows))} mismatches")
        print("  idx goals alts  one-step                 depth-1")
        for row in mismatching_rows[: args.show_mismatches]:
            print(
                f"  {row.problem_index:>3} "
                f"{row.goal_count:>5} "
                f"{row.alternative_count:>4}  "
                f"{format_query(row.onestep_query):<24} "
                f"{format_query(row.depth_one_query)}"
            )

    if onestep_terminations:
        print()
        print(f"First {min(args.show_mismatches, len(onestep_terminations))} one-step terminations")
        print("  idx goals alts  one-step                 depth-1")
        for row in onestep_terminations[: args.show_mismatches]:
            print(
                f"  {row.problem_index:>3} "
                f"{row.goal_count:>5} "
                f"{row.alternative_count:>4}  "
                f"{format_onestep_output(row.onestep_output):<24} "
                f"{format_query(row.depth_one_query)}"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    rows = run_comparison(args)
    print_summary(rows=rows, args=args)


if __name__ == "__main__":
    main()
