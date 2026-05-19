from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.candidates import compute_candidate_set
from multistep.src.models import AlternativenMatrix, Query, QueryOperator
from multistep.src.onestep_query_candidates import compute_onestep_query_candidates
from multistep.src.query_probability import ANSWER_OPTIONS
from multistep.src.ratio_intervals import (
    GoalPairRatioIntervals,
    RatioInterval,
    compute_all_ratio_intervals,
)
from multistep.src.weight_space import build_weight_space


@dataclass(frozen=True)
class BranchDifference:
    problem_index: int
    goal_count: int
    alternative_count: int
    query: Query
    answer: QueryOperator
    ratio_candidates: frozenset[int]
    lp_candidates: frozenset[int]

    @property
    def ratio_only(self) -> frozenset[int]:
        return self.ratio_candidates - self.lp_candidates

    @property
    def lp_only(self) -> frozenset[int]:
        return self.lp_candidates - self.ratio_candidates

    @property
    def symmetric_difference_size(self) -> int:
        return len(self.ratio_candidates ^ self.lp_candidates)


@dataclass(frozen=True)
class ProblemSummary:
    problem_index: int
    goal_count: int
    alternative_count: int
    candidate_count: int
    query_count: int
    differing_query_count: int
    differing_branch_count: int
    max_branch_difference_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how often branch candidate sets differ between the ratio "
            "rules from one-step and LP-based child weight spaces."
        ),
    )
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--min-goals", type=int, default=3)
    parser.add_argument("--max-goals", type=int, default=7)
    parser.add_argument("--min-alternatives", type=int, default=3)
    parser.add_argument("--max-alternatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--query-epsilon", type=float, default=1e-3)
    parser.add_argument("--show-examples", type=int, default=20)
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
        raise ValueError("--min-alternatives must not be greater than --max-alternatives")

    if args.query_epsilon <= 0.0:
        raise ValueError("--query-epsilon must be positive")

    if args.show_examples < 0:
        raise ValueError("--show-examples must not be negative")


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


def build_ratio_interval_lookup(
    ratio_intervals: list[GoalPairRatioIntervals],
) -> dict[tuple[int, int], GoalPairRatioIntervals]:
    lookup: dict[tuple[int, int], GoalPairRatioIntervals] = {}
    for goal_pair_intervals in ratio_intervals:
        key = (
            int(goal_pair_intervals.goal_index_a),
            int(goal_pair_intervals.goal_index_b),
        )
        if key in lookup:
            raise ValueError("ratio_intervals must not contain duplicate goal pairs")

        lookup[key] = goal_pair_intervals

    return lookup


def compute_ratio_branch_candidate_sets(
    query: Query,
    goal_pair_intervals: GoalPairRatioIntervals,
) -> dict[QueryOperator, frozenset[int]]:
    query_value = float(query.value)
    candidate_sets: dict[QueryOperator, set[int]] = {
        "<": set(),
        "=": set(),
        ">": set(),
    }

    for candidate_index, ratio_interval in goal_pair_intervals.intervals_by_candidate.items():
        lower_value = _get_lower_ratio_value_or_none(ratio_interval)
        if lower_value is None:
            continue

        upper_status = ratio_interval.upper.status

        if query_value > lower_value:
            candidate_sets["<"].add(int(candidate_index))

        if upper_status == "optimal":
            if ratio_interval.upper.optimal_value is None:
                raise RuntimeError("optimal upper ratio bound has no optimal_value")

            upper_value = float(ratio_interval.upper.optimal_value)
            if query_value < upper_value:
                candidate_sets[">"].add(int(candidate_index))

            if lower_value <= query_value <= upper_value:
                candidate_sets["="].add(int(candidate_index))
        elif upper_status == "unbounded":
            candidate_sets[">"].add(int(candidate_index))

            if lower_value <= query_value:
                candidate_sets["="].add(int(candidate_index))
        elif upper_status == "infeasible":
            continue
        else:
            raise ValueError(f"unknown upper ratio bound status: {upper_status}")

    return {
        answer: frozenset(candidates)
        for answer, candidates in candidate_sets.items()
    }


def _get_lower_ratio_value_or_none(ratio_interval: RatioInterval) -> float | None:
    if ratio_interval.lower.status == "unbounded":
        raise ValueError("lower ratio bound must not be unbounded")

    if ratio_interval.lower.status == "infeasible":
        return None

    if ratio_interval.lower.optimal_value is None:
        raise RuntimeError("optimal lower ratio bound has no optimal_value")

    return float(ratio_interval.lower.optimal_value)


def compute_lp_branch_candidate_sets(
    alternatives: AlternativenMatrix,
    query: Query,
) -> dict[QueryOperator, frozenset[int]]:
    candidate_sets: dict[QueryOperator, frozenset[int]] = {}

    for answer in ANSWER_OPTIONS:
        child_weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=[query.answer(answer)],
        )
        if child_weight_space.is_feasible():
            candidates = compute_candidate_set(
                alternatives=alternatives,
                weight_space=child_weight_space,
            )
        else:
            candidates = []

        candidate_sets[answer] = frozenset(candidates)

    return candidate_sets


def analyze_problem(
    problem_index: int,
    goal_count: int,
    alternative_count: int,
    entries: list[list[float]],
    query_epsilon: float,
) -> tuple[ProblemSummary, list[BranchDifference]]:
    alternatives = AlternativenMatrix(entries=entries)
    weight_space = build_weight_space(
        goal_count=goal_count,
        answered_queries=[],
    )
    candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )
    ratio_intervals = compute_all_ratio_intervals(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
    )
    ratio_interval_lookup = build_ratio_interval_lookup(ratio_intervals)
    query_candidates = compute_onestep_query_candidates(
        goal_pair_ratio_intervals=ratio_intervals,
        epsilon=query_epsilon,
    )

    differences: list[BranchDifference] = []
    differing_query_keys: set[tuple[int, int, float]] = set()

    for query in query_candidates:
        ratio_branch_candidate_sets = compute_ratio_branch_candidate_sets(
            query=query,
            goal_pair_intervals=ratio_interval_lookup[
                (int(query.ziel_index_a), int(query.ziel_index_b))
            ],
        )
        lp_branch_candidate_sets = compute_lp_branch_candidate_sets(
            alternatives=alternatives,
            query=query,
        )

        for answer in ANSWER_OPTIONS:
            ratio_candidates = ratio_branch_candidate_sets[answer]
            lp_candidates = lp_branch_candidate_sets[answer]
            if ratio_candidates == lp_candidates:
                continue

            differences.append(
                BranchDifference(
                    problem_index=problem_index,
                    goal_count=goal_count,
                    alternative_count=alternative_count,
                    query=query,
                    answer=answer,
                    ratio_candidates=ratio_candidates,
                    lp_candidates=lp_candidates,
                )
            )
            differing_query_keys.add(
                (
                    int(query.ziel_index_a),
                    int(query.ziel_index_b),
                    round(float(query.value), 12),
                )
            )

    summary = ProblemSummary(
        problem_index=problem_index,
        goal_count=goal_count,
        alternative_count=alternative_count,
        candidate_count=len(candidates),
        query_count=len(query_candidates),
        differing_query_count=len(differing_query_keys),
        differing_branch_count=len(differences),
        max_branch_difference_size=max(
            (difference.symmetric_difference_size for difference in differences),
            default=0,
        ),
    )
    return summary, differences


def format_query(query: Query) -> str:
    return (
        f"({int(query.ziel_index_a)}, {int(query.ziel_index_b)}, "
        f"{float(query.value):.8g})"
    )


def print_summary(
    problem_summaries: list[ProblemSummary],
    differences: list[BranchDifference],
    args: argparse.Namespace,
) -> None:
    query_count = sum(summary.query_count for summary in problem_summaries)
    branch_count = query_count * len(ANSWER_OPTIONS)
    differing_query_count = sum(summary.differing_query_count for summary in problem_summaries)
    differing_branch_count = len(differences)
    total_symmetric_difference_size = sum(
        difference.symmetric_difference_size for difference in differences
    )

    print("Configuration")
    print(f"  problems: {args.problems}")
    print(f"  goals: {args.min_goals}..{args.max_goals}")
    print(f"  alternatives: {args.min_alternatives}..{args.max_alternatives}")
    print(f"  seed: {args.seed}")
    print(f"  query epsilon: {args.query_epsilon}")
    print()
    print("Summary")
    print(f"  query candidates: {query_count}")
    print(f"  branch comparisons: {branch_count}")
    print(f"  queries with any difference: {differing_query_count}")
    print(f"  branches with difference: {differing_branch_count}")
    if query_count:
        print(f"  query difference rate: {differing_query_count / query_count:.2%}")
    if branch_count:
        print(f"  branch difference rate: {differing_branch_count / branch_count:.2%}")
    if differing_branch_count:
        print(
            "  avg symmetric difference per differing branch: "
            f"{total_symmetric_difference_size / differing_branch_count:.3f}"
        )
        print(
            "  max symmetric difference: "
            f"{max(difference.symmetric_difference_size for difference in differences)}"
        )

    print()
    print("Per answer")
    print("  ans branches diff avg_sym ratio_only lp_only")
    for answer in ANSWER_OPTIONS:
        answer_differences = [
            difference for difference in differences if difference.answer == answer
        ]
        answer_branch_count = query_count
        answer_total_symmetric_difference_size = sum(
            difference.symmetric_difference_size
            for difference in answer_differences
        )
        ratio_only_count = sum(
            len(difference.ratio_only)
            for difference in answer_differences
        )
        lp_only_count = sum(
            len(difference.lp_only)
            for difference in answer_differences
        )
        avg_symmetric_difference = (
            answer_total_symmetric_difference_size / len(answer_differences)
            if answer_differences
            else 0.0
        )
        print(
            f"  {answer:>3} "
            f"{answer_branch_count:>8} "
            f"{len(answer_differences):>4} "
            f"{avg_symmetric_difference:>7.3f} "
            f"{ratio_only_count:>10} "
            f"{lp_only_count:>7}"
        )

    print()
    print("Per goal count")
    print("  goals problems queries branches diff_q diff_b diff_q_rate diff_b_rate max_diff")
    goal_counts = sorted({summary.goal_count for summary in problem_summaries})
    for goal_count in goal_counts:
        summaries_for_goal_count = [
            summary
            for summary in problem_summaries
            if summary.goal_count == goal_count
        ]
        differences_for_goal_count = [
            difference
            for difference in differences
            if difference.goal_count == goal_count
        ]
        goal_query_count = sum(
            summary.query_count
            for summary in summaries_for_goal_count
        )
        goal_branch_count = goal_query_count * len(ANSWER_OPTIONS)
        goal_differing_query_count = sum(
            summary.differing_query_count
            for summary in summaries_for_goal_count
        )
        goal_differing_branch_count = len(differences_for_goal_count)
        goal_max_difference = max(
            (
                difference.symmetric_difference_size
                for difference in differences_for_goal_count
            ),
            default=0,
        )
        goal_query_difference_rate = (
            goal_differing_query_count / goal_query_count
            if goal_query_count
            else 0.0
        )
        goal_branch_difference_rate = (
            goal_differing_branch_count / goal_branch_count
            if goal_branch_count
            else 0.0
        )
        print(
            f"  {goal_count:>5} "
            f"{len(summaries_for_goal_count):>8} "
            f"{goal_query_count:>7} "
            f"{goal_branch_count:>8} "
            f"{goal_differing_query_count:>6} "
            f"{goal_differing_branch_count:>6} "
            f"{goal_query_difference_rate:>11.2%} "
            f"{goal_branch_difference_rate:>11.2%} "
            f"{goal_max_difference:>8}"
        )

    print()
    print("Per problem")
    print("  idx goals alts cand queries diff_q diff_b max_diff")
    for summary in problem_summaries:
        print(
            f"  {summary.problem_index:>3} "
            f"{summary.goal_count:>5} "
            f"{summary.alternative_count:>4} "
            f"{summary.candidate_count:>4} "
            f"{summary.query_count:>7} "
            f"{summary.differing_query_count:>6} "
            f"{summary.differing_branch_count:>6} "
            f"{summary.max_branch_difference_size:>8}"
        )

    if differences and args.show_examples:
        print()
        print(f"First {min(args.show_examples, len(differences))} branch differences")
        print("  idx goals alts query                    ans ratio        lp           ratio_only lp_only")
        for difference in differences[: args.show_examples]:
            print(
                f"  {difference.problem_index:>3} "
                f"{difference.goal_count:>5} "
                f"{difference.alternative_count:>4} "
                f"{format_query(difference.query):<24} "
                f"{difference.answer:>3} "
                f"{sorted(difference.ratio_candidates)!s:<12} "
                f"{sorted(difference.lp_candidates)!s:<12} "
                f"{sorted(difference.ratio_only)!s:<10} "
                f"{sorted(difference.lp_only)}"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    problem_summaries: list[ProblemSummary] = []
    all_differences: list[BranchDifference] = []

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
        summary, differences = analyze_problem(
            problem_index=problem_index,
            goal_count=goal_count,
            alternative_count=alternative_count,
            entries=entries,
            query_epsilon=args.query_epsilon,
        )
        problem_summaries.append(summary)
        all_differences.extend(differences)

    print_summary(
        problem_summaries=problem_summaries,
        differences=all_differences,
        args=args,
    )


if __name__ == "__main__":
    main()
