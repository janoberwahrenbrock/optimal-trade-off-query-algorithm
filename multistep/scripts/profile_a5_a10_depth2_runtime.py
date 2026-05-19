from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.candidates import compute_candidate_set
from multistep.src.grid_query_candidates import GridSpacing, compute_grid_query_candidates
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from multistep.src.onestep_query_candidates import (
    QUERY_EPSILON,
    compute_onestep_query_candidates,
)
from multistep.src.query_probability import (
    ANSWER_OPTIONS,
    estimate_query_answer_probabilities,
)
from multistep.src.ratio_intervals import compute_all_ratio_intervals
from multistep.src.sampling import sample_points_from_constraint_system
from multistep.src.value_function import (
    MultistepConfig,
    QueryBranchResult,
    QueryEvaluation,
    ValueFunctionResult,
)
from multistep.src.weight_space import build_weight_space


DEFAULT_CASE_PATH = REPO_ROOT / "onestep" / "data" / "a5_a10_case.json"


@dataclass(frozen=True)
class CaseData:
    case_id: str
    goal_labels: list[str]
    entries: list[list[float]]
    answered_queries: list[AnsweredQuery]


@dataclass
class RuntimeStats:
    seconds_by_step: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    calls_by_step: Counter[str] = field(default_factory=Counter)
    seconds_by_depth_and_step: dict[tuple[int, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    calls_by_depth_and_step: Counter[tuple[int, str]] = field(
        default_factory=Counter
    )
    counters: Counter[str] = field(default_factory=Counter)
    counters_by_depth: Counter[tuple[int, str]] = field(default_factory=Counter)

    @contextmanager
    def timed(self, step: str, depth: int) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.seconds_by_step[step] += elapsed
            self.calls_by_step[step] += 1
            self.seconds_by_depth_and_step[(depth, step)] += elapsed
            self.calls_by_depth_and_step[(depth, step)] += 1

    def increment(self, counter: str, depth: int, amount: int = 1) -> None:
        self.counters[counter] += amount
        self.counters_by_depth[(depth, counter)] += amount


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the depth-2 multistep evaluation on the prepared "
            "a5/a10 case and print a step-level runtime breakdown."
        )
    )
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thinning", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--min-query-value", type=float, default=1e-3)
    parser.add_argument("--max-query-value", type=float, default=100.0)
    parser.add_argument(
        "--grid-spacing",
        choices=["linear", "log"],
        default="log",
    )
    parser.add_argument("--query-epsilon", type=float, default=QUERY_EPSILON)
    parser.add_argument("--top-steps", type=int, default=20)
    return parser.parse_args()


def load_case(path: Path) -> CaseData:
    with path.open("r", encoding="utf-8") as file:
        raw_case: dict[str, Any] = json.load(file)

    answered_queries = [
        AnsweredQuery(
            ziel_index_a=int(raw_query["ziel_index_a"]),
            ziel_index_b=int(raw_query["ziel_index_b"]),
            value=float(raw_query["value"]),
            operator=raw_query["operator"],
        )
        for raw_query in raw_case.get("tradeoffs", [])
    ]

    return CaseData(
        case_id=str(raw_case.get("id", path.stem)),
        goal_labels=[str(label) for label in raw_case["goal_labels"]],
        entries=[
            [float(value) for value in alternative]
            for alternative in raw_case["handlungsalternativenmatrix"]
        ],
        answered_queries=answered_queries,
    )


def format_query(query: Query | None, goal_labels: list[str]) -> str:
    if query is None:
        return "-"

    goal_a = goal_labels[int(query.ziel_index_a)]
    goal_b = goal_labels[int(query.ziel_index_b)]
    return f"{goal_a} ? {float(query.value):.8g} * {goal_b}"


def timed_compute_value_function(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    remaining_depth: int,
    config: MultistepConfig,
    stats: RuntimeStats,
) -> ValueFunctionResult:
    if remaining_depth < 0:
        raise ValueError("remaining_depth must not be negative")

    stats.increment("state_calls", remaining_depth)

    with stats.timed("state.build_weight_space", remaining_depth):
        weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=answered_queries,
        )

    with stats.timed("state.feasibility_check", remaining_depth):
        is_feasible = weight_space.is_feasible()

    if not is_feasible:
        stats.increment("infeasible_states", remaining_depth)
        return ValueFunctionResult(
            remaining_depth=remaining_depth,
            value=0.0,
            best_query=None,
            candidate_count=0,
            query_evaluations=(),
            is_feasible=False,
        )

    with stats.timed("state.compute_candidate_set", remaining_depth):
        candidates = compute_candidate_set(
            alternatives=alternatives,
            weight_space=weight_space,
        )
    candidate_count = len(candidates)
    stats.increment("candidate_count_sum", remaining_depth, candidate_count)

    if remaining_depth == 0 or candidate_count <= 1:
        stats.increment("terminal_states", remaining_depth)
        return ValueFunctionResult(
            remaining_depth=remaining_depth,
            value=float(candidate_count),
            best_query=None,
            candidate_count=candidate_count,
            query_evaluations=(),
            is_feasible=True,
        )

    query_candidates = timed_compute_query_candidates_for_depth(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
        remaining_depth=remaining_depth,
        config=config,
        stats=stats,
    )
    stats.increment("query_candidates", remaining_depth, len(query_candidates))
    if not query_candidates:
        stats.increment("states_without_query_candidates", remaining_depth)
        return ValueFunctionResult(
            remaining_depth=remaining_depth,
            value=float(candidate_count),
            best_query=None,
            candidate_count=candidate_count,
            query_evaluations=(),
            is_feasible=True,
        )

    with stats.timed("state.sampling", remaining_depth):
        samples = sample_points_from_constraint_system(
            system=weight_space,
            num_samples=config.sample_count,
            burn_in=config.burn_in,
            thinning=config.thinning,
            seed=config.random_seed,
        )

    query_evaluations = tuple(
        timed_evaluate_query_candidate(
            alternatives=alternatives,
            answered_queries=answered_queries,
            query=query,
            samples=samples,
            remaining_depth=remaining_depth,
            config=config,
            stats=stats,
        )
        for query in query_candidates
    )

    with stats.timed("state.best_query_selection", remaining_depth):
        best_evaluation = min(
            query_evaluations,
            key=lambda evaluation: evaluation.expected_value,
        )

    return ValueFunctionResult(
        remaining_depth=remaining_depth,
        value=best_evaluation.expected_value,
        best_query=best_evaluation.query,
        candidate_count=candidate_count,
        query_evaluations=query_evaluations,
        is_feasible=True,
    )


def timed_compute_query_candidates_for_depth(
    alternatives: AlternativenMatrix,
    weight_space: Any,
    candidates: list[int],
    remaining_depth: int,
    config: MultistepConfig,
    stats: RuntimeStats,
) -> list[Query]:
    if remaining_depth <= 0:
        return []

    if remaining_depth == 1:
        with stats.timed("query_candidates.ratio_intervals", remaining_depth):
            ratio_intervals = compute_all_ratio_intervals(
                alternatives=alternatives,
                weight_space=weight_space,
                candidates=candidates,
            )
        with stats.timed("query_candidates.onestep_from_intervals", remaining_depth):
            return compute_onestep_query_candidates(
                goal_pair_ratio_intervals=ratio_intervals,
                epsilon=config.query_epsilon,
            )

    with stats.timed("query_candidates.grid", remaining_depth):
        return compute_grid_query_candidates(
            weight_space=weight_space,
            grid_size=config.grid_size,
            min_query_value=config.min_query_value,
            max_query_value=config.max_query_value,
            spacing=config.grid_spacing,
        )


def timed_evaluate_query_candidate(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    query: Query,
    samples: list[list[float]],
    remaining_depth: int,
    config: MultistepConfig,
    stats: RuntimeStats,
) -> QueryEvaluation:
    if remaining_depth <= 0:
        raise ValueError("remaining_depth must be positive")

    stats.increment("query_evaluations", remaining_depth)

    with stats.timed("query.probabilities", remaining_depth):
        probabilities = estimate_query_answer_probabilities(
            query=query,
            samples=samples,
            equality_tol=config.equality_tol,
        )

    branches: list[QueryBranchResult] = []
    expected_value = 0.0

    for answer in ANSWER_OPTIONS:
        stats.increment("branch_checks", remaining_depth)
        probability = probabilities[answer]
        child_answered_queries = answered_queries + [query.answer(answer)]

        with stats.timed("branch.build_weight_space", remaining_depth):
            child_weight_space = build_weight_space(
                goal_count=alternatives.get_anzahl_spalten(),
                answered_queries=child_answered_queries,
            )

        with stats.timed("branch.feasibility_check", remaining_depth):
            is_child_feasible = child_weight_space.is_feasible()

        if is_child_feasible:
            child_result = timed_compute_value_function(
                alternatives=alternatives,
                answered_queries=child_answered_queries,
                remaining_depth=remaining_depth - 1,
                config=config,
                stats=stats,
            )
            child_value = child_result.value
            child_candidate_count: int | None = child_result.candidate_count
        else:
            stats.increment("infeasible_branches", remaining_depth)
            child_value = 0.0
            child_candidate_count = 0

        expected_value += probability * child_value
        branches.append(
            QueryBranchResult(
                answer=answer,
                probability=probability,
                child_value=child_value,
                child_candidate_count=child_candidate_count,
                is_child_feasible=is_child_feasible,
            )
        )

    return QueryEvaluation(
        query=query,
        expected_value=expected_value,
        branches=tuple(branches),
    )


def print_counter_by_depth(
    stats: RuntimeStats,
    counter_name: str,
    depths: list[int],
) -> None:
    values = [
        stats.counters_by_depth[(depth, counter_name)]
        for depth in depths
    ]
    print(f"  {counter_name:<32} " + " ".join(f"{value:>8}" for value in values))


def print_step_summary(
    stats: RuntimeStats,
    wall_time: float,
    top_steps: int,
) -> None:
    rows = sorted(
        stats.seconds_by_step.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_steps]

    print()
    print("Runtime by step")
    print("  step                                      seconds    share    calls   ms/call")
    for step, seconds in rows:
        calls = stats.calls_by_step[step]
        ms_per_call = 1000.0 * seconds / calls if calls else 0.0
        share = seconds / wall_time if wall_time else 0.0
        print(
            f"  {step:<38} "
            f"{seconds:>8.3f} "
            f"{share:>8.2%} "
            f"{calls:>8} "
            f"{ms_per_call:>9.2f}"
        )


def print_step_by_depth_summary(stats: RuntimeStats, wall_time: float) -> None:
    rows = sorted(
        stats.seconds_by_depth_and_step.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("Runtime by depth and step")
    print("  depth step                                      seconds    share    calls   ms/call")
    for (depth, step), seconds in rows:
        calls = stats.calls_by_depth_and_step[(depth, step)]
        ms_per_call = 1000.0 * seconds / calls if calls else 0.0
        share = seconds / wall_time if wall_time else 0.0
        print(
            f"  {depth:>5} {step:<38} "
            f"{seconds:>8.3f} "
            f"{share:>8.2%} "
            f"{calls:>8} "
            f"{ms_per_call:>9.2f}"
        )


def main() -> None:
    args = parse_args()
    case_data = load_case(args.case)
    alternatives = AlternativenMatrix(entries=case_data.entries)
    config = MultistepConfig(
        sample_count=args.samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        random_seed=args.seed,
        grid_size=args.grid_size,
        min_query_value=args.min_query_value,
        max_query_value=args.max_query_value,
        grid_spacing=args.grid_spacing,
        query_epsilon=args.query_epsilon,
    )
    stats = RuntimeStats()

    start = time.perf_counter()
    result = timed_compute_value_function(
        alternatives=alternatives,
        answered_queries=case_data.answered_queries,
        remaining_depth=2,
        config=config,
        stats=stats,
    )
    wall_time = time.perf_counter() - start

    depths = [2, 1, 0]

    print("Configuration")
    print(f"  case: {args.case}")
    print(f"  goals: {alternatives.get_anzahl_spalten()}")
    print(f"  alternatives: {alternatives.get_anzahl_zeilen()}")
    print(f"  answered queries: {len(case_data.answered_queries)}")
    print(f"  depth: 2")
    print(f"  samples: {args.samples}")
    print(f"  burn-in: {args.burn_in}")
    print(f"  thinning: {args.thinning}")
    print(f"  seed: {args.seed}")
    print(f"  grid-size: {args.grid_size}")
    print(f"  grid-spacing: {args.grid_spacing}")
    print(f"  min query value: {args.min_query_value:g}")
    print(f"  max query value: {args.max_query_value:g}")
    print()
    print("Result")
    print(f"  total runtime: {wall_time:.3f} s")
    print(f"  candidate count at root: {result.candidate_count}")
    print(f"  value: {result.value:.6g}")
    print(f"  best query: {format_query(result.best_query, case_data.goal_labels)}")
    print(f"  root query evaluations: {len(result.query_evaluations)}")
    print()
    print("Counters by remaining depth")
    print("  counter                              depth=2  depth=1  depth=0")
    for counter_name in [
        "state_calls",
        "terminal_states",
        "query_candidates",
        "query_evaluations",
        "branch_checks",
        "infeasible_branches",
        "candidate_count_sum",
    ]:
        print_counter_by_depth(
            stats=stats,
            counter_name=counter_name,
            depths=depths,
        )

    print_step_summary(
        stats=stats,
        wall_time=wall_time,
        top_steps=args.top_steps,
    )
    print_step_by_depth_summary(
        stats=stats,
        wall_time=wall_time,
    )


if __name__ == "__main__":
    main()
