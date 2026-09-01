from __future__ import annotations

"""Benchmark depth-three planning with a depth-two safety band."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np


MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.optimized import (  # noqa: E402
    OptimizedMultistepConfig,
    OptimizedValueFunctionSession,
)
from multistep.optimized.value_function import (  # noqa: E402
    canonical_query_key,
    query_evaluation_sort_key,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query  # noqa: E402
from multistep.src.polytope_volume import (  # noqa: E402
    clear_polytope_volume_cache,
    polytope_volume_cache_info,
)
from multistep.src.query_probability import classify_query_answer  # noqa: E402
from multistep.src.value_function import QueryEvaluation  # noqa: E402


ExecutionMode = Literal["receding", "countdown"]


@dataclass(frozen=True)
class GuardedQuerySelection:
    query: Query
    expected_candidates_depth_three: float
    expected_candidates_depth_two: float
    best_expected_candidates_depth_two: float
    admissible_query_count: int
    total_query_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mode", choices=["receding", "countdown"], required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--start-problem", type=int, default=1)
    parser.add_argument("--problems", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=100)
    return parser.parse_args()


def build_config() -> OptimizedMultistepConfig:
    return OptimizedMultistepConfig(
        answer_probability_mode="exact_volume",
        skip_zero_probability_branches=True,
        pass_candidate_subset=True,
        use_ratio_terminal_counts=True,
        validate_ratio_terminal_counts=False,
        candidate_count_mode="ratio_relevant",
        ratio_interval_engine="geometry",
        grid_depth_query_source_mode="central",
        depth_one_query_source_mode="central",
        parallelize_root=False,
    )


def load_problems(paths: list[Path]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            problems.append(
                {
                    "source": str(path),
                    "source_index": source_index,
                    "source_problem_index": int(result["problem_index"]),
                    "goal_count": int(result["goal_count"]),
                    "alternatives": result["alternatives"],
                    "target_weights": result["target_weights"],
                }
            )
    return problems


def select_depth_three_query_with_depth_two_guard(
    session: OptimizedValueFunctionSession,
    answered_queries: list[AnsweredQuery],
    delta: float,
    numerical_tolerance: float = 1e-12,
) -> GuardedQuerySelection:
    """Minimize E3 among queries whose E2 is at most E2* + delta."""

    if delta < 0.0:
        raise ValueError("delta must not be negative")
    if numerical_tolerance < 0.0:
        raise ValueError("numerical_tolerance must not be negative")

    depth_three_result = session.compute(
        answered_queries=answered_queries,
        remaining_depth=3,
    )
    depth_two_result = session.compute(
        answered_queries=answered_queries,
        remaining_depth=2,
    )
    if not depth_three_result.query_evaluations:
        raise RuntimeError("depth-three planner returned no query evaluations")
    if not depth_two_result.query_evaluations:
        raise RuntimeError("depth-two planner returned no query evaluations")

    depth_two_by_query = {
        canonical_query_key(evaluation.query): evaluation
        for evaluation in depth_two_result.query_evaluations
    }
    best_depth_two_value = min(
        evaluation.expected_value
        for evaluation in depth_two_result.query_evaluations
    )
    admissible: list[tuple[QueryEvaluation, QueryEvaluation]] = []
    for depth_three_evaluation in depth_three_result.query_evaluations:
        key = canonical_query_key(depth_three_evaluation.query)
        depth_two_evaluation = depth_two_by_query.get(key)
        if depth_two_evaluation is None:
            raise RuntimeError(
                "depth-two and depth-three query candidate sets do not match"
            )
        if (
            depth_two_evaluation.expected_value
            <= best_depth_two_value + delta + numerical_tolerance
        ):
            admissible.append((depth_three_evaluation, depth_two_evaluation))

    if not admissible:
        raise RuntimeError("depth-two safety band excluded every query")
    best_depth_three, corresponding_depth_two = min(
        admissible,
        key=lambda evaluations: query_evaluation_sort_key(evaluations[0]),
    )
    return GuardedQuerySelection(
        query=best_depth_three.query,
        expected_candidates_depth_three=float(best_depth_three.expected_value),
        expected_candidates_depth_two=float(corresponding_depth_two.expected_value),
        best_expected_candidates_depth_two=float(best_depth_two_value),
        admissible_query_count=len(admissible),
        total_query_count=len(depth_three_result.query_evaluations),
    )


def target_winner(
    alternatives: AlternativenMatrix,
    target_weights: list[float],
) -> int:
    return int(
        np.argmax(
            np.asarray(target_weights, dtype=float)
            @ np.asarray(alternatives.entries, dtype=float).T
        )
    )


def query_to_json(
    query: Query,
    answer: str,
    seconds: float,
    planning_depth: int,
    guarded_selection: GuardedQuerySelection | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "goal_index_a": int(query.ziel_index_a),
        "goal_index_b": int(query.ziel_index_b),
        "value": float(query.value),
        "answer": answer,
        "seconds": seconds,
        "planning_depth": planning_depth,
        "used_depth_two_guard": guarded_selection is not None,
    }
    if guarded_selection is not None:
        result.update(
            {
                "expected_candidates_depth_three": (
                    guarded_selection.expected_candidates_depth_three
                ),
                "expected_candidates_depth_two": (
                    guarded_selection.expected_candidates_depth_two
                ),
                "best_expected_candidates_depth_two": (
                    guarded_selection.best_expected_candidates_depth_two
                ),
                "depth_two_sacrifice": (
                    guarded_selection.expected_candidates_depth_two
                    - guarded_selection.best_expected_candidates_depth_two
                ),
                "admissible_query_count": guarded_selection.admissible_query_count,
                "total_query_count": guarded_selection.total_query_count,
            }
        )
    return result


def solve_problem(
    problem: dict[str, Any],
    mode: ExecutionMode,
    delta: float,
    max_questions: int,
) -> dict[str, Any]:
    clear_polytope_volume_cache()
    alternatives = AlternativenMatrix(entries=problem["alternatives"])
    target_weights = [float(value) for value in problem["target_weights"]]
    true_winner = target_winner(alternatives, target_weights)
    answered_queries: list[AnsweredQuery] = []
    query_records: list[dict[str, Any]] = []
    initial_candidate_count: int | None = None
    final_candidates: list[int] = []
    started = time.perf_counter()

    with OptimizedValueFunctionSession(
        alternatives=alternatives,
        config=build_config(),
        max_cached_results=3 * max_questions + 3,
    ) as session:
        for question_index in range(max_questions + 1):
            state = session.analyze_state(answered_queries)
            if not state.is_feasible or state.candidate_analysis is None:
                raise RuntimeError("query trajectory became infeasible")
            candidates = state.candidate_analysis.candidates
            if initial_candidate_count is None:
                initial_candidate_count = len(candidates)
            if true_winner not in candidates:
                raise RuntimeError("true winner was removed from the candidate set")
            final_candidates = candidates
            if len(candidates) <= 1:
                break
            if question_index >= max_questions:
                break

            iteration_started = time.perf_counter()
            countdown_phase = question_index % 3
            use_guard = mode == "receding" or countdown_phase == 0
            guarded_selection: GuardedQuerySelection | None = None
            if use_guard:
                planning_depth = 3
                guarded_selection = select_depth_three_query_with_depth_two_guard(
                    session=session,
                    answered_queries=answered_queries,
                    delta=delta,
                )
                best_query = guarded_selection.query
            else:
                planning_depth = 3 - countdown_phase
                planning_result = session.compute(
                    answered_queries=answered_queries,
                    remaining_depth=planning_depth,
                )
                best_query = planning_result.best_query
                if best_query is None:
                    raise RuntimeError(
                        f"depth-{planning_depth} planner returned no query"
                    )

            answer = classify_query_answer(
                weights=target_weights,
                query=best_query,
                equality_tol=0.0,
            )
            if answer == "=":
                raise RuntimeError("random target produced an exact equality answer")
            query_records.append(
                query_to_json(
                    query=best_query,
                    answer=answer,
                    seconds=time.perf_counter() - iteration_started,
                    planning_depth=planning_depth,
                    guarded_selection=guarded_selection,
                )
            )
            answered_queries.append(best_query.answer(answer))

    seconds = time.perf_counter() - started
    cache = polytope_volume_cache_info()
    return {
        "solved": len(final_candidates) == 1,
        "question_count": len(answered_queries),
        "seconds": seconds,
        "initial_candidate_count": initial_candidate_count,
        "final_candidates": final_candidates,
        "target_winner": true_winner,
        "final_candidate": final_candidates[0] if len(final_candidates) == 1 else None,
        "query_seconds": [record["seconds"] for record in query_records],
        "queries": query_records,
        "guarded_query_count": sum(
            bool(record["used_depth_two_guard"]) for record in query_records
        ),
        "changed_from_depth_two_count": sum(
            bool(record.get("depth_two_sacrifice", 0.0) > 1e-12)
            for record in query_records
        ),
        "volume_cache": {
            "hits": cache.hits,
            "misses": cache.misses,
            "size": cache.currsize,
        },
    }


def write_checkpoint(
    path: Path,
    settings: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"settings": settings, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.delta < 0.0:
        raise ValueError("--delta must not be negative")
    if args.start_problem <= 0:
        raise ValueError("--start-problem must be positive")
    if args.problems is not None and args.problems <= 0:
        raise ValueError("--problems must be positive")
    if args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")

    all_problems = load_problems(args.input_json)
    start = args.start_problem - 1
    stop = None if args.problems is None else start + args.problems
    selected_problems = all_problems[start:stop]
    if not selected_problems:
        raise ValueError("selected problem range is empty")

    settings = {
        "input_json": [str(path) for path in args.input_json],
        "start_problem": args.start_problem,
        "problems": len(selected_problems),
        "mode": args.mode,
        "delta": args.delta,
        "max_questions": args.max_questions,
        "answer_probability_mode": "exact_volume",
        "query_candidates": "one vertex-centroid ratio per canonical goal pair",
        "objective": "minimum E3 subject to E2 <= E2* + delta",
    }
    results: list[dict[str, Any]] = []
    for benchmark_index, problem in enumerate(selected_problems, start=1):
        result = solve_problem(
            problem=problem,
            mode=args.mode,
            delta=float(args.delta),
            max_questions=int(args.max_questions),
        )
        result.update(problem)
        result["benchmark_index"] = benchmark_index
        results.append(result)
        write_checkpoint(args.output_json, settings, results)
        print(
            f"problem={benchmark_index:03d} mode={args.mode} delta={args.delta:g} "
            f"solved={result['solved']} questions={result['question_count']:02d} "
            f"seconds={result['seconds']:.3f}",
            flush=True,
        )

    print(
        f"wrote {args.output_json}; solved "
        f"{sum(bool(result['solved']) for result in results)}/{len(results)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
