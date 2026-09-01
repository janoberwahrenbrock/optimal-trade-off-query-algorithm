from __future__ import annotations

"""Benchmark stopping-time rollout on previously generated exact problems."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.optimized import (  # noqa: E402
    OptimizedMultistepConfig,
    StoppingTimeRolloutSession,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query  # noqa: E402
from multistep.src.polytope_volume import (  # noqa: E402
    clear_polytope_volume_cache,
    polytope_volume_cache_info,
)
from multistep.src.query_probability import classify_query_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--start-problem", type=int, default=1)
    parser.add_argument("--problems", type=int, default=None)
    parser.add_argument("--rollout-depth", type=int, default=3)
    parser.add_argument("--baseline-depth", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--max-baseline-questions", type=int, default=100)
    parser.add_argument(
        "--baseline-path-probability-cutoff",
        type=float,
        default=1e-8,
    )
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


def query_to_json(query: Query, answer: str, seconds: float) -> dict[str, Any]:
    return {
        "goal_index_a": int(query.ziel_index_a),
        "goal_index_b": int(query.ziel_index_b),
        "value": float(query.value),
        "answer": answer,
        "seconds": seconds,
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


def solve_problem(
    problem: dict[str, Any],
    rollout_depth: int,
    baseline_depth: int,
    max_questions: int,
    max_baseline_questions: int,
    baseline_path_probability_cutoff: float,
) -> dict[str, Any]:
    clear_polytope_volume_cache()
    alternatives = AlternativenMatrix(entries=problem["alternatives"])
    target_weights = [float(value) for value in problem["target_weights"]]
    true_winner = target_winner(alternatives, target_weights)
    answered_queries: list[AnsweredQuery] = []
    query_records: list[dict[str, Any]] = []
    rollout_values: list[float] = []
    baseline_values: list[float] = []
    differs_from_baseline: list[bool] = []
    initial_candidate_count: int | None = None
    final_candidates: list[int] = []
    started = time.perf_counter()

    with StoppingTimeRolloutSession(
        alternatives=alternatives,
        config=build_config(),
        rollout_depth=rollout_depth,
        baseline_depth=baseline_depth,
        max_baseline_questions=max_baseline_questions,
        baseline_path_probability_cutoff=baseline_path_probability_cutoff,
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
            rollout_result = session.compute(answered_queries)
            best_query = rollout_result.best_query
            if best_query is None:
                raise RuntimeError("stopping-time rollout returned no query")
            answer = classify_query_answer(
                weights=target_weights,
                query=best_query,
                equality_tol=0.0,
            )
            if answer == "=":
                raise RuntimeError("random target produced an exact equality answer")
            seconds = time.perf_counter() - iteration_started
            query_records.append(query_to_json(best_query, answer, seconds))
            rollout_values.append(float(rollout_result.expected_questions))
            baseline_values.append(
                float(rollout_result.baseline_expected_questions)
            )
            differs_from_baseline.append(best_query != rollout_result.baseline_query)
            answered_queries.append(best_query.answer(answer))

        statistics = session.statistics

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
        "rollout_expected_questions": rollout_values,
        "baseline_expected_questions": baseline_values,
        "differs_from_baseline": differs_from_baseline,
        "rollout_statistics": {
            "baseline_states_evaluated": statistics.baseline_states_evaluated,
            "baseline_cache_hits": statistics.baseline_cache_hits,
            "rollout_states_evaluated": statistics.rollout_states_evaluated,
            "rollout_cache_hits": statistics.rollout_cache_hits,
            "maximum_pruned_baseline_probability": (
                statistics.maximum_pruned_baseline_probability
            ),
        },
        "volume_cache": {
            "hits": cache.hits,
            "misses": cache.misses,
            "size": cache.currsize,
        },
    }


def main() -> None:
    args = parse_args()
    if args.start_problem <= 0:
        raise ValueError("--start-problem must be positive")
    if args.problems is not None and args.problems <= 0:
        raise ValueError("--problems must be positive")
    if args.rollout_depth <= 0 or args.baseline_depth <= 0:
        raise ValueError("depths must be positive")
    if not 0.0 < args.baseline_path_probability_cutoff < 1.0:
        raise ValueError(
            "--baseline-path-probability-cutoff must be between zero and one"
        )

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
        "rollout_depth": args.rollout_depth,
        "baseline_depth": args.baseline_depth,
        "max_questions": args.max_questions,
        "max_baseline_questions": args.max_baseline_questions,
        "baseline_path_probability_cutoff": (
            args.baseline_path_probability_cutoff
        ),
        "answer_probability_mode": "exact_volume",
        "query_candidates": "one vertex-centroid ratio per canonical goal pair",
        "objective": (
            "expected questions to termination with depth-two baseline cost-to-go"
        ),
    }
    results: list[dict[str, Any]] = []
    for benchmark_index, problem in enumerate(selected_problems, start=1):
        result = solve_problem(
            problem=problem,
            rollout_depth=int(args.rollout_depth),
            baseline_depth=int(args.baseline_depth),
            max_questions=int(args.max_questions),
            max_baseline_questions=int(args.max_baseline_questions),
            baseline_path_probability_cutoff=float(
                args.baseline_path_probability_cutoff
            ),
        )
        result.update(problem)
        result["benchmark_index"] = benchmark_index
        results.append(result)
        write_checkpoint(args.output_json, settings, results)
        print(
            f"problem={benchmark_index:03d} solved={result['solved']} "
            f"questions={result['question_count']:02d} "
            f"seconds={result['seconds']:.3f} "
            f"baseline_states={result['rollout_statistics']['baseline_states_evaluated']}",
            flush=True,
        )

    print(
        f"wrote {args.output_json}; solved "
        f"{sum(bool(result['solved']) for result in results)}/{len(results)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
