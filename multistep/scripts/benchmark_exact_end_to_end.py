from __future__ import annotations

"""Solve deterministic random problems with exact volume planning."""

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
    OptimizedValueFunctionSession,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query  # noqa: E402
from multistep.src.polytope_volume import (  # noqa: E402
    clear_polytope_volume_cache,
    polytope_volume_cache_info,
)
from multistep.src.query_probability import classify_query_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--alternatives", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--depth", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.goals or any(goal_count < 3 for goal_count in args.goals):
        raise ValueError("goal counts must be at least three")
    if args.problems <= 0 or args.alternatives <= 1:
        raise ValueError("invalid problem or alternative count")
    if args.max_questions <= 0:
        raise ValueError("max-questions must be positive")


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


def generate_problem(
    rng: np.random.Generator,
    goal_count: int,
    alternative_count: int,
) -> tuple[AlternativenMatrix, list[float]]:
    entries = rng.uniform(0.0, 1.0, size=(alternative_count, goal_count))
    target_weights = rng.dirichlet(np.ones(goal_count, dtype=float))
    return AlternativenMatrix(entries=entries.tolist()), target_weights.tolist()


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
    alternatives: AlternativenMatrix,
    target_weights: list[float],
    max_questions: int,
    depth: int,
) -> dict[str, Any]:
    clear_polytope_volume_cache()
    config = build_config()
    answered_queries: list[AnsweredQuery] = []
    query_records: list[dict[str, Any]] = []
    true_winner = target_winner(alternatives, target_weights)
    started = time.perf_counter()
    initial_candidate_count: int | None = None
    final_candidates: list[int] = []
    with OptimizedValueFunctionSession(
        alternatives=alternatives,
        config=config,
        max_cached_results=max_questions + 1,
    ) as session:
        for question_index in range(max_questions + 1):
            iteration_started = time.perf_counter()
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

            result = session.compute(
                answered_queries=answered_queries,
                remaining_depth=depth,
            )
            if result.best_query is None:
                raise RuntimeError("exact planner returned no central query")
            best_query = result.best_query
            answer = classify_query_answer(
                weights=target_weights,
                query=best_query,
                equality_tol=0.0,
            )
            if answer == "=":
                raise RuntimeError("random target produced an exact equality answer")
            answered_queries.append(best_query.answer(answer))
            query_records.append(
                query_to_json(
                    best_query,
                    answer,
                    time.perf_counter() - iteration_started,
                )
            )

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
        "volume_cache": {
            "hits": cache.hits,
            "misses": cache.misses,
            "size": cache.currsize,
        },
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    settings.update(
        {
            "lookahead_depth": int(args.depth),
            "answer_probability_mode": "exact_volume",
            "query_candidates": "one vertex-centroid ratio per canonical goal pair",
            "objective": (
                "lexicographic minimum of expected remaining candidates "
                "(E_d, ..., E_1)"
            ),
        }
    )
    rng = np.random.default_rng(int(args.seed))
    results: list[dict[str, Any]] = []
    for goal_count in args.goals:
        for problem_index in range(1, int(args.problems) + 1):
            alternatives, weights = generate_problem(
                rng=rng,
                goal_count=int(goal_count),
                alternative_count=int(args.alternatives),
            )
            result = solve_problem(
                alternatives=alternatives,
                target_weights=weights,
                max_questions=int(args.max_questions),
                depth=int(args.depth),
            )
            result.update(
                {
                    "goal_count": int(goal_count),
                    "problem_index": problem_index,
                    "alternatives": alternatives.entries,
                    "target_weights": weights,
                }
            )
            results.append(result)
            write_checkpoint(args.output_json, settings, results)
            print(
                f"goals={goal_count} problem={problem_index:02d} "
                f"solved={result['solved']} questions={result['question_count']:02d} "
                f"seconds={result['seconds']:.3f} "
                f"last={max(result['query_seconds'], default=0.0):.3f}",
                flush=True,
            )
    return results


def main() -> None:
    args = parse_args()
    validate_args(args)
    results = run(args)
    solved = sum(bool(result["solved"]) for result in results)
    print(f"wrote {args.output_json}; solved {solved}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
