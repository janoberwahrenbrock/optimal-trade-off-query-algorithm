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
    evaluate_volume_confidence_termination,
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
    parser.add_argument("--start-problem", type=int, default=1)
    parser.add_argument("--alternatives", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--depth", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--parallel-root",
        action="store_true",
        help="Evaluate root query candidates in a persistent process pool.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument(
        "--root-query-source",
        choices=["grid", "ratio", "both", "central"],
        default="central",
    )
    parser.add_argument(
        "--depth-one-query-source",
        choices=["grid", "ratio", "both", "central"],
        default="central",
    )
    parser.add_argument(
        "--progress-per-query",
        action="store_true",
        help="Print the runtime and remaining candidate count after every answer.",
    )
    parser.add_argument(
        "--volume-confidence-threshold",
        type=float,
        default=0.99,
        help="Stop when one candidate owns at least this volume share.",
    )
    parser.add_argument(
        "--require-exact-termination",
        action="store_true",
        help="Disable volume-confidence stopping and require one exact candidate.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.goals or any(goal_count < 3 for goal_count in args.goals):
        raise ValueError("goal counts must be at least three")
    if args.problems <= 0 or args.alternatives <= 1:
        raise ValueError("invalid problem or alternative count")
    if args.start_problem <= 0:
        raise ValueError("start-problem must be positive")
    if args.max_questions <= 0:
        raise ValueError("max-questions must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.grid_size <= 0:
        raise ValueError("grid-size must be positive")
    if not 0.0 < args.volume_confidence_threshold <= 1.0:
        raise ValueError("volume-confidence-threshold must be in (0, 1]")


def build_config(
    parallelize_root: bool = False,
    max_workers: int = 4,
    grid_size: int = 21,
    root_query_source: str = "central",
    depth_one_query_source: str = "central",
    volume_confidence_threshold: float | None = 0.99,
) -> OptimizedMultistepConfig:
    return OptimizedMultistepConfig(
        answer_probability_mode="exact_volume",
        skip_zero_probability_branches=True,
        pass_candidate_subset=True,
        use_ratio_terminal_counts=True,
        validate_ratio_terminal_counts=False,
        candidate_count_mode="ratio_relevant",
        ratio_interval_engine="geometry",
        grid_size=grid_size,
        grid_depth_query_source_mode=root_query_source,
        depth_one_query_source_mode=depth_one_query_source,
        parallelize_root=parallelize_root,
        max_workers=max_workers,
        volume_confidence_threshold=volume_confidence_threshold,
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


def format_candidate_volume_shares(
    candidates: list[int],
    volume_shares: dict[int, float] | None,
) -> str:
    if volume_shares is None:
        return "unavailable"
    return ",".join(
        f"{candidate}:{100.0 * volume_shares[candidate]:.4f}%"
        for candidate in candidates
    )


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
    parallelize_root: bool = False,
    max_workers: int = 4,
    grid_size: int = 21,
    root_query_source: str = "central",
    depth_one_query_source: str = "central",
    progress_per_query: bool = False,
    volume_confidence_threshold: float | None = 0.99,
) -> dict[str, Any]:
    clear_polytope_volume_cache()
    config = build_config(
        parallelize_root=parallelize_root,
        max_workers=max_workers,
        grid_size=grid_size,
        root_query_source=root_query_source,
        depth_one_query_source=depth_one_query_source,
        volume_confidence_threshold=volume_confidence_threshold,
    )
    answered_queries: list[AnsweredQuery] = []
    query_records: list[dict[str, Any]] = []
    true_winner = target_winner(alternatives, target_weights)
    started = time.perf_counter()
    initial_candidate_count: int | None = None
    initial_candidates: list[int] = []
    initial_candidate_volumes: dict[int, float] | None = None
    initial_candidate_volume_shares: dict[int, float] | None = None
    final_candidates: list[int] = []
    final_candidate_volumes: dict[int, float] | None = None
    final_candidate_volume_shares: dict[int, float] | None = None
    termination_reason = "max_questions"
    selected_candidate: int | None = None
    selected_candidate_volume: float | None = None
    selected_candidate_volume_share: float | None = None
    residual_volume: float | None = None
    residual_volume_share: float | None = None
    with OptimizedValueFunctionSession(
        alternatives=alternatives,
        config=config,
        max_cached_results=max_questions + 1,
    ) as session:
        for question_index in range(max_questions + 1):
            iteration_started = time.perf_counter()
            state = session.analyze_state(
                answered_queries,
                include_candidate_volumes=True,
            )
            if not state.is_feasible or state.candidate_analysis is None:
                raise RuntimeError("query trajectory became infeasible")
            candidates = state.candidate_analysis.candidates
            candidate_volumes = state.candidate_analysis.candidate_volumes
            candidate_volume_shares = (
                state.candidate_analysis.candidate_volume_shares
            )
            if initial_candidate_count is None:
                initial_candidate_count = len(candidates)
                initial_candidates = [int(candidate) for candidate in candidates]
                initial_candidate_volumes = candidate_volumes
                initial_candidate_volume_shares = candidate_volume_shares
                if progress_per_query:
                    print(
                        f"query=00 candidates={initial_candidates} "
                        f"volume_shares={format_candidate_volume_shares(candidates, candidate_volume_shares)} "
                        "initial=True",
                        flush=True,
                    )
            if query_records and "remaining_candidate_count" not in query_records[-1]:
                query_record = query_records[-1]
                query_record["remaining_candidate_count"] = len(candidates)
                query_record["remaining_candidates"] = [
                    int(candidate) for candidate in candidates
                ]
                query_record["candidate_volumes"] = candidate_volumes
                query_record["candidate_volume_shares"] = candidate_volume_shares
                if progress_per_query:
                    print(
                        f"query={len(query_records):02d} "
                        f"candidates={query_record['remaining_candidates']} "
                        f"volume_shares={format_candidate_volume_shares(candidates, candidate_volume_shares)} "
                        f"expected_value={query_record['expected_value']:.6f} "
                        f"seconds={query_record['seconds']:.3f}",
                        flush=True,
                    )
            if true_winner not in candidates:
                raise RuntimeError("true winner was removed from the candidate set")
            final_candidates = candidates
            final_candidate_volumes = candidate_volumes
            final_candidate_volume_shares = candidate_volume_shares
            if len(candidates) <= 1:
                termination_reason = "exact_single_candidate"
                if candidates:
                    selected_candidate = int(candidates[0])
                    if candidate_volumes is not None:
                        selected_candidate_volume = float(
                            candidate_volumes[selected_candidate]
                        )
                    if candidate_volume_shares is not None:
                        selected_candidate_volume_share = float(
                            candidate_volume_shares[selected_candidate]
                        )
                        residual_volume_share = max(
                            0.0,
                            1.0 - selected_candidate_volume_share,
                        )
                    if candidate_volumes is not None:
                        residual_volume = max(
                            0.0,
                            sum(candidate_volumes.values())
                            - float(candidate_volumes[selected_candidate]),
                        )
                break

            confidence_decision = evaluate_volume_confidence_termination(
                candidate_analysis=state.candidate_analysis,
                threshold=config.volume_confidence_threshold,
            )
            if confidence_decision is not None:
                termination_reason = "volume_confidence"
                selected_candidate = confidence_decision.selected_candidate
                selected_candidate_volume = (
                    confidence_decision.selected_candidate_volume
                )
                selected_candidate_volume_share = (
                    confidence_decision.selected_candidate_volume_share
                )
                residual_volume = confidence_decision.residual_volume
                residual_volume_share = confidence_decision.residual_volume_share
                if progress_per_query:
                    print(
                        f"termination={termination_reason} "
                        f"selected_candidate={selected_candidate} "
                        f"confidence={selected_candidate_volume_share:.6f} "
                        f"residual_volume_share={residual_volume_share:.6f}",
                        flush=True,
                    )
                break
            if question_index >= max_questions:
                break

            result = session.compute(
                answered_queries=answered_queries,
                remaining_depth=depth,
            )
            if result.best_query is None:
                raise RuntimeError("exact planner returned no query")
            best_query = result.best_query
            best_evaluation = next(
                evaluation
                for evaluation in result.query_evaluations
                if evaluation.query == best_query
            )
            expected_values = (
                best_evaluation.lexicographic_expected_values
                or (float(best_evaluation.expected_value),)
            )
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
            query_records[-1]["expected_value"] = float(result.value)
            query_records[-1]["lexicographic_expected_values"] = [
                float(value) for value in expected_values
            ]

    seconds = time.perf_counter() - started
    cache = polytope_volume_cache_info()
    return {
        "solved": selected_candidate is not None,
        "exactly_solved": len(final_candidates) == 1,
        "termination_reason": termination_reason,
        "question_count": len(answered_queries),
        "seconds": seconds,
        "initial_candidate_count": initial_candidate_count,
        "initial_candidates": initial_candidates,
        "initial_candidate_volumes": initial_candidate_volumes,
        "initial_candidate_volume_shares": initial_candidate_volume_shares,
        "final_candidates": final_candidates,
        "final_candidate_volumes": final_candidate_volumes,
        "final_candidate_volume_shares": final_candidate_volume_shares,
        "target_winner": true_winner,
        "final_candidate": final_candidates[0] if len(final_candidates) == 1 else None,
        "selected_candidate": selected_candidate,
        "selected_candidate_volume": selected_candidate_volume,
        "selected_candidate_volume_share": selected_candidate_volume_share,
        "residual_volume": residual_volume,
        "residual_volume_share": residual_volume_share,
        "selection_is_correct": (
            selected_candidate == true_winner
            if selected_candidate is not None
            else None
        ),
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
            "query_candidates": (
                "one vertex-centroid ratio per canonical goal pair"
                if args.root_query_source == "central"
                else (
                    f"root={args.root_query_source}, "
                    f"depth_one={args.depth_one_query_source}, "
                    f"grid_size={args.grid_size}"
                )
            ),
            "objective": (
                "lexicographic minimum of expected remaining candidates "
                "(E_d, ..., E_1)"
            ),
        }
    )
    rng = np.random.default_rng(int(args.seed))
    results: list[dict[str, Any]] = []
    for goal_count in args.goals:
        for _ in range(1, int(args.start_problem)):
            generate_problem(
                rng=rng,
                goal_count=int(goal_count),
                alternative_count=int(args.alternatives),
            )
        last_problem_index = int(args.start_problem) + int(args.problems) - 1
        for problem_index in range(int(args.start_problem), last_problem_index + 1):
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
                parallelize_root=bool(args.parallel_root),
                max_workers=int(args.workers),
                grid_size=int(args.grid_size),
                root_query_source=str(args.root_query_source),
                depth_one_query_source=str(args.depth_one_query_source),
                progress_per_query=bool(args.progress_per_query),
                volume_confidence_threshold=(
                    None
                    if args.require_exact_termination
                    else float(args.volume_confidence_threshold)
                ),
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
                f"termination={result['termination_reason']} "
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
