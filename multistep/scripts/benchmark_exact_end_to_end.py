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
    parser.add_argument(
        "--query-strategy", choices=["fixed", "reduction_per_query", "reduction_per_query_stop_aware"], default="fixed",
        help="Use fixed lookahead, candidate reduction per expected query, or stop-aware loss reduction per expected query; adaptive horizons are at most three.",
    )
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
    if getattr(args, "query_strategy", "fixed") in {"reduction_per_query", "reduction_per_query_stop_aware"}:
        if args.depth not in (1, 2, 3):
            raise ValueError("reduction_per_query supports maximum horizons 1, 2 and 3")
        if args.root_query_source not in {"grid", "both"} or args.depth_one_query_source != "ratio":
            raise ValueError("reduction_per_query requires --root-query-source grid/both --depth-one-query-source ratio")


def build_config(
    parallelize_root: bool = False,
    max_workers: int = 4,
    grid_size: int = 21,
    root_query_source: str = "central",
    depth_one_query_source: str = "central",
    volume_confidence_threshold: float | None = 0.99,
    query_efficiency_stop_aware: bool = False,
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
        query_efficiency_stop_aware=query_efficiency_stop_aware,
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
    query_strategy: str = "fixed",
) -> dict[str, Any]:
    """Benchmark adapter: simulate answers around the shared exact solver."""
    from multistep.optimized.algorithm import _solve_with_optimized_config

    if query_strategy not in {"fixed", "reduction_per_query", "reduction_per_query_stop_aware"}:
        raise ValueError("unsupported query strategy")
    if query_strategy in {"reduction_per_query", "reduction_per_query_stop_aware"}:
        if depth not in (1, 2, 3):
            raise ValueError("reduction_per_query supports maximum horizons 1, 2 and 3")
        if root_query_source not in {"both", "grid"} or depth_one_query_source != "ratio":
            raise ValueError("reduction_per_query requires grid or grid plus ratios above depth one and ratio queries at depth one")
    config = build_config(
        parallelize_root=parallelize_root,
        max_workers=max_workers,
        grid_size=grid_size,
        root_query_source=root_query_source,
        depth_one_query_source=depth_one_query_source,
        volume_confidence_threshold=volume_confidence_threshold,
        query_efficiency_stop_aware=(query_strategy == "reduction_per_query_stop_aware"),
    )
    true_winner = target_winner(alternatives, target_weights)

    def answer_query(query: Query) -> str:
        answer = classify_query_answer(weights=target_weights, query=query, equality_tol=0.0)
        if answer == "=":
            raise RuntimeError("random target produced an exact equality answer")
        return answer

    def check_true_winner(state: Any, question_index: int) -> None:
        if true_winner not in state.candidate_analysis.candidates:
            raise RuntimeError("true winner was removed from the candidate set")

    result = _solve_with_optimized_config(
        alternatives=alternatives,
        optimized_config=config,
        answer_query=answer_query,
        depth=depth,
        query_strategy=query_strategy,
        max_queries=max_questions,
        session_factory=OptimizedValueFunctionSession,
        on_state=check_true_winner,
        progress_per_query=progress_per_query,
        raise_on_no_query=True,
    )
    result["target_winner"] = true_winner
    result["selection_is_correct"] = (
        result["selected_candidate"] == true_winner if result["selected_candidate"] is not None else None
    )
    return result


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
    if getattr(args, "query_strategy", "fixed") in {"reduction_per_query", "reduction_per_query_stop_aware"}:
        settings["objective"] = "maximum expected candidate reduction divided by expected executed queries; no hypothetical volume-confidence stop"
        settings["lookahead_depth"] = "adaptive horizons 1.." + str(args.depth)
        if args.query_strategy == "reduction_per_query_stop_aware":
            settings["objective"] = "maximum stop-aware terminal loss reduction divided by expected executed queries"
            settings["planning_stop_aware"] = True
            settings["planning_terminal_loss_semantics"] = (
                "one_for_single_candidate_or_volume_confidence_else_actual_candidate_count"
            )
    else:
        # Preserve the existing fixed-mode settings shape for ordinary callers.
        settings.pop("query_strategy", None)
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
                query_strategy=str(getattr(args, "query_strategy", "fixed")),
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
