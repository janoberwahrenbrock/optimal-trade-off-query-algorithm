from __future__ import annotations

"""Run a resumable depth-two benchmark grid with durable per-problem checkpoints."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np


MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.scripts.benchmark_exact_end_to_end import (  # noqa: E402
    generate_problem,
    solve_problem,
)


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--alternatives", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--problems-per-combination", type=int, default=100)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--parallel-root",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry records whose previous status was error.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.goals or any(value < 2 for value in args.goals):
        raise ValueError("goals must contain values of at least two")
    if not args.alternatives or any(value < 2 for value in args.alternatives):
        raise ValueError("alternatives must contain values of at least two")
    if len(set(args.goals)) != len(args.goals):
        raise ValueError("goals must not contain duplicates")
    if len(set(args.alternatives)) != len(args.alternatives):
        raise ValueError("alternatives must not contain duplicates")
    if args.problems_per_combination <= 0:
        raise ValueError("problems-per-combination must be positive")
    if args.max_questions <= 0:
        raise ValueError("max-questions must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "goals": [int(value) for value in args.goals],
        "alternatives": [int(value) for value in args.alternatives],
        "problems_per_combination": int(args.problems_per_combination),
        "combination_count": len(args.goals) * len(args.alternatives),
        "planned_problem_count": (
            len(args.goals)
            * len(args.alternatives)
            * int(args.problems_per_combination)
        ),
        "max_questions": int(args.max_questions),
        "seed": int(args.seed),
        "lookahead_depth": 2,
        "answer_probability_mode": "exact_volume",
        "ratio_interval_engine": "geometry",
        "query_candidates": "one vertex-centroid ratio per canonical goal pair",
        "candidate_count_mode": "ratio_relevant",
        "parallel_root": bool(args.parallel_root),
        "workers": int(args.workers),
    }


def new_checkpoint(settings: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "run": {
            "created_at": now,
            "updated_at": now,
            "status": "running",
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "last_interruption": None,
        },
        "summary": {},
        "results": [],
    }


def load_or_create_checkpoint(
    path: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return new_checkpoint(settings)
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema version does not match")
    if checkpoint.get("settings") != settings:
        raise RuntimeError(
            "checkpoint settings differ from the requested benchmark configuration"
        )
    checkpoint["run"]["status"] = "running"
    checkpoint["run"]["updated_at"] = utc_now()
    return checkpoint


def update_summary(checkpoint: dict[str, Any]) -> None:
    results = checkpoint["results"]
    status_counts = Counter(str(result["status"]) for result in results)
    solved = [result for result in results if result["status"] == "solved"]
    checkpoint["summary"] = {
        "recorded_problem_count": len(results),
        "solved_problem_count": len(solved),
        "status_counts": dict(sorted(status_counts.items())),
        "total_recorded_seconds": sum(
            float(result.get("seconds", 0.0)) for result in results
        ),
        "total_questions_for_solved_problems": sum(
            int(result["question_count"]) for result in solved
        ),
    }
    checkpoint["run"]["updated_at"] = utc_now()


def write_checkpoint_atomic(path: Path, checkpoint: dict[str, Any]) -> None:
    """Fsync a temporary file and atomically replace the previous checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def problem_key(
    goal_count: int,
    alternative_count: int,
    problem_index: int,
) -> tuple[int, int, int]:
    return goal_count, alternative_count, problem_index


def deterministic_problem(
    base_seed: int,
    goal_count: int,
    alternative_count: int,
    problem_index: int,
):
    seed_components = [
        int(base_seed),
        int(goal_count),
        int(alternative_count),
        int(problem_index),
    ]
    rng = np.random.default_rng(np.random.SeedSequence(seed_components))
    alternatives, target_weights = generate_problem(
        rng=rng,
        goal_count=goal_count,
        alternative_count=alternative_count,
    )
    return alternatives, target_weights, seed_components


def run_one_problem(
    settings: dict[str, Any],
    goal_count: int,
    alternative_count: int,
    problem_index: int,
) -> dict[str, Any]:
    alternatives, target_weights, seed_components = deterministic_problem(
        base_seed=int(settings["seed"]),
        goal_count=goal_count,
        alternative_count=alternative_count,
        problem_index=problem_index,
    )
    started = time.perf_counter()
    common = {
        "goal_count": goal_count,
        "alternative_count": alternative_count,
        "problem_index": problem_index,
        "seed_components": seed_components,
        "alternatives_matrix": alternatives.entries,
        "target_weights": target_weights,
        "started_at": utc_now(),
    }
    try:
        result = solve_problem(
            alternatives=alternatives,
            target_weights=target_weights,
            max_questions=int(settings["max_questions"]),
            depth=2,
            parallelize_root=bool(settings["parallel_root"]),
            max_workers=int(settings["workers"]),
        )
        status = "solved" if bool(result["solved"]) else "max_questions"
        return {
            **common,
            **result,
            "status": status,
            "finished_at": utc_now(),
        }
    except Exception as exc:
        return {
            **common,
            "status": "error",
            "solved": False,
            "seconds": time.perf_counter() - started,
            "question_count": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at": utc_now(),
        }


def main() -> None:
    args = parse_args()
    validate_args(args)
    settings = build_settings(args)
    checkpoint = load_or_create_checkpoint(args.output_json, settings)
    update_summary(checkpoint)
    write_checkpoint_atomic(args.output_json, checkpoint)

    existing_by_key = {
        problem_key(
            int(result["goal_count"]),
            int(result["alternative_count"]),
            int(result["problem_index"]),
        ): result
        for result in checkpoint["results"]
    }
    try:
        for goal_count in settings["goals"]:
            for alternative_count in settings["alternatives"]:
                for problem_index in range(
                    1,
                    int(settings["problems_per_combination"]) + 1,
                ):
                    key = problem_key(goal_count, alternative_count, problem_index)
                    previous = existing_by_key.get(key)
                    if previous is not None and not (
                        args.retry_errors and previous["status"] == "error"
                    ):
                        continue

                    result = run_one_problem(
                        settings=settings,
                        goal_count=goal_count,
                        alternative_count=alternative_count,
                        problem_index=problem_index,
                    )
                    if previous is None:
                        checkpoint["results"].append(result)
                    else:
                        result_index = checkpoint["results"].index(previous)
                        checkpoint["results"][result_index] = result
                    existing_by_key[key] = result
                    update_summary(checkpoint)
                    write_checkpoint_atomic(args.output_json, checkpoint)
                    print(
                        f"g={goal_count} alternatives={alternative_count} "
                        f"problem={problem_index:03d} status={result['status']} "
                        f"questions={result.get('question_count')} "
                        f"seconds={float(result['seconds']):.3f}",
                        flush=True,
                    )
    except BaseException as exc:
        checkpoint["run"]["status"] = "interrupted"
        checkpoint["run"]["last_interruption"] = {
            "at": utc_now(),
            "type": type(exc).__name__,
            "message": str(exc),
        }
        update_summary(checkpoint)
        write_checkpoint_atomic(args.output_json, checkpoint)
        raise

    checkpoint["run"]["status"] = "complete"
    update_summary(checkpoint)
    write_checkpoint_atomic(args.output_json, checkpoint)
    print(
        f"complete: {checkpoint['summary']['solved_problem_count']}/"
        f"{settings['planned_problem_count']} solved; wrote {args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
