from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.optimized import (  # noqa: E402
    OptimizedMultistepConfig,
    compute_ratio_relevant_candidate_set,
    compute_value_function_optimized,
)
from multistep.optimized.value_function import (  # noqa: E402
    is_query_already_answered,
    query_evaluation_sort_key,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery  # noqa: E402
from multistep.src.query_probability import classify_query_answer  # noqa: E402
from multistep.src.weight_space import build_weight_space  # noqa: E402


DEFAULT_GOALS = 3
DEFAULT_ALTERNATIVES = 10
DEFAULT_PROBLEMS = 10
DEFAULT_DEPTH = 2
DEFAULT_MAX_QUESTIONS = 50
DEFAULT_SAMPLE_COUNT = 400
DEFAULT_BURN_IN = 200
DEFAULT_THINNING = 5
DEFAULT_GRID_SIZE = 21
DEFAULT_MIN_QUERY_VALUE = 1e-3
DEFAULT_MAX_QUERY_VALUE = 100.0
DEFAULT_MIN_CONDITIONED_SAMPLE_COUNT = 50


@dataclass(frozen=True)
class ProblemRunResult:
    problem_index: int
    question_count: int
    initial_candidate_count: int
    final_candidate_count: int
    final_candidate: int | None
    seconds: float


@dataclass
class ExportLog:
    path: Path | None
    data: dict

    def write(self) -> None:
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random problems and apply the optimized multistep procedure "
            "until the ratio-relevant candidate set has size one."
        )
    )
    parser.add_argument("--goals", type=int, default=DEFAULT_GOALS)
    parser.add_argument("--alternatives", type=int, default=DEFAULT_ALTERNATIVES)
    parser.add_argument("--problems", type=int, default=DEFAULT_PROBLEMS)
    parser.add_argument("--start-problem", type=int, default=1)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--thinning", type=int, default=DEFAULT_THINNING)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--min-s", type=float, default=DEFAULT_MIN_QUERY_VALUE)
    parser.add_argument("--max-s", type=float, default=DEFAULT_MAX_QUERY_VALUE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--min-conditioned-samples",
        type=int,
        default=DEFAULT_MIN_CONDITIONED_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--conditioned-samples",
        action="store_true",
        help="Reuse branch-conditioned samples instead of resampling each state.",
    )
    parser.add_argument(
        "--all-goal-pair-orientations",
        action="store_true",
        help="Use both (i,j) and (j,i) for grid queries instead of canonical pairs.",
    )
    parser.add_argument(
        "--no-ratio-root-queries",
        action="store_true",
        help="Do not add ratio-generated queries on grid depths.",
    )
    parser.add_argument(
        "--root-query-source",
        choices=["grid", "ratio", "both"],
        default="both",
        help=(
            "Which query sources to use on depths greater than one. "
            "The intended policy is 'both': grid queries plus ratio-generated queries. "
            "For experiments, 'ratio' or 'grid' can be faster."
        ),
    )
    parser.add_argument(
        "--no-parallel-root",
        action="store_true",
        help="Disable root-query parallelization.",
    )
    parser.add_argument(
        "--print-problems",
        action="store_true",
        help="Print one result row per generated problem.",
    )
    parser.add_argument(
        "--debug-branches",
        action="store_true",
        help="Print branch details for the selected query in every call.",
    )
    parser.add_argument(
        "--stop-on-value-below-one",
        action="store_true",
        help="Raise immediately if the computed value is below one.",
    )
    parser.add_argument(
        "--disable-terminal-zero-fallback",
        action="store_true",
        help="Reproduce the old terminal shortcut behavior without repairing zero counts.",
    )
    parser.add_argument(
        "--validate-terminal-counts",
        action="store_true",
        help=(
            "Validate terminal ratio shortcut counts with exact child candidate "
            "counts. This is slower but useful for debugging."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output while solving.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="Write generated problems, target weights, queries, and answers to JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.goals <= 1:
        raise ValueError("--goals must be greater than one")

    if args.alternatives <= 0:
        raise ValueError("--alternatives must be positive")

    if args.problems <= 0:
        raise ValueError("--problems must be positive")

    if args.start_problem <= 0:
        raise ValueError("--start-problem must be positive")

    if args.depth <= 0:
        raise ValueError("--depth must be positive")

    if args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    if args.burn_in < 0:
        raise ValueError("--burn-in must not be negative")

    if args.thinning <= 0:
        raise ValueError("--thinning must be positive")

    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")

    if args.min_s <= 0.0:
        raise ValueError("--min-s must be positive")

    if args.max_s <= 0.0:
        raise ValueError("--max-s must be positive")

    if args.min_s > args.max_s:
        raise ValueError("--min-s must not be greater than --max-s")

    if args.workers <= 0:
        raise ValueError("--workers must be positive")


def generate_random_problem(
    rng: np.random.Generator,
    goal_count: int,
    alternative_count: int,
) -> tuple[AlternativenMatrix, list[float]]:
    utilities = rng.uniform(
        low=0.0,
        high=1.0,
        size=(alternative_count, goal_count),
    )
    target_weights = rng.dirichlet(alpha=np.ones(goal_count, dtype=float))
    return (
        AlternativenMatrix(entries=utilities.astype(float).tolist()),
        target_weights.astype(float).tolist(),
    )


def query_to_json(query: object) -> dict:
    return {
        "goal_index_a": int(getattr(query, "ziel_index_a")),
        "goal_index_b": int(getattr(query, "ziel_index_b")),
        "value": float(getattr(query, "value")),
        **(
            {"operator": str(getattr(query, "operator"))}
            if hasattr(query, "operator")
            else {}
        ),
    }


def build_export_log(args: argparse.Namespace) -> ExportLog:
    return ExportLog(
        path=args.export_json,
        data={
            "problems": [],
        },
    )


def start_problem_export(
    export_log: ExportLog,
    problem_index: int,
    alternatives: AlternativenMatrix,
    target_weights: list[float],
) -> dict | None:
    if export_log.path is None:
        return None

    problem_export = {
        "problem_index": int(problem_index),
        "alternatives_matrix": [
            [float(value) for value in row]
            for row in alternatives.entries
        ],
        "target_weights": [float(value) for value in target_weights],
        "queries": [],
    }
    export_log.data["problems"].append(problem_export)
    export_log.write()
    return problem_export


def build_config(args: argparse.Namespace, random_seed: int) -> OptimizedMultistepConfig:
    return OptimizedMultistepConfig(
        sample_count=int(args.samples),
        burn_in=int(args.burn_in),
        thinning=int(args.thinning),
        random_seed=random_seed,
        grid_size=int(args.grid_size),
        min_query_value=float(args.min_s),
        max_query_value=float(args.max_s),
        skip_zero_probability_branches=True,
        pass_candidate_subset=True,
        reuse_conditioned_samples=bool(args.conditioned_samples),
        min_conditioned_sample_count=int(args.min_conditioned_samples),
        use_ratio_terminal_counts=True,
        canonical_grid_goal_pairs_only=not bool(args.all_goal_pair_orientations),
        filter_answered_query_candidates=True,
        parallelize_root=not bool(args.no_parallel_root),
        max_workers=int(args.workers),
        candidate_count_mode="ratio_relevant",
        include_ratio_queries_on_grid_depths=not bool(args.no_ratio_root_queries),
        grid_depth_query_source_mode=str(args.root_query_source),
        depth_one_query_source_mode="ratio",
        repair_zero_terminal_counts=not bool(args.disable_terminal_zero_fallback),
        validate_ratio_terminal_counts=bool(args.validate_terminal_counts),
    )


def compute_current_candidates(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
) -> list[int]:
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=answered_queries,
    )
    if not weight_space.is_feasible():
        raise RuntimeError("current weight space is infeasible")

    return compute_ratio_relevant_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )


def solve_problem_until_termination(
    alternatives: AlternativenMatrix,
    target_weights: list[float],
    depth: int,
    max_questions: int,
    config: OptimizedMultistepConfig,
    problem_index: int,
    quiet: bool,
    debug_branches: bool,
    stop_on_value_below_one: bool,
    export_log: ExportLog | None = None,
    problem_export: dict | None = None,
) -> ProblemRunResult:
    start = time.perf_counter()
    answered_queries: list[AnsweredQuery] = []
    candidates = compute_current_candidates(
        alternatives=alternatives,
        answered_queries=answered_queries,
    )
    initial_candidate_count = len(candidates)
    question_count = 0

    if not quiet:
        print(
            f"problem {problem_index}: start with {initial_candidate_count} candidates",
            flush=True,
        )

    while len(candidates) > 1:
        if question_count >= max_questions:
            raise RuntimeError(
                f"Problem {problem_index} did not terminate after "
                f"{max_questions} questions"
            )

        if not quiet:
            print(
                f"problem {problem_index}: call {question_count + 1}, "
                f"current candidates={len(candidates)}",
                flush=True,
            )

        call_start = time.perf_counter()
        result = compute_value_function_optimized(
            alternatives=alternatives,
            answered_queries=answered_queries,
            remaining_depth=depth,
            config=config,
        )
        call_seconds = time.perf_counter() - call_start
        if result.best_query is None:
            raise RuntimeError(
                f"Problem {problem_index} has {len(candidates)} candidates left, "
                "but the procedure returned no query"
            )

        best_evaluation = min(result.query_evaluations, key=query_evaluation_sort_key)
        if debug_branches and not quiet:
            print(
                f"problem {problem_index}: best_query={result.best_query}, "
                f"source={best_evaluation.query_source}",
                flush=True,
            )
            for branch in best_evaluation.branches:
                print(
                    f"  branch {branch.answer}: "
                    f"p={branch.probability:.6g}, "
                    f"V={branch.child_value:.6g}, "
                    f"N={branch.child_candidate_count}, "
                    f"feasible={branch.is_child_feasible}",
                    flush=True,
                )

        if result.value < 1.0 and stop_on_value_below_one:
            raise RuntimeError(
                f"Problem {problem_index} produced value below one: "
                f"{result.value:.12g}"
            )

        if is_query_already_answered(
            query=result.best_query,
            answered_queries=answered_queries,
        ):
            raise RuntimeError(
                f"Problem {problem_index} returned an already answered query: "
                f"{result.best_query}"
            )

        answer = classify_query_answer(
            weights=target_weights,
            query=result.best_query,
            equality_tol=config.equality_tol,
        )
        if problem_export is not None:
            problem_export["queries"].append(query_to_json(result.best_query))
            export_log.write() if export_log is not None else None

        answered_queries.append(result.best_query.answer(answer))
        question_count += 1
        candidates = compute_current_candidates(
            alternatives=alternatives,
            answered_queries=answered_queries,
        )

        if not quiet:
            print(
                f"problem {problem_index}: answered {answer}, "
                f"value={result.value:.6g}, "
                f"root_queries={len(result.query_evaluations)}, "
                f"call_seconds={call_seconds:.3f}, "
                f"remaining_candidates={len(candidates)}",
                flush=True,
            )

    seconds = time.perf_counter() - start
    return ProblemRunResult(
        problem_index=problem_index,
        question_count=question_count,
        initial_candidate_count=initial_candidate_count,
        final_candidate_count=len(candidates),
        final_candidate=candidates[0] if len(candidates) == 1 else None,
        seconds=seconds,
    )


def run_analysis(args: argparse.Namespace, export_log: ExportLog | None = None) -> list[ProblemRunResult]:
    rng = np.random.default_rng(args.seed)
    results: list[ProblemRunResult] = []

    for _ in range(1, int(args.start_problem)):
        generate_random_problem(
            rng=rng,
            goal_count=int(args.goals),
            alternative_count=int(args.alternatives),
        )

    last_problem_index = int(args.start_problem) + int(args.problems) - 1
    for problem_index in range(int(args.start_problem), last_problem_index + 1):
        alternatives, target_weights = generate_random_problem(
            rng=rng,
            goal_count=int(args.goals),
            alternative_count=int(args.alternatives),
        )
        problem_export = (
            start_problem_export(
                export_log=export_log,
                problem_index=problem_index,
                alternatives=alternatives,
                target_weights=target_weights,
            )
            if export_log is not None
            else None
        )
        config = build_config(
            args=args,
            random_seed=int(args.seed) + problem_index - 1,
        )
        try:
            results.append(
                solve_problem_until_termination(
                    alternatives=alternatives,
                    target_weights=target_weights,
                    depth=int(args.depth),
                    max_questions=int(args.max_questions),
                    config=config,
                    problem_index=problem_index,
                    quiet=bool(args.quiet),
                    debug_branches=bool(args.debug_branches),
                    stop_on_value_below_one=bool(args.stop_on_value_below_one),
                    export_log=export_log,
                    problem_export=problem_export,
                )
            )
        except Exception as exc:
            export_log.write() if export_log is not None else None
            raise

    return results


def print_summary(results: list[ProblemRunResult], args: argparse.Namespace) -> None:
    question_counts = Counter(result.question_count for result in results)
    total_seconds = sum(result.seconds for result in results)

    print("Configuration")
    print(f"  goals: {args.goals}")
    print(f"  alternatives: {args.alternatives}")
    print(f"  problems: {args.problems}")
    print(f"  start problem: {args.start_problem}")
    print(f"  depth: {args.depth}")
    print(f"  max questions: {args.max_questions}")
    print(f"  samples: {args.samples}")
    print(f"  burn-in: {args.burn_in}")
    print(f"  thinning: {args.thinning}")
    print(f"  grid size: {args.grid_size}")
    print(f"  s range: [{args.min_s}, {args.max_s}]")
    print(f"  root query source: {args.root_query_source}")
    print(f"  seed: {args.seed}")
    print()

    print("Question-count distribution")
    print("  questions  problems  share")
    for question_count in sorted(question_counts):
        problem_count = question_counts[question_count]
        share = problem_count / len(results)
        print(f"  {question_count:>9}  {problem_count:>8}  {share:>6.1%}")
    print()

    question_values = [result.question_count for result in results]
    print("Summary")
    print(f"  solved problems: {len(results)}")
    print(f"  min questions: {min(question_values)}")
    print(f"  max questions: {max(question_values)}")
    print(f"  avg questions: {sum(question_values) / len(question_values):.3f}")
    print(f"  total runtime: {total_seconds:.3f} s")
    print(f"  avg runtime/problem: {total_seconds / len(results):.3f} s")

    if args.print_problems:
        print()
        print("Problem details")
        print("  idx  questions  initialK  finalK  finalCandidate  seconds")
        for result in results:
            final_candidate = (
                "-" if result.final_candidate is None else str(result.final_candidate + 1)
            )
            print(
                f"  {result.problem_index:>3}  "
                f"{result.question_count:>9}  "
                f"{result.initial_candidate_count:>8}  "
                f"{result.final_candidate_count:>6}  "
                f"{final_candidate:>14}  "
                f"{result.seconds:>7.3f}"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    export_log = build_export_log(args)
    try:
        results = run_analysis(args, export_log=export_log)
    except Exception:
        export_log.write()
        raise
    export_log.write()
    print_summary(results=results, args=args)


if __name__ == "__main__":
    main()
