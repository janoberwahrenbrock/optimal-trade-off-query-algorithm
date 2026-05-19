from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.optimized.value_function import (
    OptimizedMultistepConfig,
    compute_value_function_optimized,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query
from multistep.src.value_function import MultistepConfig, compute_value_function


DEFAULT_CASE_PATH = REPO_ROOT / "onestep" / "data" / "a5_a10_case.json"

BenchmarkMode = Literal[
    "reference",
    "skip-only",
    "pass-only",
    "conditioned-only",
    "terminal-only",
    "canonical-grid-only",
    "parallel-only",
    "all-without-conditioned-canonical-grid",
    "all-without-conditioned",
    "optimized-all",
]


@dataclass(frozen=True)
class CaseData:
    goal_labels: list[str]
    entries: list[list[float]]
    answered_queries: list[AnsweredQuery]


@dataclass(frozen=True)
class OptimizationFlags:
    skip: bool
    pass_subset: bool
    conditioned: bool
    terminal: bool
    canonical_grid: bool
    parallel: bool


@dataclass(frozen=True)
class BenchmarkResult:
    mode: BenchmarkMode
    seconds: float
    value: float
    best_query: Query | None
    root_query_evaluations: int
    flags: OptimizationFlags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare isolated optimized depth-2 evaluations on a5/a10."
    )
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thinning", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--max-query-value", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-conditioned-samples", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "reference",
            "skip-only",
            "pass-only",
            "conditioned-only",
            "terminal-only",
            "canonical-grid-only",
            "parallel-only",
            "all-without-conditioned-canonical-grid",
            "all-without-conditioned",
            "optimized-all",
        ],
        default="all",
    )
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
        goal_labels=[str(label) for label in raw_case["goal_labels"]],
        entries=[
            [float(value) for value in alternative]
            for alternative in raw_case["handlungsalternativenmatrix"]
        ],
        answered_queries=answered_queries,
    )


def resolve_modes(mode: str) -> list[BenchmarkMode]:
    if mode != "all":
        return [mode]  # type: ignore[list-item]

    return [
        "reference",
        "skip-only",
        "pass-only",
        "conditioned-only",
        "terminal-only",
        "canonical-grid-only",
        "parallel-only",
        "all-without-conditioned-canonical-grid",
        "all-without-conditioned",
        "optimized-all",
    ]


def get_flags_for_mode(mode: BenchmarkMode) -> OptimizationFlags:
    if mode == "reference":
        return OptimizationFlags(
            skip=False,
            pass_subset=False,
            conditioned=False,
            terminal=False,
            canonical_grid=False,
            parallel=False,
        )

    if mode == "skip-only":
        return OptimizationFlags(
            skip=True,
            pass_subset=False,
            conditioned=False,
            terminal=False,
            canonical_grid=False,
            parallel=False,
        )

    if mode == "pass-only":
        return OptimizationFlags(
            skip=False,
            pass_subset=True,
            conditioned=False,
            terminal=False,
            canonical_grid=False,
            parallel=False,
        )

    if mode == "conditioned-only":
        return OptimizationFlags(
            skip=False,
            pass_subset=False,
            conditioned=True,
            terminal=False,
            canonical_grid=False,
            parallel=False,
        )

    if mode == "terminal-only":
        return OptimizationFlags(
            skip=False,
            pass_subset=False,
            conditioned=False,
            terminal=True,
            canonical_grid=False,
            parallel=False,
        )

    if mode == "canonical-grid-only":
        return OptimizationFlags(
            skip=False,
            pass_subset=False,
            conditioned=False,
            terminal=False,
            canonical_grid=True,
            parallel=False,
        )

    if mode == "parallel-only":
        return OptimizationFlags(
            skip=False,
            pass_subset=False,
            conditioned=False,
            terminal=False,
            canonical_grid=False,
            parallel=True,
        )

    if mode == "all-without-conditioned":
        return OptimizationFlags(
            skip=True,
            pass_subset=True,
            conditioned=False,
            terminal=True,
            canonical_grid=True,
            parallel=True,
        )

    if mode == "all-without-conditioned-canonical-grid":
        return OptimizationFlags(
            skip=True,
            pass_subset=True,
            conditioned=False,
            terminal=True,
            canonical_grid=False,
            parallel=True,
        )

    if mode == "optimized-all":
        return OptimizationFlags(
            skip=True,
            pass_subset=True,
            conditioned=True,
            terminal=True,
            canonical_grid=True,
            parallel=True,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def build_optimized_config(
    flags: OptimizationFlags,
    args: argparse.Namespace,
) -> OptimizedMultistepConfig:
    return OptimizedMultistepConfig(
        sample_count=args.samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        random_seed=args.seed,
        grid_size=args.grid_size,
        max_query_value=args.max_query_value,
        skip_zero_probability_branches=flags.skip,
        pass_candidate_subset=flags.pass_subset,
        reuse_conditioned_samples=flags.conditioned,
        min_conditioned_sample_count=args.min_conditioned_samples,
        use_ratio_terminal_counts=flags.terminal,
        canonical_grid_goal_pairs_only=flags.canonical_grid,
        parallelize_root=flags.parallel,
        max_workers=args.workers,
    )


def format_query(query: Query | None, goal_labels: list[str]) -> str:
    if query is None:
        return "-"

    goal_a = goal_labels[int(query.ziel_index_a)]
    goal_b = goal_labels[int(query.ziel_index_b)]
    return f"{goal_a} ? {float(query.value):.8g} * {goal_b}"


def run_benchmark_mode(
    mode: BenchmarkMode,
    case_data: CaseData,
    args: argparse.Namespace,
) -> BenchmarkResult:
    alternatives = AlternativenMatrix(entries=case_data.entries)
    flags = get_flags_for_mode(mode)

    start = time.perf_counter()

    if mode == "reference":
        result = compute_value_function(
            alternatives=alternatives,
            answered_queries=case_data.answered_queries,
            remaining_depth=2,
            config=MultistepConfig(
                sample_count=args.samples,
                burn_in=args.burn_in,
                thinning=args.thinning,
                random_seed=args.seed,
                grid_size=args.grid_size,
                max_query_value=args.max_query_value,
            ),
        )
    else:
        result = compute_value_function_optimized(
            alternatives=alternatives,
            answered_queries=case_data.answered_queries,
            remaining_depth=2,
            config=build_optimized_config(flags=flags, args=args),
        )

    seconds = time.perf_counter() - start

    return BenchmarkResult(
        mode=mode,
        seconds=seconds,
        value=result.value,
        best_query=result.best_query,
        root_query_evaluations=len(result.query_evaluations),
        flags=flags,
    )


def main() -> None:
    args = parse_args()
    case_data = load_case(args.case)
    modes = resolve_modes(args.mode)

    results = [
        run_benchmark_mode(
            mode=mode,
            case_data=case_data,
            args=args,
        )
        for mode in modes
    ]

    reference_result = results[0]
    reference_seconds = reference_result.seconds
    reference_value = reference_result.value
    reference_query = reference_result.best_query

    print("Configuration")
    print(f"  case: {args.case}")
    print(f"  samples: {args.samples}")
    print(f"  burn-in: {args.burn_in}")
    print(f"  thinning: {args.thinning}")
    print(f"  seed: {args.seed}")
    print(f"  grid-size: {args.grid_size}")
    print(f"  workers: {args.workers}")
    print(f"  min-conditioned-samples: {args.min_conditioned_samples}")
    print()
    print("Flags")
    print("  skip=skip_zero_probability_branches")
    print("  pass=pass_candidate_subset")
    print("  cond=reuse_conditioned_samples")
    print("  term=use_ratio_terminal_counts")
    print("  cgrid=canonical_grid_goal_pairs_only")
    print("  par=parallelize_root")
    print()
    print("Results")
    print(
        "  mode                         "
        "skip pass cond term cgrid par "
        " seconds  speedup      value same_q same_v evals best_query"
    )

    for result in results:
        speedup = reference_seconds / result.seconds if result.seconds else 0.0
        same_query = result.best_query == reference_query
        same_value = abs(result.value - reference_value) <= 1e-12

        print(
            f"  {result.mode:<28} "
            f"{int(result.flags.skip):>4} "
            f"{int(result.flags.pass_subset):>4} "
            f"{int(result.flags.conditioned):>4} "
            f"{int(result.flags.terminal):>4} "
            f"{int(result.flags.canonical_grid):>5} "
            f"{int(result.flags.parallel):>3} "
            f"{result.seconds:>8.3f} "
            f"{speedup:>8.2f}x "
            f"{result.value:>10.6g} "
            f"{str(same_query):>6} "
            f"{str(same_value):>6} "
            f"{result.root_query_evaluations:>5} "
            f"{format_query(result.best_query, case_data.goal_labels)}"
        )


if __name__ == "__main__":
    main()
