from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.candidates import compute_candidate_set
from multistep.src.models import AlternativenMatrix as MultistepAlternativenMatrix
from multistep.src.models import Query as MultistepQuery
from multistep.src.onestep_query_candidates import compute_onestep_query_candidates
from multistep.src.ratio_intervals import compute_all_ratio_intervals as compute_multistep_ratio_intervals
from multistep.src.weight_space import build_weight_space
from onestep.src.build_ungleichungssysteme import build_W, build_optimal_region_in_W
from onestep.src.io_models import AlternativenMatrix as OnestepAlternativenMatrix
from onestep.src.io_models import Query as OnestepQuery
from onestep.src.query_kandidaten import compute_all_query_kandidaten
from onestep.src.ratio_intervalle import compute_all_ratio_intervals as compute_onestep_ratio_intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy one-step query candidates with multistep terminal query candidates.",
    )
    parser.add_argument("--problems", type=int, default=100)
    parser.add_argument("--min-goals", type=int, default=3)
    parser.add_argument("--max-goals", type=int, default=7)
    parser.add_argument("--min-alternatives", type=int, default=3)
    parser.add_argument("--max-alternatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--query-epsilon", type=float, default=1e-3)
    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--rel-tol", type=float, default=1e-8)
    parser.add_argument("--show-mismatches", type=int, default=10)
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

    if args.abs_tol < 0.0:
        raise ValueError("--abs-tol must not be negative")

    if args.rel_tol < 0.0:
        raise ValueError("--rel-tol must not be negative")


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


def compute_legacy_query_candidates(
    entries: list[list[float]],
    query_epsilon: float,
) -> list[OnestepQuery]:
    alternatives = OnestepAlternativenMatrix(entries=entries)
    goal_count = alternatives.get_anzahl_spalten()
    weight_space = build_W(goal_count, [])
    candidates = [
        alternative_index
        for alternative_index in range(alternatives.get_anzahl_zeilen())
        if build_optimal_region_in_W(
            alternatives,
            weight_space,
            alternative_index,
        ).is_feasible()
    ]
    ratio_intervals = compute_onestep_ratio_intervals(
        alternativen_matrix=alternatives,
        answered_queries=[],
        kandidatenmenge=candidates,
    )
    return compute_all_query_kandidaten(
        zielpaar_intervalle_liste=ratio_intervals,
        epsilon=query_epsilon,
    )


def compute_multistep_terminal_query_candidates(
    entries: list[list[float]],
    query_epsilon: float,
) -> list[MultistepQuery]:
    alternatives = MultistepAlternativenMatrix(entries=entries)
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=[],
    )
    candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )
    ratio_intervals = compute_multistep_ratio_intervals(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
    )
    return compute_onestep_query_candidates(
        goal_pair_ratio_intervals=ratio_intervals,
        epsilon=query_epsilon,
    )


def canonical_query_key(
    query: OnestepQuery | MultistepQuery,
    abs_tol: float,
    rel_tol: float,
) -> tuple[int, int, int]:
    value = float(query.value)
    scale = max(abs_tol, abs(value) * rel_tol)
    if scale == 0.0:
        scale = 1e-12

    return (
        int(query.ziel_index_a),
        int(query.ziel_index_b),
        round(value / scale),
    )


def exact_query_keys(
    queries: list[OnestepQuery] | list[MultistepQuery],
    abs_tol: float,
    rel_tol: float,
) -> set[tuple[int, int, int]]:
    return {
        canonical_query_key(
            query=query,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        for query in queries
    }


def format_query(query: OnestepQuery | MultistepQuery) -> str:
    return (
        f"({int(query.ziel_index_a)}, {int(query.ziel_index_b)}, "
        f"{float(query.value):.8g})"
    )


def find_missing_examples(
    left_queries: list[OnestepQuery] | list[MultistepQuery],
    right_queries: list[OnestepQuery] | list[MultistepQuery],
    abs_tol: float,
    rel_tol: float,
    limit: int,
) -> list[str]:
    right_keys = exact_query_keys(
        queries=right_queries,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    examples: list[str] = []

    for query in left_queries:
        if canonical_query_key(query, abs_tol=abs_tol, rel_tol=rel_tol) not in right_keys:
            examples.append(format_query(query))
            if len(examples) >= limit:
                break

    return examples


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    mismatch_rows = []
    matching_count = 0

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
        legacy_queries = compute_legacy_query_candidates(
            entries=entries,
            query_epsilon=args.query_epsilon,
        )
        multistep_queries = compute_multistep_terminal_query_candidates(
            entries=entries,
            query_epsilon=args.query_epsilon,
        )
        legacy_keys = exact_query_keys(
            queries=legacy_queries,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
        multistep_keys = exact_query_keys(
            queries=multistep_queries,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )

        if legacy_keys == multistep_keys:
            matching_count += 1
            continue

        mismatch_rows.append(
            {
                "problem_index": problem_index,
                "goal_count": goal_count,
                "alternative_count": alternative_count,
                "legacy_count": len(legacy_queries),
                "multistep_count": len(multistep_queries),
                "legacy_minus_multistep": len(legacy_keys - multistep_keys),
                "multistep_minus_legacy": len(multistep_keys - legacy_keys),
                "legacy_examples": find_missing_examples(
                    left_queries=legacy_queries,
                    right_queries=multistep_queries,
                    abs_tol=args.abs_tol,
                    rel_tol=args.rel_tol,
                    limit=3,
                ),
                "multistep_examples": find_missing_examples(
                    left_queries=multistep_queries,
                    right_queries=legacy_queries,
                    abs_tol=args.abs_tol,
                    rel_tol=args.rel_tol,
                    limit=3,
                ),
            }
        )

    print("Configuration")
    print(f"  problems: {args.problems}")
    print(f"  goals: {args.min_goals}..{args.max_goals}")
    print(f"  alternatives: {args.min_alternatives}..{args.max_alternatives}")
    print(f"  seed: {args.seed}")
    print(f"  query epsilon: {args.query_epsilon}")
    print()
    print("Summary")
    print(f"  identical query candidate sets: {matching_count}")
    print(f"  different query candidate sets: {len(mismatch_rows)}")
    print(f"  match rate: {matching_count / args.problems:.2%}")

    if mismatch_rows:
        print()
        print(f"First {min(args.show_mismatches, len(mismatch_rows))} mismatches")
        print("  idx goals alts old_n new_n old-new new-old examples")
        for row in mismatch_rows[: args.show_mismatches]:
            print(
                f"  {row['problem_index']:>3} "
                f"{row['goal_count']:>5} "
                f"{row['alternative_count']:>4} "
                f"{row['legacy_count']:>5} "
                f"{row['multistep_count']:>5} "
                f"{row['legacy_minus_multistep']:>7} "
                f"{row['multistep_minus_legacy']:>7} "
                f"old_missing={row['legacy_examples']} "
                f"new_missing={row['multistep_examples']}"
            )


if __name__ == "__main__":
    main()
