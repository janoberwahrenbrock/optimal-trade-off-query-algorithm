from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import (
    AlternativenMatrix,
    AnsweredQuery,
    Query,
    TerminationResult,
    build_W,
    build_optimal_region_in_W,
    run_algorithmus,
)


DEFAULT_GOAL_COUNTS = [3, 5, 7, 9]
DEFAULT_ALTERNATIVE_COUNTS = [5, 10, 15, 20]
DEFAULT_REPETITIONS = 10
DEFAULT_SEED = 42
MAX_ALGORITHM_CALLS = 100
UTILITY_DECIMALS = 2
EQUALITY_TOL = 1e-12
ONE_HUNDRED_PERCENT_TOL = 1e-12
MAX_UINT32_SEED = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class SimulationRunResult:
    algorithm_calls: int
    termination_reason: str
    max_optimality_share: float
    has_100_percent_candidate: bool
    remaining_candidates: list[int]
    optimality_shares: dict[int, float]


class NonTerminationError(RuntimeError):
    def __init__(self, message: str, debug_info: dict[str, Any]) -> None:
        super().__init__(message)
        self.debug_info = debug_info


def parse_int_list(value: str) -> list[int]:
    try:
        parsed_values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Wert muss eine kommaseparierte Liste ganzer Zahlen sein."
        ) from exc

    if not parsed_values:
        raise argparse.ArgumentTypeError("Liste darf nicht leer sein.")

    if any(parsed_value <= 0 for parsed_value in parsed_values):
        raise argparse.ArgumentTypeError("Alle Werte muessen positiv sein.")

    return parsed_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Performance-Analyse des optimal trade-off query Algorithmus.",
    )
    parser.add_argument(
        "--goals",
        type=parse_int_list,
        default=DEFAULT_GOAL_COUNTS,
        help="Kommaseparierte Zielanzahlen. Default: 3,5,7,9",
    )
    parser.add_argument(
        "--alternatives",
        type=parse_int_list,
        default=DEFAULT_ALTERNATIVE_COUNTS,
        help="Kommaseparierte Alternativenanzahlen. Default: 5,10,15,20",
    )
    parser.add_argument(
        "-x",
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Wiederholungen je Konfiguration. Default: 10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random Seed fuer Problem-Generierung und Sampling. Default: 42",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=MAX_ALGORITHM_CALLS,
        help="Maximale Algorithmusaufrufe je Lauf. Default: 100",
    )
    args = parser.parse_args()

    if args.repetitions <= 0:
        parser.error("--repetitions muss positiv sein.")

    if args.max_calls <= 0:
        parser.error("--max-calls muss positiv sein.")

    return args


def generate_random_problem(
    n_goals: int,
    n_alternatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    utilities = np.round(
        rng.uniform(0.0, 1.0, size=(n_alternatives, n_goals)),
        UTILITY_DECIMALS,
    )
    target_weights = rng.dirichlet(alpha=np.ones(n_goals, dtype=float)).astype(float)
    return utilities, target_weights


def answer_query_from_target_weights(
    query: Query,
    target_weights: np.ndarray,
) -> AnsweredQuery:
    left_value = float(target_weights[query.ziel_index_a])
    right_value = float(query.value * target_weights[query.ziel_index_b])
    difference = left_value - right_value

    if abs(difference) <= EQUALITY_TOL:
        return query.answer("=")
    if difference < 0.0:
        return query.answer("<")
    return query.answer(">")


def build_nontermination_debug_info(
    alternativen_matrix: AlternativenMatrix,
    utilities: np.ndarray,
    target_weights: np.ndarray,
    answered_queries: list[AnsweredQuery],
    max_algorithm_calls: int,
) -> dict[str, Any]:
    query_identity_counts = Counter(
        build_answered_query_identity(answered_query)
        for answered_query in answered_queries
    )
    repeated_queries = [
        {
            "count": count,
            "ziel_index_a": ziel_index_a,
            "ziel_index_b": ziel_index_b,
            "value_rounded_12": value_rounded_12,
        }
        for (ziel_index_a, ziel_index_b, value_rounded_12), count
        in query_identity_counts.most_common()
        if count > 1
    ]

    return {
        "max_algorithm_calls": max_algorithm_calls,
        "answered_query_count": len(answered_queries),
        "unique_query_count": len(query_identity_counts),
        "repeated_query_count": sum(count - 1 for count in query_identity_counts.values()),
        "top_repeated_queries": repeated_queries[:10],
        "last_answered_queries": [
            answered_query.model_dump(mode="json")
            for answered_query in answered_queries[-10:]
        ],
        "candidate_status": build_candidate_status(
            alternativen_matrix=alternativen_matrix,
            answered_queries=answered_queries,
        ),
        "true_best_alternatives": build_true_best_alternatives(
            utilities=utilities,
            target_weights=target_weights,
        ),
        "target_weights": target_weights.astype(float).tolist(),
        "utilities": utilities.astype(float).tolist(),
    }


def build_answered_query_identity(answered_query: AnsweredQuery) -> tuple[int, int, float]:
    return (
        int(answered_query.ziel_index_a),
        int(answered_query.ziel_index_b),
        round(float(answered_query.value), 12),
    )


def build_candidate_status(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
) -> dict[str, Any]:
    try:
        W = build_W(alternativen_matrix.get_anzahl_spalten(), answered_queries)
        if not W.is_feasible():
            return {"status": "infeasible_W"}

        candidates: list[int] = []
        for alternative_index in range(alternativen_matrix.get_anzahl_zeilen()):
            optimal_region = build_optimal_region_in_W(
                alternativen_matrix=alternativen_matrix,
                W=W,
                alternative_index=alternative_index,
            )
            if optimal_region.is_feasible():
                candidates.append(alternative_index)

        return {
            "status": "ok",
            "candidate_count": len(candidates),
            "candidate_indices": candidates,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_true_best_alternatives(
    utilities: np.ndarray,
    target_weights: np.ndarray,
) -> dict[str, Any]:
    utility_values = utilities @ target_weights
    best_value = float(np.max(utility_values))
    best_indices = [
        int(index)
        for index, value in enumerate(utility_values)
        if abs(float(value) - best_value) <= 1e-12
    ]
    return {
        "best_indices": best_indices,
        "best_utility": best_value,
        "utility_values": utility_values.astype(float).tolist(),
    }


def run_until_termination(
    utilities: np.ndarray,
    target_weights: np.ndarray,
    max_algorithm_calls: int,
    sampling_seed: int | None = None,
) -> SimulationRunResult:
    alternativen_matrix = AlternativenMatrix(entries=utilities.tolist())
    answered_queries: list[AnsweredQuery] = []
    sampling_seed_rng = np.random.default_rng(sampling_seed)

    for call_index in range(1, max_algorithm_calls + 1):
        algorithm_seed = int(sampling_seed_rng.integers(0, MAX_UINT32_SEED))
        algorithm_output = run_algorithmus(
            alternativen_matrix=alternativen_matrix,
            answered_queries=answered_queries,
            seed=algorithm_seed,
        )

        if isinstance(algorithm_output, TerminationResult):
            max_share = max(float(value) for value in algorithm_output.optimality_shares.values())
            return SimulationRunResult(
                algorithm_calls=call_index,
                termination_reason=algorithm_output.reason,
                max_optimality_share=max_share,
                has_100_percent_candidate=max_share >= 1.0 - ONE_HUNDRED_PERCENT_TOL,
                remaining_candidates=[
                    int(candidate_index)
                    for candidate_index in algorithm_output.remaining_candidates
                ],
                optimality_shares={
                    int(candidate_index): float(optimality_share)
                    for candidate_index, optimality_share
                    in algorithm_output.optimality_shares.items()
                },
            )

        answered_queries.append(
            answer_query_from_target_weights(
                query=algorithm_output,
                target_weights=target_weights,
            )
        )

    raise NonTerminationError(
        (
            "Der Algorithmus hat nach "
            f"{max_algorithm_calls} Aufrufen nicht terminiert."
        ),
        debug_info=build_nontermination_debug_info(
            alternativen_matrix=alternativen_matrix,
            utilities=utilities,
            target_weights=target_weights,
            answered_queries=answered_queries,
            max_algorithm_calls=max_algorithm_calls,
        ),
    )


def format_reason_counts(reason_counts: Counter[str]) -> str:
    if not reason_counts:
        return "-"
    return ", ".join(
        f"{reason}: {count}"
        for reason, count in sorted(reason_counts.items())
    )


def build_summary_row(
    n_goals: int,
    n_alternatives: int,
    repetitions: int,
    completed_runs: list[SimulationRunResult],
    elapsed_seconds: float,
) -> dict[str, Any]:
    failures = [
        run_result
        for run_result in completed_runs
        if not run_result.has_100_percent_candidate
    ]
    calls = [run_result.algorithm_calls for run_result in completed_runs]
    remaining_candidate_counts = [
        len(run_result.remaining_candidates)
        for run_result in completed_runs
    ]
    reason_counts = Counter(run_result.termination_reason for run_result in completed_runs)
    completed_count = len(completed_runs)

    return {
        "Ziele": n_goals,
        "Alternativen": n_alternatives,
        "Laeufe": f"{completed_count}/{repetitions}",
        "Verbleibende Kandidaten Mittel": (
            float(np.mean(remaining_candidate_counts))
            if remaining_candidate_counts
            else 0.0
        ),
        "Anteil ohne 100%-Kandidat": (
            len(failures) / completed_count * 100.0
            if completed_count > 0
            else 0.0
        ),
        "Algorithmusaufrufe Mittel": float(np.mean(calls)) if calls else 0.0,
        "Algorithmusaufrufe Max": max(calls) if calls else 0,
        "Terminierungsgruende": format_reason_counts(reason_counts),
        "Sekunden": elapsed_seconds,
    }


def format_summary_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "Ziele",
        "Alternativen",
        "Laeufe",
        "Verbleibende Kandidaten Mittel",
        "Anteil ohne 100%-Kandidat",
        "Algorithmusaufrufe Mittel",
        "Algorithmusaufrufe Max",
        "Sekunden",
        "Terminierungsgruende",
    ]
    formatted_rows = [
        {
            **row,
            "Verbleibende Kandidaten Mittel": (
                f"{row['Verbleibende Kandidaten Mittel']:.2f}"
            ),
            "Anteil ohne 100%-Kandidat": f"{row['Anteil ohne 100%-Kandidat']:.2f}%",
            "Algorithmusaufrufe Mittel": f"{row['Algorithmusaufrufe Mittel']:.2f}",
            "Sekunden": f"{row['Sekunden']:.2f}",
        }
        for row in rows
    ]

    widths = {
        column: max(
            len(column),
            *(len(str(row[column])) for row in formatted_rows),
        )
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(str(row[column]).ljust(widths[column]) for column in columns)
        for row in formatted_rows
    ]
    return "\n".join([header, separator, *body])


def build_sampling_seed(root_seed: int, run_index: int) -> int:
    seed_sequence = np.random.SeedSequence([root_seed, run_index])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def print_run_result(
    repetition_index: int,
    repetitions: int,
    result: SimulationRunResult,
) -> None:
    has_100_percent_text = "ja" if result.has_100_percent_candidate else "nein"
    shares_text = format_optimality_shares(result.optimality_shares)
    print(
        f"  Lauf {repetition_index}/{repetitions}: "
        f"calls={result.algorithm_calls}, "
        f"reason={result.termination_reason}, "
        f"max_share={result.max_optimality_share:.6f}, "
        f"remaining={result.remaining_candidates}, "
        f"shares={shares_text}, "
        f"100%-Kandidat={has_100_percent_text}",
        flush=True,
    )


def format_optimality_shares(optimality_shares: dict[int, float]) -> str:
    return "{" + ", ".join(
        f"{candidate_index}: {share:.3f}"
        for candidate_index, share in sorted(optimality_shares.items())
    ) + "}"


def print_nontermination_debug(
    n_goals: int,
    n_alternatives: int,
    repetition_index: int,
    repetitions: int,
    debug_info: dict[str, Any],
) -> None:
    debug_payload = {
        "configuration": {
            "goals": n_goals,
            "alternatives": n_alternatives,
            "run": f"{repetition_index}/{repetitions}",
        },
        **debug_info,
    }
    print("\nNicht-Terminierung Debug:", flush=True)
    print(json.dumps(debug_payload, ensure_ascii=True, indent=2), flush=True)


def run_performance_analysis(
    goal_counts: list[int],
    alternative_counts: list[int],
    repetitions: int,
    seed: int,
    max_algorithm_calls: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    total_runs = len(goal_counts) * len(alternative_counts) * repetitions
    completed_total_runs = 0
    rows: list[dict[str, Any]] = []

    print(
        "Performance-Analyse gestartet: "
        f"goals={goal_counts}, alternatives={alternative_counts}, "
        f"x={repetitions}, seed={seed}, max_calls={max_algorithm_calls}",
        flush=True,
    )
    print(f"Gesamtlaeufe: {total_runs}", flush=True)

    for n_goals in goal_counts:
        for n_alternatives in alternative_counts:
            config_results: list[SimulationRunResult] = []
            config_start = perf_counter()
            print(
                "\n"
                f"Konfiguration: {n_goals} Ziele, "
                f"{n_alternatives} Alternativen",
                flush=True,
            )

            for repetition_index in range(1, repetitions + 1):
                completed_total_runs += 1
                print(
                    f"Fortschritt: {completed_total_runs}/{total_runs}",
                    flush=True,
                )
                utilities, target_weights = generate_random_problem(
                    n_goals=n_goals,
                    n_alternatives=n_alternatives,
                    rng=rng,
                )
                try:
                    run_result = run_until_termination(
                        utilities=utilities,
                        target_weights=target_weights,
                        max_algorithm_calls=max_algorithm_calls,
                        sampling_seed=build_sampling_seed(
                            root_seed=seed,
                            run_index=completed_total_runs,
                        ),
                    )
                except NonTerminationError as exc:
                    print_nontermination_debug(
                        n_goals=n_goals,
                        n_alternatives=n_alternatives,
                        repetition_index=repetition_index,
                        repetitions=repetitions,
                        debug_info=exc.debug_info,
                    )
                    raise

                config_results.append(run_result)
                print_run_result(
                    repetition_index=repetition_index,
                    repetitions=repetitions,
                    result=run_result,
                )

            elapsed_seconds = perf_counter() - config_start
            row = build_summary_row(
                n_goals=n_goals,
                n_alternatives=n_alternatives,
                repetitions=repetitions,
                completed_runs=config_results,
                elapsed_seconds=elapsed_seconds,
            )
            rows.append(row)

            print("  Zusammenfassung:", flush=True)
            print(format_summary_table([row]), flush=True)

    return rows


def main() -> None:
    args = parse_args()
    try:
        rows = run_performance_analysis(
            goal_counts=args.goals,
            alternative_counts=args.alternatives,
            repetitions=args.repetitions,
            seed=args.seed,
            max_algorithm_calls=args.max_calls,
        )
    except NonTerminationError as exc:
        print(f"\nAbbruch: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    print("\nEndergebnis:", flush=True)
    print(format_summary_table(rows), flush=True)


if __name__ == "__main__":
    main()
