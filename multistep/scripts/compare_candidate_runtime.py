from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.candidates import (
    compute_candidate_set,
    estimate_candidate_set_from_samples,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery
from multistep.src.sampling import sample_points_from_constraint_system
from multistep.src.weight_space import build_weight_space


@dataclass(frozen=True)
class CandidateRuntimeResult:
    goal_count: int
    alternative_count: int
    sample_count: int
    answered_query_count: int
    repetition: int
    exact_seconds: float
    sampling_seconds: float
    sample_candidate_seconds: float
    exact_candidate_count: int
    sample_candidate_count: int
    missed_candidate_count: int
    extra_candidate_count: int

    @property
    def sample_total_seconds(self) -> float:
        return self.sampling_seconds + self.sample_candidate_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact LP-based candidate set computation with sample-based "
            "candidate set estimation across multiple configurations."
        ),
    )
    parser.add_argument("--goals", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--alternatives", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--samples", type=int, nargs="+", default=[300, 1500])
    parser.add_argument("--answered-queries", type=int, nargs="+", default=[0])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thinning", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print machine-readable CSV instead of an aligned table.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(goal_count < 2 for goal_count in args.goals):
        raise ValueError("--goals entries must be at least 2")

    if any(alternative_count < 2 for alternative_count in args.alternatives):
        raise ValueError("--alternatives entries must be at least 2")

    if any(sample_count <= 0 for sample_count in args.samples):
        raise ValueError("--samples entries must be positive")

    if any(answered_query_count < 0 for answered_query_count in args.answered_queries):
        raise ValueError("--answered-queries entries must not be negative")

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    if args.burn_in < 0:
        raise ValueError("--burn-in must not be negative")

    if args.thinning <= 0:
        raise ValueError("--thinning must be positive")


def generate_random_problem(
    goal_count: int,
    alternative_count: int,
    rng: np.random.Generator,
) -> AlternativenMatrix:
    entries = np.round(
        rng.uniform(0.0, 1.0, size=(alternative_count, goal_count)),
        2,
    )
    return AlternativenMatrix(entries=entries.tolist())


def generate_feasible_answered_queries(
    goal_count: int,
    answered_query_count: int,
    rng: np.random.Generator,
) -> list[AnsweredQuery]:
    true_weights = rng.dirichlet(np.ones(goal_count))
    answered_queries: list[AnsweredQuery] = []

    for _ in range(answered_query_count):
        goal_index_a, goal_index_b = rng.choice(goal_count, size=2, replace=False)
        true_ratio = true_weights[goal_index_a] / true_weights[goal_index_b]
        query_value = float(true_ratio * np.exp(rng.uniform(-1.0, 1.0)))

        if true_weights[goal_index_a] < query_value * true_weights[goal_index_b]:
            operator = "<"
        elif true_weights[goal_index_a] > query_value * true_weights[goal_index_b]:
            operator = ">"
        else:
            operator = "="

        answered_queries.append(
            AnsweredQuery(
                ziel_index_a=int(goal_index_a),
                ziel_index_b=int(goal_index_b),
                value=query_value,
                operator=operator,
            )
        )

    return answered_queries


def time_call(function):
    start = perf_counter()
    result = function()
    return perf_counter() - start, result


def run_single_benchmark(
    goal_count: int,
    alternative_count: int,
    sample_count: int,
    answered_query_count: int,
    repetition: int,
    burn_in: int,
    thinning: int,
    seed: int,
) -> CandidateRuntimeResult:
    rng = np.random.default_rng(seed)
    alternatives = generate_random_problem(
        goal_count=goal_count,
        alternative_count=alternative_count,
        rng=rng,
    )
    answered_queries = generate_feasible_answered_queries(
        goal_count=goal_count,
        answered_query_count=answered_query_count,
        rng=rng,
    )
    weight_space = build_weight_space(
        goal_count=goal_count,
        answered_queries=answered_queries,
    )

    exact_seconds, exact_candidates = time_call(
        lambda: compute_candidate_set(
            alternatives=alternatives,
            weight_space=weight_space,
        )
    )
    sampling_seconds, samples = time_call(
        lambda: sample_points_from_constraint_system(
            system=weight_space,
            num_samples=sample_count,
            burn_in=burn_in,
            thinning=thinning,
            seed=seed + 10_000,
        )
    )
    sample_candidate_seconds, sample_candidates = time_call(
        lambda: estimate_candidate_set_from_samples(
            alternatives=alternatives,
            samples=samples,
        )
    )

    exact_candidate_set = set(exact_candidates)
    sample_candidate_set = set(sample_candidates)

    return CandidateRuntimeResult(
        goal_count=goal_count,
        alternative_count=alternative_count,
        sample_count=sample_count,
        answered_query_count=answered_query_count,
        repetition=repetition,
        exact_seconds=exact_seconds,
        sampling_seconds=sampling_seconds,
        sample_candidate_seconds=sample_candidate_seconds,
        exact_candidate_count=len(exact_candidates),
        sample_candidate_count=len(sample_candidates),
        missed_candidate_count=len(exact_candidate_set - sample_candidate_set),
        extra_candidate_count=len(sample_candidate_set - exact_candidate_set),
    )


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f}"


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf")

    return numerator / denominator


def aggregate_results(
    results: list[CandidateRuntimeResult],
) -> list[dict[str, float | int]]:
    grouped_results: dict[tuple[int, int, int, int], list[CandidateRuntimeResult]] = (
        defaultdict(list)
    )
    for result in results:
        grouped_results[
            (
                result.goal_count,
                result.alternative_count,
                result.sample_count,
                result.answered_query_count,
            )
        ].append(result)

    rows: list[dict[str, float | int]] = []
    for (
        goal_count,
        alternative_count,
        sample_count,
        answered_query_count,
    ), group in sorted(grouped_results.items()):
        exact_seconds = mean([result.exact_seconds for result in group])
        sampling_seconds = mean([result.sampling_seconds for result in group])
        sample_candidate_seconds = mean(
            [result.sample_candidate_seconds for result in group]
        )
        sample_total_seconds = mean([result.sample_total_seconds for result in group])

        rows.append(
            {
                "goals": goal_count,
                "alts": alternative_count,
                "samples": sample_count,
                "answered": answered_query_count,
                "exact_ms": exact_seconds * 1000.0,
                "sampling_ms": sampling_seconds * 1000.0,
                "sample_k_ms": sample_candidate_seconds * 1000.0,
                "sample_total_ms": sample_total_seconds * 1000.0,
                "sample_k_over_exact": ratio(
                    sample_candidate_seconds,
                    exact_seconds,
                ),
                "sample_total_over_exact": ratio(
                    sample_total_seconds,
                    exact_seconds,
                ),
                "exact_k": mean([result.exact_candidate_count for result in group]),
                "sample_k": mean([result.sample_candidate_count for result in group]),
                "missed": mean([result.missed_candidate_count for result in group]),
                "extra": mean([result.extra_candidate_count for result in group]),
            }
        )

    return rows


def print_table(rows: list[dict[str, float | int]]) -> None:
    headers = [
        "goals",
        "alts",
        "samples",
        "answered",
        "exact_ms",
        "sampling_ms",
        "sampleK_ms",
        "sampleTotal_ms",
        "sampleK/exact",
        "sampleTotal/exact",
        "exactK",
        "sampleK",
        "missed",
        "extra",
    ]
    print(
        " ".join(
            [
                f"{headers[0]:>5}",
                f"{headers[1]:>4}",
                f"{headers[2]:>7}",
                f"{headers[3]:>8}",
                f"{headers[4]:>10}",
                f"{headers[5]:>11}",
                f"{headers[6]:>11}",
                f"{headers[7]:>14}",
                f"{headers[8]:>13}",
                f"{headers[9]:>17}",
                f"{headers[10]:>7}",
                f"{headers[11]:>7}",
                f"{headers[12]:>7}",
                f"{headers[13]:>7}",
            ]
        )
    )

    for row in rows:
        print(
            " ".join(
                [
                    f"{int(row['goals']):>5}",
                    f"{int(row['alts']):>4}",
                    f"{int(row['samples']):>7}",
                    f"{int(row['answered']):>8}",
                    f"{float(row['exact_ms']):>10.2f}",
                    f"{float(row['sampling_ms']):>11.2f}",
                    f"{float(row['sample_k_ms']):>11.2f}",
                    f"{float(row['sample_total_ms']):>14.2f}",
                    f"{float(row['sample_k_over_exact']):>13.2f}",
                    f"{float(row['sample_total_over_exact']):>17.2f}",
                    f"{float(row['exact_k']):>7.2f}",
                    f"{float(row['sample_k']):>7.2f}",
                    f"{float(row['missed']):>7.2f}",
                    f"{float(row['extra']):>7.2f}",
                ]
            )
        )


def print_csv(rows: list[dict[str, float | int]]) -> None:
    headers = [
        "goals",
        "alternatives",
        "samples",
        "answered_queries",
        "exact_ms",
        "sampling_ms",
        "sample_candidate_ms",
        "sample_total_ms",
        "sample_candidate_over_exact",
        "sample_total_over_exact",
        "exact_candidate_count",
        "sample_candidate_count",
        "missed_candidate_count",
        "extra_candidate_count",
    ]
    print(",".join(headers))

    for row in rows:
        print(
            ",".join(
                [
                    str(int(row["goals"])),
                    str(int(row["alts"])),
                    str(int(row["samples"])),
                    str(int(row["answered"])),
                    f"{float(row['exact_ms']):.6f}",
                    f"{float(row['sampling_ms']):.6f}",
                    f"{float(row['sample_k_ms']):.6f}",
                    f"{float(row['sample_total_ms']):.6f}",
                    f"{float(row['sample_k_over_exact']):.6f}",
                    f"{float(row['sample_total_over_exact']):.6f}",
                    f"{float(row['exact_k']):.6f}",
                    f"{float(row['sample_k']):.6f}",
                    f"{float(row['missed']):.6f}",
                    f"{float(row['extra']):.6f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    results: list[CandidateRuntimeResult] = []
    for goal_count in args.goals:
        for alternative_count in args.alternatives:
            for sample_count in args.samples:
                for answered_query_count in args.answered_queries:
                    for repetition in range(args.repetitions):
                        seed = (
                            args.seed
                            + 1_000_000 * goal_count
                            + 100_000 * alternative_count
                            + 10 * sample_count
                            + 1_000 * answered_query_count
                            + repetition
                        )
                        results.append(
                            run_single_benchmark(
                                goal_count=goal_count,
                                alternative_count=alternative_count,
                                sample_count=sample_count,
                                answered_query_count=answered_query_count,
                                repetition=repetition,
                                burn_in=args.burn_in,
                                thinning=args.thinning,
                                seed=seed,
                            )
                        )

    rows = aggregate_results(results)
    if args.csv:
        print_csv(rows)
    else:
        print_table(rows)
        print()
        print("Interpretation:")
        print("  sampleK/exact < 1 means sample-based K is faster if samples already exist.")
        print("  sampleTotal/exact < 1 means sampling plus sample-based K is faster.")
        print("  missed > 0 means the sample-based estimate missed exact candidates.")


if __name__ == "__main__":
    main()
