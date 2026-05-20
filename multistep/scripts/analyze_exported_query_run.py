from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.alternative_utility import compute_utility_value  # noqa: E402
from multistep.src.candidates import compute_candidate_set  # noqa: E402
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query  # noqa: E402
from multistep.src.query_probability import classify_query_answer  # noqa: E402
from multistep.src.weight_space import build_weight_space  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "multistep"
    / "data"
    / "termination_runs"
    / "goals5_alts10_seed1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay exported multistep query runs and compute exact candidate sets "
            "in the restricted weight space using multistep.src."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for writing the analysis result as JSON.",
    )
    parser.add_argument(
        "--print-steps",
        action="store_true",
        help="Print candidate counts after every answered query.",
    )
    return parser.parse_args()


def load_export(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def query_from_json(raw_query: dict[str, Any]) -> Query:
    return Query(
        ziel_index_a=int(raw_query["goal_index_a"]),
        ziel_index_b=int(raw_query["goal_index_b"]),
        value=float(raw_query["value"]),
    )


def answered_query_to_json(answered_query: AnsweredQuery) -> dict[str, Any]:
    return {
        "goal_index_a": int(answered_query.ziel_index_a),
        "goal_index_b": int(answered_query.ziel_index_b),
        "value": float(answered_query.value),
        "operator": answered_query.operator,
    }


def compute_target_best_alternatives(
    alternatives: AlternativenMatrix,
    target_weights: list[float],
    tolerance: float = 1e-12,
) -> list[int]:
    utility_values = [
        compute_utility_value(
            alternatives=alternatives,
            alternative_index=alternative_index,
            weights=target_weights,
        )
        for alternative_index in range(alternatives.get_anzahl_zeilen())
    ]
    max_utility = max(utility_values)
    return [
        alternative_index
        for alternative_index, utility_value in enumerate(utility_values)
        if utility_value >= max_utility - tolerance
    ]


def analyze_problem(
    raw_problem: dict[str, Any],
    print_steps: bool,
) -> dict[str, Any]:
    problem_index = int(raw_problem["problem_index"])
    alternatives = AlternativenMatrix(entries=raw_problem["alternatives_matrix"])
    target_weights = [float(value) for value in raw_problem["target_weights"]]
    answered_queries: list[AnsweredQuery] = []
    steps: list[dict[str, Any]] = []

    for step_index, raw_query in enumerate(raw_problem["queries"], start=1):
        query = query_from_json(raw_query)
        answer = classify_query_answer(
            weights=target_weights,
            query=query,
            equality_tol=0.0,
        )
        answered_query = query.answer(answer)
        answered_queries.append(answered_query)

        weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=answered_queries,
        )
        if not weight_space.is_feasible():
            raise RuntimeError(
                f"problem {problem_index}, step {step_index}: restricted weight space is infeasible"
            )

        candidates = compute_candidate_set(
            alternatives=alternatives,
            weight_space=weight_space,
        )
        step_result = {
            "step": step_index,
            "answered_query": answered_query_to_json(answered_query),
            "candidate_count": len(candidates),
            "candidates": [int(candidate) for candidate in candidates],
        }
        steps.append(step_result)

        if print_steps:
            print(
                f"  step {step_index:>2}: answer={answer}, "
                f"candidates={[candidate + 1 for candidate in candidates]}"
            )

    final_weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=answered_queries,
    )
    final_candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=final_weight_space,
    )
    target_best_alternatives = compute_target_best_alternatives(
        alternatives=alternatives,
        target_weights=target_weights,
    )

    return {
        "problem_index": problem_index,
        "query_count": len(answered_queries),
        "answered_queries": [
            answered_query_to_json(answered_query)
            for answered_query in answered_queries
        ],
        "final_candidate_count": len(final_candidates),
        "final_candidates": [int(candidate) for candidate in final_candidates],
        "target_best_alternatives": [
            int(alternative_index)
            for alternative_index in target_best_alternatives
        ],
        "steps": steps,
    }


def analyze_export(raw_export: dict[str, Any], print_steps: bool) -> dict[str, Any]:
    problem_results = []
    for raw_problem in raw_export.get("problems", []):
        problem_index = int(raw_problem["problem_index"])
        print(f"problem {problem_index}:")
        problem_result = analyze_problem(
            raw_problem=raw_problem,
            print_steps=print_steps,
        )
        problem_results.append(problem_result)
        print(
            "  final candidates: "
            f"{[candidate + 1 for candidate in problem_result['final_candidates']]}; "
            "target best: "
            f"{[candidate + 1 for candidate in problem_result['target_best_alternatives']]}"
        )

    return {
        "problems": problem_results,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    raw_export = load_export(args.input)
    analysis = analyze_export(
        raw_export=raw_export,
        print_steps=bool(args.print_steps),
    )
    if args.output_json is not None:
        write_json(path=args.output_json, data=analysis)


if __name__ == "__main__":
    main()
