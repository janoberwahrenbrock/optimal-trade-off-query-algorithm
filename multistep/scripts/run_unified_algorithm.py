#!/usr/bin/env python3
"""Run the unified exact algorithm with an interactive or simulated answerer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multistep.optimized import AlgorithmConfig, solve
from multistep.src.models import AlternativenMatrix, AnsweredQuery
from multistep.src.query_probability import classify_query_answer


def optional_positive_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("use a positive integer or 'none'")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True,
                        help="Problem with alternatives and optional initial answered_queries/target_weights.")
    parser.add_argument("--config-json", type=Path,
                        help="AlgorithmConfig object; explicitly supplied flags override its fields.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--simulate", action="store_true",
                        help="Use input target_weights as an external simulated answerer.")
    parser.add_argument("--query-strategy", choices=["fixed", "reduction_per_query"])
    parser.add_argument("--depth", type=int, choices=[1, 2, 3])
    parser.add_argument("--higher-depth-query-source", choices=["grid", "both"])
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--stop-rule", choices=["single_candidate", "volume_confidence"])
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--consider-stop-in-planning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-queries", type=optional_positive_int, default=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_json.resolve() in {args.input_json.resolve(), args.config_json.resolve() if args.config_json else None}:
        raise ValueError("output must differ from input and configuration")
    # An explicit --max-queries none must override a configured finite cap.
    fields = asdict(AlgorithmConfig())
    if args.config_json:
        fields.update(json.loads(args.config_json.read_text()))
    for key in tuple(fields):
        if hasattr(args, key) and getattr(args, key) is not None:
            fields[key] = getattr(args, key)
    if hasattr(args, "max_queries"):
        fields["max_queries"] = args.max_queries
    config = AlgorithmConfig(**fields)
    problem = json.loads(args.input_json.read_text())
    matrix = AlternativenMatrix(entries=problem["alternatives"])
    initial = [AnsweredQuery(**answer) for answer in problem.get("answered_queries", [])]
    weights = problem.get("target_weights")
    if args.simulate:
        if not isinstance(weights, list) or len(weights) != matrix.get_anzahl_spalten() or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
            for w in weights
        ) or not any(w > 0 for w in weights):
            raise ValueError("simulation requires one finite nonnegative target weight per goal and a positive total")

    def answer_query(query):
        if args.simulate:
            return classify_query_answer(weights=weights, query=query, equality_tol=0.0)
        while True:
            answer = input(f"w[{query.ziel_index_a}] / w[{query.ziel_index_b}] compared with {query.value:.12g} (<, >, =): ").strip()
            if answer in {"<", ">", "="}:
                return answer
            print("Please enter <, > or =.")

    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": hashlib.sha256(args.input_json.read_bytes()).hexdigest(),
              "configuration": asdict(config), "resolved_configuration": asdict(config.to_optimized_config()),
              "answer_source": "simulated_weights" if args.simulate else "interactive",
              "runtime": {"python": sys.version, "platform": platform.platform(),
                          "hostname": platform.node(), "multiprocessing_start_method": "forkserver"}}
    multiprocessing.set_start_method("forkserver")
    try:
        report["result"] = solve(matrix, config, answer_query, initial_answered_queries=initial)
        if args.simulate:
            utilities = [math.fsum(float(w) * float(value) for w, value in zip(weights, row)) for row in matrix.entries]
            winner = max(range(len(utilities)), key=lambda i: utilities[i])
            selected = report["result"]["selected_candidate"]
            report["benchmark_quality"] = {"target_winner": winner, "target_utilities": utilities,
                "selection_is_correct": selected == winner if selected is not None else None}
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output_json)
    print(json.dumps({"solved": report["result"]["solved"], "termination_reason": report["result"]["termination_reason"],
                      "queries": report["result"]["question_count"], "seconds": report["result"]["seconds"]}))


if __name__ == "__main__":
    main()
