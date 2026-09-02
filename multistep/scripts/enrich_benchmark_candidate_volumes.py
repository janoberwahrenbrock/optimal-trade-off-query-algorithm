from __future__ import annotations

"""Add exact per-state candidate volumes to an end-to-end benchmark JSON."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any


MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.optimized import OptimizedMultistepConfig  # noqa: E402
from multistep.optimized.value_function import (  # noqa: E402
    compute_candidate_analysis_for_mode,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query  # noqa: E402
from multistep.src.weight_space import build_weight_space  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", type=Path)
    return parser.parse_args()


def answered_query_from_record(record: dict[str, Any]) -> AnsweredQuery:
    query = Query(
        ziel_index_a=int(record["goal_index_a"]),
        ziel_index_b=int(record["goal_index_b"]),
        value=float(record["value"]),
    )
    return query.answer(record["answer"])


def snapshot_to_json(
    query_count: int,
    candidates: list[int],
    candidate_volumes: dict[int, float] | None,
    candidate_volume_shares: dict[int, float] | None,
) -> dict[str, Any]:
    return {
        "query_count": query_count,
        "candidates": [int(candidate) for candidate in candidates],
        "candidate_count": len(candidates),
        "candidate_volumes": candidate_volumes,
        "candidate_volume_shares": candidate_volume_shares,
    }


def enrich_result(result: dict[str, Any]) -> None:
    alternatives = AlternativenMatrix(entries=result["alternatives"])
    config = OptimizedMultistepConfig(
        candidate_count_mode="ratio_relevant",
        ratio_interval_engine="geometry",
    )
    answered_queries: list[AnsweredQuery] = []
    snapshots: list[dict[str, Any]] = []
    query_records = result["queries"]

    for query_count in range(len(query_records) + 1):
        weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=answered_queries,
        )
        analysis = compute_candidate_analysis_for_mode(
            alternatives=alternatives,
            weight_space=weight_space,
            candidate_subset=None,
            config=config,
            include_candidate_volumes=True,
        )
        snapshot = snapshot_to_json(
            query_count=query_count,
            candidates=analysis.candidates,
            candidate_volumes=analysis.candidate_volumes,
            candidate_volume_shares=analysis.candidate_volume_shares,
        )
        snapshots.append(snapshot)

        if query_count == 0:
            result["initial_candidates"] = snapshot["candidates"]
            result["initial_candidate_volumes"] = snapshot["candidate_volumes"]
            result["initial_candidate_volume_shares"] = snapshot[
                "candidate_volume_shares"
            ]
        else:
            query_record = query_records[query_count - 1]
            query_record["remaining_candidates"] = snapshot["candidates"]
            query_record["remaining_candidate_count"] = snapshot[
                "candidate_count"
            ]
            query_record["candidate_volumes"] = snapshot["candidate_volumes"]
            query_record["candidate_volume_shares"] = snapshot[
                "candidate_volume_shares"
            ]

        if query_count < len(query_records):
            answered_queries.append(
                answered_query_from_record(query_records[query_count])
            )

    result["candidate_volume_snapshots"] = snapshots


def main() -> None:
    args = parse_args()
    payload = json.loads(args.benchmark_json.read_text(encoding="utf-8"))
    for result in payload["results"]:
        enrich_result(result)

    temporary_path = args.benchmark_json.with_suffix(
        args.benchmark_json.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(args.benchmark_json)
    print(f"enriched {args.benchmark_json}", flush=True)


if __name__ == "__main__":
    main()
