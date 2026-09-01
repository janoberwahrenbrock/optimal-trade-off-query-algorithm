from __future__ import annotations

"""Reproducible dimension-7 benchmark matrix for query-policy experiments."""

import argparse
from dataclasses import asdict
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

from multistep.optimized.profiling import collect_optimization_profile  # noqa: E402
from multistep.optimized.value_function import (  # noqa: E402
    OptimizedMultistepConfig,
    compute_value_function_optimized,
    score_query_candidates_by_posterior,
)
from multistep.src.models import AlternativenMatrix, Query  # noqa: E402
from multistep.src.sampling import sample_points_with_diagnostics  # noqa: E402
from multistep.src.weight_space import build_weight_space  # noqa: E402


POLICIES = ("ratio", "ratio-quantile", "entropy", "regret", "grid-ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark corrected sampling, geometric intervals, and posterior "
            "query policies on deterministic seven-dimensional problems."
        )
    )
    parser.add_argument("--preset", choices=["quick", "standard"], default="quick")
    parser.add_argument("--problems", type=int, default=None)
    parser.add_argument("--depths", type=int, nargs="+", choices=[1, 2], default=None)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=None)
    parser.add_argument("--alternatives", type=int, default=10)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=5)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--ratio-engine",
        choices=["geometry", "lp"],
        default="geometry",
    )
    parser.add_argument(
        "--posterior-additions",
        type=int,
        default=21,
        help="Non-ratio queries retained by entropy/regret policies.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "quick":
        defaults = {
            "problems": 1,
            "depths": [1],
            "policies": ["ratio", "ratio-quantile", "entropy", "regret"],
            "samples": 240,
            "burn_in": 160,
            "grid_size": 7,
        }
    else:
        defaults = {
            "problems": 3,
            "depths": [1, 2],
            "policies": list(POLICIES),
            "samples": 800,
            "burn_in": 400,
            "grid_size": 21,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.problems <= 0 or args.samples <= 0 or args.burn_in < 0:
        raise ValueError("problems/samples must be positive and burn-in non-negative")
    if args.chains <= 0 or args.chains > args.samples:
        raise ValueError("chains must be between one and the sample count")
    if args.posterior_additions <= 0:
        raise ValueError("posterior-additions must be positive")
    return args


def build_policy_config(
    args: argparse.Namespace,
    policy: str,
    random_seed: int,
) -> OptimizedMultistepConfig:
    quantiles = (0.25, 0.5, 0.75) if policy in {
        "ratio-quantile",
        "entropy",
        "regret",
    } else ()
    posterior_objective = policy if policy in {"entropy", "regret"} else None
    return OptimizedMultistepConfig(
        sample_count=int(args.samples),
        burn_in=int(args.burn_in),
        thinning=int(args.thinning),
        random_seed=random_seed,
        sampling_chain_count=int(args.chains),
        grid_size=int(args.grid_size),
        canonical_grid_goal_pairs_only=True,
        skip_zero_probability_branches=True,
        pass_candidate_subset=True,
        use_ratio_terminal_counts=True,
        parallelize_root=args.workers > 1,
        max_workers=int(args.workers),
        candidate_count_mode="ratio_relevant",
        grid_depth_query_source_mode="both" if policy == "grid-ratio" else "ratio",
        depth_one_query_source_mode="ratio",
        ratio_interval_engine=str(args.ratio_engine),
        posterior_quantile_levels=quantiles,
        posterior_query_objective=posterior_objective,
        posterior_query_shortlist_size=(
            int(args.posterior_additions) if posterior_objective is not None else None
        ),
    )


def query_to_dict(query: Query | None) -> dict[str, Any] | None:
    if query is None:
        return None
    return {
        "goal_index_a": int(query.ziel_index_a),
        "goal_index_b": int(query.ziel_index_b),
        "value": float(query.value),
    }


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    for problem_index in range(1, int(args.problems) + 1):
        entries = rng.random((int(args.alternatives), 7))
        alternatives = AlternativenMatrix(entries=entries.tolist())
        weight_space = build_weight_space(goal_count=7, answered_queries=[])
        sampling_seed = int(args.seed) + 10_000 * problem_index
        sampling_started = time.perf_counter()
        diagnostic_samples, sampling_diagnostics = sample_points_with_diagnostics(
            system=weight_space,
            num_samples=int(args.samples),
            burn_in=int(args.burn_in),
            thinning=int(args.thinning),
            seed=sampling_seed,
            chain_count=int(args.chains),
        )
        sampling_seconds = time.perf_counter() - sampling_started

        for depth in args.depths:
            for policy in args.policies:
                config = build_policy_config(
                    args=args,
                    policy=policy,
                    random_seed=sampling_seed,
                )
                started = time.perf_counter()
                with collect_optimization_profile() as profile:
                    result = compute_value_function_optimized(
                        alternatives=alternatives,
                        answered_queries=[],
                        remaining_depth=int(depth),
                        config=config,
                    )
                seconds = time.perf_counter() - started
                best_posterior_score: dict[str, float] | None = None
                if result.best_query is not None:
                    score = score_query_candidates_by_posterior(
                        alternatives=alternatives,
                        query_candidates=[result.best_query],
                        samples=diagnostic_samples,
                    )[0]
                    best_posterior_score = {
                        "expected_entropy": score.expected_entropy,
                        "information_gain": score.information_gain,
                        "expected_regret": score.expected_regret,
                        "partition_balance": score.partition_balance,
                    }
                record = {
                    "problem": problem_index,
                    "depth": int(depth),
                    "policy": policy,
                    "ratio_engine": args.ratio_engine,
                    "seconds": seconds,
                    "value": result.value,
                    "candidate_count": result.candidate_count,
                    "root_query_count": len(result.query_evaluations),
                    "best_query": query_to_dict(result.best_query),
                    "best_query_posterior": best_posterior_score,
                    "sampling_seconds": sampling_seconds,
                    "sampling_diagnostics": asdict(sampling_diagnostics),
                    "profile_counters": dict(profile.counters),
                    "profile_seconds": dict(profile.seconds_by_operation),
                }
                records.append(record)
                print(
                    f"p={problem_index} d={depth} policy={policy:<14} "
                    f"seconds={seconds:8.3f} candidates={result.candidate_count:2d} "
                    f"queries={len(result.query_evaluations):3d} value={result.value:.5g}",
                    flush=True,
                )
    return records


def main() -> None:
    args = resolve_settings(parse_args())
    records = run_benchmark(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {"settings": vars(args), "results": records},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
