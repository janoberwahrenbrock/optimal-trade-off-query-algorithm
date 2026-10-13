#!/usr/bin/env python3
"""Replay one fixed query-efficiency case with unchanged mathematical evaluation and external observers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import inspect
import itertools
import fcntl
import functools
import traceback
from typing import Any


_POOL_SHUTDOWNS: list[dict[str, Any]] = []
_PROGRESS_PATH: str | None = None
_CHECKPOINT_PATH: str | None = None
_LATEST_SELECTED_PLAN: dict[str, Any] = {}
_CURRENT_PLANNING: dict[str, Any] = {}
_CURRENT_BATCH: dict[str, Any] = {}
_TASK_SEQUENCE = itertools.count(1)
_PLANNING_SEQUENCE = itertools.count(1)
_BATCH_SEQUENCE = itertools.count(1)
_ROOT_TASKS_SUBMITTED = 0
STOP_AWARE_LOSS_SEMANTICS = 'one_for_single_candidate_or_volume_confidence_else_actual_candidate_count'


def journal(path: str | None, event: str, **fields: Any) -> None:
    """Append one small record under an interprocess lock; never inspect recursion."""
    if path is None:
        return
    record = {"event": event, "at_utc": now(), "monotonic_seconds": time.monotonic(),
              "pid": os.getpid(), **fields}
    data = (json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
    finally:
        os.close(descriptor)


def query_signature(query: Any) -> dict[str, Any] | None:
    if query is None:
        return None
    return {"goal_index_a": int(query.ziel_index_a), "goal_index_b": int(query.ziel_index_b),
            "value": float(query.value), "value_hex": float(query.value).hex()}


def checkpoint(event: str, **fields: Any) -> None:
    """Persist actual question boundaries; this does not resume inside a solve."""
    if _CHECKPOINT_PATH is None:
        return
    record = {"event": event, "at_utc": now(), "monotonic_seconds": time.monotonic(),
              "pid": os.getpid(), **fields}
    data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(_CHECKPOINT_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(str(Path(_CHECKPOINT_PATH).parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def submitted_payloads(function: Any, arguments: tuple[Any, ...]) -> list[Any]:
    # Standard ProcessPoolExecutor.map submits partial(_process_chunk, worker), (chunk,).
    # Retain support for direct _process_chunk submissions as well.
    if isinstance(function, functools.partial) and getattr(function.func, "__name__", "") == "_process_chunk" and len(arguments) == 1:
        return [item[0] for item in arguments[0] if len(item) == 1]
    if getattr(function, "__name__", "") == "_process_chunk" and len(arguments) == 2:
        return [item[0] for item in arguments[1] if len(item) == 1]
    if getattr(function, "__name__", "") in {"_evaluate_query_candidate_worker", "_evaluate_efficiency_worker"} and len(arguments) == 1:
        return [arguments[0]]
    return []


def observed_worker_job(function: Any, arguments: tuple[Any, ...], keywords: dict[str, Any],
                        observation: dict[str, Any]) -> Any:
    """Picklable process entry: invoke the unchanged submitted callable and payload."""
    path = observation.pop("progress_path")
    started = time.monotonic()
    journal(path, "root_task_start", **observation)
    outcome = "completed"
    error_details: dict[str, Any] = {}
    try:
        return function(*arguments, **keywords)
    except BaseException as error:
        outcome = "error"
        error_details = {"error_type": type(error).__name__, "error_message": str(error),
                         "traceback": traceback.format_exc()}
        raise
    finally:
        journal(path, "root_task_end", **observation, **error_details,
                duration_seconds=time.monotonic() - started, outcome=outcome)


class ObservedProcessPoolExecutor(ProcessPoolExecutor):
    """Observe existing root submissions and ordinary pool shutdown."""

    def submit(self, function: Any, /, *arguments: Any, **keywords: Any) -> Any:
        global _ROOT_TASKS_SUBMITTED
        if _PROGRESS_PATH is None:
            return super().submit(function, *arguments, **keywords)
        payloads = submitted_payloads(function, arguments)
        if not payloads:
            return super().submit(function, *arguments, **keywords)
        tasks = [{"query": query_signature(payload[3]), "remaining_depth": int(payload[5]),
                  "query_source": str(payload[9])} for payload in payloads]
        _ROOT_TASKS_SUBMITTED += len(tasks)
        observation = {**{**_CURRENT_PLANNING, **_CURRENT_BATCH},
            "progress_path": _PROGRESS_PATH,
            "task_id": f"{os.getpid()}-{next(_TASK_SEQUENCE)}", "root_queries_in_task": len(tasks),
            "queries": tasks}
        return super().submit(observed_worker_job, function, arguments, keywords, observation)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        processes = self._processes
        if processes is not None:
            _POOL_SHUTDOWNS.append({"worker_pids": sorted(processes)})
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


def install_parent_observers(value_function: Any, benchmark: Any) -> None:
    original_compute = value_function.OptimizedValueFunctionSession.compute
    original_analyze_state = value_function.OptimizedValueFunctionSession.analyze_state
    original_query_to_json = benchmark.query_to_json
    original_batch = value_function.evaluate_query_candidates_optimized
    compute_signature = inspect.signature(original_compute)
    batch_signature = inspect.signature(original_batch)

    def observed_compute(session: Any, *arguments: Any, **keywords: Any) -> Any:
        global _CURRENT_PLANNING, _LATEST_SELECTED_PLAN
        bound = compute_signature.bind(session, *arguments, **keywords).arguments
        previous = _CURRENT_PLANNING
        _CURRENT_PLANNING = {"planning_id": next(_PLANNING_SEQUENCE),
            "question_index": len(bound["answered_queries"]) + 1,
            "planning_depth": int(bound["remaining_depth"])}
        started = time.monotonic()
        journal(_PROGRESS_PATH, "planning_start", **_CURRENT_PLANNING)
        try:
            result = original_compute(session, *arguments, **keywords)
            chosen = next((evaluation for evaluation in result.query_evaluations
                           if evaluation.query == result.best_query), None)
            _LATEST_SELECTED_PLAN = {
                "question_index": _CURRENT_PLANNING["question_index"],
                "best_query": query_signature(result.best_query),
                "expected_value": float(result.value),
                "lexicographic_expected_values": (
                    [float(x) for x in (chosen.lexicographic_expected_values
                                      or (float(chosen.expected_value),))]
                    if chosen is not None else None),
            }
            journal(_PROGRESS_PATH, "planning_end", **_CURRENT_PLANNING,
                duration_seconds=time.monotonic() - started,
                query_evaluation_count=len(result.query_evaluations),
                best_query=query_signature(result.best_query), value=float(result.value))
            return result
        except BaseException as error:
            journal(_PROGRESS_PATH, "planning_error", **_CURRENT_PLANNING,
                duration_seconds=time.monotonic() - started, error_type=type(error).__name__,
                error_message=str(error), traceback=traceback.format_exc())
            raise
        finally:
            _CURRENT_PLANNING = previous

    def observed_analyze_state(session: Any, answered_queries: Any,
                               *arguments: Any, **keywords: Any) -> Any:
        state = original_analyze_state(session, answered_queries, *arguments, **keywords)
        if _CHECKPOINT_PATH is not None and not _CURRENT_PLANNING:
            analysis = state.candidate_analysis
            checkpoint("state_boundary", answered_query_count=len(answered_queries),
                is_feasible=bool(state.is_feasible),
                candidates=list(analysis.candidates) if analysis is not None else None,
                candidate_volumes=analysis.candidate_volumes if analysis is not None else None,
                candidate_volume_shares=analysis.candidate_volume_shares if analysis is not None else None)
        return state

    def observed_query_to_json(query: Any, answer: str, seconds: float) -> Any:
        record = original_query_to_json(query, answer, seconds)
        checkpoint("answered_query", **_LATEST_SELECTED_PLAN, query_record=record)
        return record

    def observed_batch(*arguments: Any, **keywords: Any) -> Any:
        global _CURRENT_BATCH
        bound = batch_signature.bind(*arguments, **keywords).arguments
        if not bound.get("parallelize", False):
            return original_batch(*arguments, **keywords)
        previous = _CURRENT_BATCH
        _CURRENT_BATCH = {"batch_id": next(_BATCH_SEQUENCE),
            "remaining_depth": int(bound["remaining_depth"]),
            "root_query_count": len(bound["query_candidates"])}
        started = time.monotonic()
        journal(_PROGRESS_PATH, "parallel_batch_start", **{**_CURRENT_PLANNING, **_CURRENT_BATCH})
        try:
            result = original_batch(*arguments, **keywords)
            journal(_PROGRESS_PATH, "parallel_batch_end", **{**_CURRENT_PLANNING, **_CURRENT_BATCH},
                duration_seconds=time.monotonic() - started, completed_query_count=len(result))
            return result
        except BaseException as error:
            journal(_PROGRESS_PATH, "parallel_batch_error", **{**_CURRENT_PLANNING, **_CURRENT_BATCH},
                duration_seconds=time.monotonic() - started, error_type=type(error).__name__,
                error_message=str(error), traceback=traceback.format_exc())
            raise
        finally:
            _CURRENT_BATCH = previous

    value_function.ProcessPoolExecutor = ObservedProcessPoolExecutor
    value_function.OptimizedValueFunctionSession.compute = observed_compute
    value_function.OptimizedValueFunctionSession.analyze_state = observed_analyze_state
    value_function.evaluate_query_candidates_optimized = observed_batch
    benchmark.query_to_json = observed_query_to_json


def efficiency_plan_json(plan: Any) -> dict[str, Any]:
    evaluation = plan.evaluation
    return {"horizon": int(plan.horizon), "query": query_signature(evaluation.query),
            "query_source": str(evaluation.query_source),
            "expected_terminal_candidates": float(evaluation.expected_value),
            "expected_queries": float(plan.expected_queries),
            "gain": float(plan.gain), "rate": float(plan.rate)}


def install_efficiency_observers(value_function: Any) -> None:
    """Observe the new policy; diagnostic clocks never enter its decision."""
    from multistep.optimized import query_efficiency
    original_compute = value_function.OptimizedValueFunctionSession.compute_query_efficiency
    signature = inspect.signature(original_compute)
    original_batch = query_efficiency._evaluate_horizon_candidates
    batch_signature = inspect.signature(original_batch)

    def observed_compute(session: Any, *arguments: Any, **keywords: Any) -> Any:
        global _CURRENT_PLANNING, _LATEST_SELECTED_PLAN
        bound = signature.bind(session, *arguments, **keywords).arguments
        stop_aware = bool(session.config.query_efficiency_stop_aware)
        strategy = "reduction_per_query_stop_aware" if stop_aware else "reduction_per_query"
        previous = _CURRENT_PLANNING
        _CURRENT_PLANNING = {"planning_id": next(_PLANNING_SEQUENCE),
            "question_index": len(bound["answered_queries"]) + 1,
            "planning_depth": int(bound.get("max_horizon", 2)),
            "query_strategy": strategy}
        started = time.monotonic()
        journal(_PROGRESS_PATH, "planning_start", **_CURRENT_PLANNING)
        try:
            result = original_compute(session, *arguments, **keywords)
            choice = result.choice
            if choice is None:
                _LATEST_SELECTED_PLAN = {}
                journal(_PROGRESS_PATH, "planning_end", **_CURRENT_PLANNING,
                    duration_seconds=time.monotonic() - started, query_evaluation_count=0,
                    best_query=None, value=None, selected_horizon=None)
                return result
            chosen = choice.evaluation
            plan_json = query_efficiency.query_efficiency_evaluation_to_json if stop_aware else efficiency_plan_json
            best_plans = [plan_json(result.best_by_horizon[h])
                          for h in sorted(result.best_by_horizon)
                          if result.best_by_horizon[h] is not None]
            _LATEST_SELECTED_PLAN = {
                "question_index": _CURRENT_PLANNING["question_index"],
                "best_query": query_signature(chosen.query),
                "expected_value": float(chosen.expected_value),
                "lexicographic_expected_values": [float(x) for x in
                    (chosen.lexicographic_expected_values or (chosen.expected_value,))],
                "query_strategy": strategy, "selected_horizon": int(choice.horizon),
                "expected_queries": float(choice.expected_queries),
                "gain": float(choice.gain), "rate": float(choice.rate),
                "horizon_best_plans": best_plans,
                "horizon_evaluation_seconds": {str(h): float(s) for h, s in
                    result.evaluation_seconds_by_horizon.items()},
            }
            if stop_aware:
                _LATEST_SELECTED_PLAN.update(
                    expected_terminal_candidates=float(chosen.expected_value),
                    expected_terminal_loss=float(choice.expected_terminal_loss),
                    planning_stop_aware=True,
                    candidate_count_before_planning=int(result.candidate_count),
                )
            extra = ({"expected_terminal_candidates": float(chosen.expected_value),
                      "expected_terminal_loss": float(choice.expected_terminal_loss),
                      "planning_stop_aware": True,
                      "candidate_count_before_planning": int(result.candidate_count)} if stop_aware else {})
            journal(_PROGRESS_PATH, "planning_end", **_CURRENT_PLANNING,
                duration_seconds=time.monotonic() - started,
                query_evaluation_count=sum(len(v) for v in result.evaluations_by_horizon.values()),
                best_query=query_signature(chosen.query), value=float(chosen.expected_value),
                selected_horizon=int(choice.horizon), expected_queries=float(choice.expected_queries),
                gain=float(choice.gain), rate=float(choice.rate), horizon_best_plans=best_plans,
                horizon_evaluation_seconds=_LATEST_SELECTED_PLAN["horizon_evaluation_seconds"], **extra)
            return result
        except BaseException as error:
            journal(_PROGRESS_PATH, "planning_error", **_CURRENT_PLANNING,
                duration_seconds=time.monotonic() - started, error_type=type(error).__name__,
                error_message=str(error), traceback=traceback.format_exc())
            raise
        finally:
            _CURRENT_PLANNING = previous

    def observed_batch(*arguments: Any, **keywords: Any) -> Any:
        global _CURRENT_BATCH
        bound = batch_signature.bind(*arguments, **keywords).arguments
        strategy = ("reduction_per_query_stop_aware" if bound["config"].query_efficiency_stop_aware
                    else "reduction_per_query")
        previous = _CURRENT_BATCH
        _CURRENT_BATCH = {"batch_id": next(_BATCH_SEQUENCE),
            "remaining_depth": int(bound["horizon"]),
            "root_query_count": len(bound["query_candidates"]),
            "query_strategy": strategy}
        started = time.monotonic()
        journal(_PROGRESS_PATH, "parallel_batch_start", **{**_CURRENT_PLANNING, **_CURRENT_BATCH})
        try:
            result = original_batch(*arguments, **keywords)
            journal(_PROGRESS_PATH, "parallel_batch_end", **{**_CURRENT_PLANNING, **_CURRENT_BATCH},
                duration_seconds=time.monotonic() - started, completed_query_count=len(result))
            return result
        except BaseException as error:
            journal(_PROGRESS_PATH, "parallel_batch_error", **{**_CURRENT_PLANNING, **_CURRENT_BATCH},
                duration_seconds=time.monotonic() - started, error_type=type(error).__name__,
                error_message=str(error), traceback=traceback.format_exc())
            raise
        finally:
            _CURRENT_BATCH = previous

    value_function.OptimizedValueFunctionSession.compute_query_efficiency = observed_compute
    query_efficiency.ProcessPoolExecutor = ObservedProcessPoolExecutor
    query_efficiency._evaluate_horizon_candidates = observed_batch


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: stable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): stable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [stable(item) for item in value]
    return value


def fingerprint(value: Any) -> str:
    encoded = json.dumps(stable(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def save(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--problem-index", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--root-query-source", choices=["both", "grid"], required=True)
    parser.add_argument("--grid-size", type=int, choices=[5, 10], required=True)
    parser.add_argument("--depth", type=int, choices=[1, 2], required=True)
    parser.add_argument("--condition", choices=["fixed_depth1", "fixed_depth2", "adaptive_rate", "adaptive_stop_aware"], required=True)
    parser.add_argument("--query-strategy", choices=["fixed", "reduction_per_query", "reduction_per_query_stop_aware"], required=True)
    parser.add_argument("--progress-jsonl", type=Path, help="External root-task observation journal.")
    parser.add_argument("--checkpoint-jsonl", type=Path,
                        help="Fsynced actual-question and state boundaries; no inner-solver resume.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate imports, fixed case and configuration without solving.")
    return parser.parse_args()


def main() -> None:
    global _PROGRESS_PATH, _CHECKPOINT_PATH
    args = parse_args()
    expected = {"fixed_depth1": (1, "fixed"), "fixed_depth2": (2, "fixed"),
                "adaptive_rate": (2, "reduction_per_query"),
                "adaptive_stop_aware": (2, "reduction_per_query_stop_aware")}[args.condition]
    if (args.depth, args.query_strategy) != expected or args.root_query_source != "both" or args.grid_size != 5:
        raise ValueError("Condition/depth/query-strategy/grid policy differ from the frozen experiment")
    if args.progress_jsonl is not None:
        args.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        _PROGRESS_PATH = str(args.progress_jsonl.resolve())
    if args.checkpoint_jsonl is not None:
        args.checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)
        _CHECKPOINT_PATH = str(args.checkpoint_jsonl.resolve())
        if args.checkpoint_jsonl.exists() and not args.check_only:
            raise ValueError("Do not reuse an existing attempt's question checkpoint")
    report: dict[str, Any] = {
        "metadata": {
            "started_at_utc": now(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "repo_root": str(args.repo_root.resolve()),
            "source_label": args.source_label,
            "condition": args.condition,
            "source_revision": args.source_revision,
            "source_fingerprint": args.source_fingerprint,
            "progress_jsonl": _PROGRESS_PATH,
            "checkpoint_jsonl": _CHECKPOINT_PATH,
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "timing_semantics": (
                "result.seconds is the unchanged solve_problem elapsed time, including "
                "session and worker startup/shutdown. Imports, input parsing, report "
                "serialization and saving are excluded. External observers journal parent "
                "planning/batch boundaries and two events per submitted root worker task, and "
                "snapshot worker PIDs at ordinary pool shutdown. Original submitted callables "
                "and payloads are unchanged. Observation overhead is present in both depths, "
                "is not separately quantified, and can scale with task counts. No inner "
                "recursion profiling is installed. Aborted/timeout runs are not solved results."
            ),
        },
        "settings": {
            "dimension": args.dimension,
            "problem_index": args.problem_index,
            "alternatives": 5,
            "depth": args.depth,
            "query_strategy": args.query_strategy,
            "grid_size": args.grid_size,
            "root_query_source": args.root_query_source,
            "depth_one_query_source": "ratio",
            "workers": 8,
            "parallelize_root": True,
            "max_questions": 100,
            "volume_confidence_threshold": 0.99,
            "answer_probability_mode": "exact_volume",
            "progress_per_query": True,
        },
    }
    if args.condition == "adaptive_stop_aware":
        report["settings"].update(planning_stop_aware=True,
            planning_terminal_loss_semantics=STOP_AWARE_LOSS_SEMANTICS)
    try:
        if args.dimension < 3 or args.problem_index < 1:
            raise ValueError("Require dimension >= 3 and problem-index >= 1")
        repository = args.repo_root.resolve()
        if not (repository / "multistep/scripts/benchmark_exact_end_to_end.py").is_file():
            raise ValueError(f"Missing benchmark solver in {repository}")
        input_bytes = args.input_json.read_bytes()
        document = json.loads(input_bytes)
        matches = [case for case in document["cases"]
                   if case["dimension"] == args.dimension
                   and case["problem_index"] == args.problem_index]
        if len(matches) != 1:
            raise ValueError("Input must contain exactly one matching fixed case")
        case = matches[0]
        if (len(case["alternatives"]) != 5
                or any(len(row) != args.dimension for row in case["alternatives"])
                or len(case["target_weights"]) != args.dimension):
            raise ValueError("Fixed case dimensions do not match requested five-alternative case")
        report["case"] = case
        report["metadata"]["input_json"] = str(args.input_json.resolve())
        report["metadata"]["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
        sys.path.insert(0, str(repository))
        import numpy as np
        import scipy
        from multistep.optimized import value_function as vf
        from multistep.scripts import benchmark_exact_end_to_end as benchmark
        from multistep.src.models import AlternativenMatrix

        multiprocessing.set_start_method("forkserver")
        report["metadata"].update(
            numpy=np.__version__, scipy=scipy.__version__,
            multiprocessing_start_method=multiprocessing.get_start_method(),
            solver_file=str(Path(benchmark.__file__).resolve()),
        )
        stop_settings = {"query_efficiency_stop_aware": True} if args.condition == "adaptive_stop_aware" else {}
        config = benchmark.build_config(
            parallelize_root=True, max_workers=8, grid_size=args.grid_size,
            root_query_source=args.root_query_source, depth_one_query_source="ratio",
            volume_confidence_threshold=0.99,
            **stop_settings,
        )
        report["configuration"] = dataclasses.asdict(config)
        report["configuration_fingerprint"] = fingerprint(config)
        alternatives = AlternativenMatrix(entries=case["alternatives"])
        if args.check_only:
            print(json.dumps({"check": "ok", "metadata": report["metadata"],
                              "configuration": report["configuration"],
                              "configuration_fingerprint": report["configuration_fingerprint"]},
                             indent=2), flush=True)
            return
        install_parent_observers(vf, benchmark)
        install_efficiency_observers(vf)
        checkpoint("solve_start", depth=args.depth, problem_index=args.problem_index,
                   input_sha256=report["metadata"]["input_sha256"],
                   source_fingerprint=args.source_fingerprint, condition=args.condition, query_strategy=args.query_strategy)
        journal(_PROGRESS_PATH, "solve_start", depth=args.depth, problem_index=args.problem_index)
        result = benchmark.solve_problem(
            alternatives=alternatives,
            target_weights=case["target_weights"],
            max_questions=100, depth=args.depth, query_strategy=args.query_strategy,
            parallelize_root=True, max_workers=8, grid_size=args.grid_size,
            root_query_source=args.root_query_source, depth_one_query_source="ratio",
            progress_per_query=True, volume_confidence_threshold=0.99,
        )
        journal(_PROGRESS_PATH, "solve_end", depth=args.depth, problem_index=args.problem_index,
                seconds=result["seconds"], question_count=result["question_count"])
        report["result"] = result
        checkpoint("solve_end", seconds=result["seconds"], question_count=result["question_count"],
                   termination_reason=result["termination_reason"],
                   selected_candidate=result["selected_candidate"],
                   selected_candidate_volume_share=result["selected_candidate_volume_share"])
        report["root_tasks_submitted"] = _ROOT_TASKS_SUBMITTED
        report["metadata"]["finished_at_utc"] = now()
        report["pool_shutdowns"] = _POOL_SHUTDOWNS
        report["spawned_worker_pids"] = sorted({pid for entry in _POOL_SHUTDOWNS
                                                 for pid in entry["worker_pids"]})
        save(args.output_json, report)
        print(json.dumps({"source_label": args.source_label,
            "condition": args.condition,
                          "dimension": args.dimension,
                          "problem_index": args.problem_index,
                          "seconds": result["seconds"],
                          "question_count": result["question_count"],
                          "selection_is_correct": result["selection_is_correct"],
                          "spawned_worker_pids": report["spawned_worker_pids"]}), flush=True)
    except Exception as exc:
        report["metadata"]["finished_at_utc"] = now()
        report["pool_shutdowns"] = _POOL_SHUTDOWNS
        report["spawned_worker_pids"] = sorted({pid for entry in _POOL_SHUTDOWNS
                                                 for pid in entry["worker_pids"]})
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        checkpoint("solve_error", **report["error"])
        save(args.output_json, report)
        raise


if __name__ == "__main__":
    main()
