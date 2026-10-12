from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any

from scipy.optimize import linprog

from .reusable_highs import (
    clear_reusable_highs_cache,
    reusable_highs_cache_info,
    try_reusable_highs,
)

from .models.linear_optimization_result import OptimizationStatus


LINPROG_FEASIBILITY_TOLERANCE = 1e-10
LINPROG_OPTIONS = {
    "primal_feasibility_tolerance": LINPROG_FEASIBILITY_TOLERANCE,
    "dual_feasibility_tolerance": LINPROG_FEASIBILITY_TOLERANCE,
    "disp": False,
}
LINPROG_OPTIONS_WITHOUT_PRESOLVE = {
    **LINPROG_OPTIONS,
    "presolve": False,
}

_ORIGINAL_LINPROG = linprog


def _run_linprog(*, _internal_compact_result: bool = False, **kwargs: Any) -> Any:
    # Preserve explicit overrides of the public solver, including retry tests.
    if linprog is _ORIGINAL_LINPROG:
        result = try_reusable_highs(
            kwargs, compact_result=_internal_compact_result,
            _internal_continuous_check=_internal_compact_result,
        )
        if result is not None:
            return result
    with _suppress_native_solver_output():
        return linprog(**kwargs)


def classify_linprog_failure(
    solver_status_code: int,
    solver_message: str,
) -> OptimizationStatus | None:
    if solver_status_code == 2:
        return "infeasible"

    if solver_status_code == 3:
        return "unbounded"

    normalized_message = solver_message.lower()

    if (
        "model_status is infeasible" in normalized_message
        or "primal_status is infeasible" in normalized_message
    ):
        return "infeasible"

    if "model_status is unbounded" in normalized_message:
        return "unbounded"

    return None


def is_classified_linprog_result(result: Any) -> bool:
    if result.success:
        return True

    return (
        classify_linprog_failure(
            solver_status_code=int(result.status),
            solver_message=str(result.message),
        )
        is not None
    )


def run_linprog_with_retries(
    *,
    _internal_compact_result: bool = False,
    **kwargs: Any,
) -> Any:
    """Run the unchanged LP retry sequence.

    The private compact flag is used only by project callers that need primal
    values and status. It never reaches the public SciPy fallback; default
    callers receive the original complete OptimizeResult, including marginals.
    """
    result = _run_linprog(_internal_compact_result=_internal_compact_result, **kwargs)
    if result.success:
        return result

    classified_status = classify_linprog_failure(
        solver_status_code=int(result.status),
        solver_message=str(result.message),
    )

    # HiGHS presolve can occasionally report infeasible for homogeneous
    # ratio-normalized systems whose non-presolve model is feasible/unbounded.
    # Infeasible and unknown failures are therefore checked once more without
    # presolve before we accept the status.
    if classified_status != "unbounded":
        retry_kwargs = {
            **kwargs,
            "options": LINPROG_OPTIONS_WITHOUT_PRESOLVE,
        }
        retry_result = _run_linprog(
            _internal_compact_result=_internal_compact_result, **retry_kwargs,
        )
        if is_classified_linprog_result(retry_result):
            return retry_result

        # Rarely, the automatic HiGHS dispatcher and its no-presolve retry
        # both end in the native "Not Set" state for highly cut polytopes.
        # Try both concrete HiGHS algorithms before surfacing that transient
        # numerical failure to the caller.
        requested_method = str(kwargs.get("method", "highs"))
        for fallback_method in ("highs-ds", "highs-ipm"):
            if fallback_method == requested_method:
                continue
            fallback_kwargs = {
                **retry_kwargs,
                "method": fallback_method,
            }
            fallback_result = _run_linprog(
                _internal_compact_result=_internal_compact_result, **fallback_kwargs,
            )
            if is_classified_linprog_result(fallback_result):
                return fallback_result

    if classified_status is not None:
        return result

    return result


@contextmanager
def _suppress_native_solver_output() -> Any:
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)
