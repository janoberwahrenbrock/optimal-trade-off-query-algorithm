from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any

from scipy.optimize import linprog

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


def run_linprog_with_retries(**kwargs: Any) -> Any:
    with _suppress_native_solver_output():
        result = linprog(**kwargs)
    if is_classified_linprog_result(result):
        return result

    retry_kwargs = {
        **kwargs,
        "options": LINPROG_OPTIONS_WITHOUT_PRESOLVE,
    }
    with _suppress_native_solver_output():
        retry_result = linprog(**retry_kwargs)
    if is_classified_linprog_result(retry_result):
        return retry_result

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
