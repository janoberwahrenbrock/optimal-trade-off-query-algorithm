from __future__ import annotations

from typing import Any

from scipy.optimize import linprog

from .models.linear_optimization_result import OptimizationStatus


LINPROG_FEASIBILITY_TOLERANCE = 1e-10
LINPROG_OPTIONS = {
    "primal_feasibility_tolerance": LINPROG_FEASIBILITY_TOLERANCE,
    "dual_feasibility_tolerance": LINPROG_FEASIBILITY_TOLERANCE,
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
    result = linprog(**kwargs)
    if is_classified_linprog_result(result):
        return result

    retry_kwargs = {
        **kwargs,
        "options": LINPROG_OPTIONS_WITHOUT_PRESOLVE,
    }
    retry_result = linprog(**retry_kwargs)
    if is_classified_linprog_result(retry_result):
        return retry_result

    return result
