from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict


OptimizationStatus: TypeAlias = Literal["optimal", "infeasible", "unbounded"]
ObjectiveSense: TypeAlias = Literal["min", "max"]


class LinearOptimizationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: OptimizationStatus
    objective_sense: ObjectiveSense
    solver_status_code: int | None = None
    solver_message: str | None = None
    optimal_value: float | None = None
