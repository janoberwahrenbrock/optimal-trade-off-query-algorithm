from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, Field, model_validator

from .linear_programming import (
    LINPROG_OPTIONS,
    classify_linprog_failure,
    run_linprog_with_retries,
)
from .models.linear_optimization_result import LinearOptimizationResult


Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]


class LinearConstraintSystem(BaseModel):
    inequalities_left_side: Matrix = Field(default_factory=list)
    inequalities_right_side: Vector = Field(default_factory=list)
    equalities_left_side: Matrix = Field(default_factory=list)
    equalities_right_side: Vector = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dimensions(self) -> LinearConstraintSystem:
        if len(self.inequalities_left_side) != len(self.inequalities_right_side):
            raise ValueError(
                "inequalities_left_side and inequalities_right_side must have the same number of rows"
            )

        if len(self.equalities_left_side) != len(self.equalities_right_side):
            raise ValueError(
                "equalities_left_side and equalities_right_side must have the same number of rows"
            )

        variable_count = self.variable_count

        for row in self.inequalities_left_side:
            if len(row) != variable_count:
                raise ValueError(
                    "all rows in inequalities_left_side must have the same length"
                )

        for row in self.equalities_left_side:
            if len(row) != variable_count:
                raise ValueError(
                    "all rows in equalities_left_side must have the same length"
                )

        return self

    @property
    def variable_count(self) -> int:
        if self.equalities_left_side:
            return len(self.equalities_left_side[0])
        if self.inequalities_left_side:
            return len(self.inequalities_left_side[0])
        return 0

    def add_inequality(self, left_side: Vector, right_side: float) -> None:
        self._validate_left_side(left_side)
        self.inequalities_left_side.append(left_side.copy())
        self.inequalities_right_side.append(right_side)

    def add_equality(self, left_side: Vector, right_side: float) -> None:
        self._validate_left_side(left_side)
        self.equalities_left_side.append(left_side.copy())
        self.equalities_right_side.append(right_side)

    def add_constraint_system(self, other: LinearConstraintSystem) -> None:
        if self.variable_count != 0 and other.variable_count != 0:
            if self.variable_count != other.variable_count:
                raise ValueError("other must have the same number of variables")

        for left_side, right_side in zip(
            other.inequalities_left_side,
            other.inequalities_right_side,
        ):
            self.add_inequality(left_side, right_side)

        for left_side, right_side in zip(
            other.equalities_left_side,
            other.equalities_right_side,
        ):
            self.add_equality(left_side, right_side)

    def minimize(self, objective: Vector) -> LinearOptimizationResult:
        self._validate_objective(objective)

        if self.variable_count <= 0:
            return LinearOptimizationResult(
                status="infeasible",
                objective_sense="min",
                solver_message="system has no variables",
            )

        result = run_linprog_with_retries(
            c=objective,
            A_ub=self.inequalities_left_side or None,
            b_ub=self.inequalities_right_side or None,
            A_eq=self.equalities_left_side or None,
            b_eq=self.equalities_right_side or None,
            bounds=[(None, None)] * self.variable_count,
            method="highs",
            options=LINPROG_OPTIONS,
        )

        if result.success:
            if result.x is None or result.fun is None:
                raise RuntimeError("linprog returned success without a solution")

            return LinearOptimizationResult(
                status="optimal",
                objective_sense="min",
                solver_status_code=int(result.status),
                solver_message=result.message,
                optimal_value=float(result.fun),
            )

        solver_status_code = int(result.status)
        classified_status = classify_linprog_failure(
            solver_status_code=solver_status_code,
            solver_message=str(result.message),
        )
        if classified_status is not None:
            return LinearOptimizationResult(
                status=classified_status,
                objective_sense="min",
                solver_status_code=solver_status_code,
                solver_message=result.message,
            )

        raise RuntimeError(f"linprog failed: {result.message}")

    def maximize(self, objective: Vector) -> LinearOptimizationResult:
        minimization_result = self.minimize([-value for value in objective])

        if minimization_result.status != "optimal":
            return LinearOptimizationResult(
                status=minimization_result.status,
                objective_sense="max",
                solver_status_code=minimization_result.solver_status_code,
                solver_message=minimization_result.solver_message,
            )

        if minimization_result.optimal_value is None:
            raise RuntimeError("optimal minimization result has no optimal_value")

        return LinearOptimizationResult(
            status="optimal",
            objective_sense="max",
            solver_status_code=minimization_result.solver_status_code,
            solver_message=minimization_result.solver_message,
            optimal_value=-minimization_result.optimal_value,
        )

    def is_feasible(self) -> bool:
        if self.variable_count <= 0:
            return False

        return self.minimize([0.0] * self.variable_count).status == "optimal"

    def find_feasible_point(self) -> Vector:
        if self.variable_count <= 0:
            raise ValueError("system has no variables")

        result = run_linprog_with_retries(
            c=[0.0] * self.variable_count,
            A_ub=self.inequalities_left_side or None,
            b_ub=self.inequalities_right_side or None,
            A_eq=self.equalities_left_side or None,
            b_eq=self.equalities_right_side or None,
            bounds=[(None, None)] * self.variable_count,
            method="highs",
            options=LINPROG_OPTIONS,
        )

        if result.success:
            if result.x is None:
                raise RuntimeError("linprog returned success without a solution")

            return result.x.astype(float).tolist()

        classified_status = classify_linprog_failure(
            solver_status_code=int(result.status),
            solver_message=str(result.message),
        )
        if classified_status == "infeasible":
            raise ValueError("system is infeasible")

        if classified_status == "unbounded":
            raise ValueError("system is unbounded")

        raise RuntimeError(f"linprog failed: {result.message}")

    def to_latex(self, variable_name: str = "w") -> str:
        rows: list[str] = []

        for left_side, right_side in zip(
            self.inequalities_left_side,
            self.inequalities_right_side,
        ):
            rows.append(
                f"{self._format_left_side_latex(left_side, variable_name)}"
                f" \\le {self._format_number_latex(right_side)}"
            )

        for left_side, right_side in zip(
            self.equalities_left_side,
            self.equalities_right_side,
        ):
            rows.append(
                f"{self._format_left_side_latex(left_side, variable_name)}"
                f" = {self._format_number_latex(right_side)}"
            )

        if not rows:
            return r"\left\{\begin{aligned}0 = 0\end{aligned}\right."

        return r"\left\{\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}\right."

    def _validate_left_side(self, left_side: Vector) -> None:
        if not left_side:
            raise ValueError("left_side must not be empty")

        if self.variable_count == 0:
            return

        if len(left_side) != self.variable_count:
            raise ValueError("left_side must match the number of variables")

    def _validate_objective(self, objective: Vector) -> None:
        if self.variable_count <= 0:
            return

        if len(objective) != self.variable_count:
            raise ValueError("objective must match the number of variables")

    def _format_left_side_latex(self, left_side: Vector, variable_name: str) -> str:
        terms: list[str] = []

        for index, coefficient in enumerate(left_side, start=1):
            if coefficient == 0:
                continue

            variable_term = rf"{variable_name}_{{{index}}}"
            absolute_value = abs(coefficient)

            if absolute_value == 1:
                term_without_sign = variable_term
            else:
                term_without_sign = (
                    f"{self._format_number_latex(absolute_value)} {variable_term}"
                )

            if not terms:
                if coefficient < 0:
                    terms.append(f"-{term_without_sign}")
                else:
                    terms.append(term_without_sign)
                continue

            if coefficient < 0:
                terms.append(f"- {term_without_sign}")
            else:
                terms.append(f"+ {term_without_sign}")

        if not terms:
            return "0"

        return " ".join(terms)

    def _format_number_latex(self, value: float) -> str:
        return f"{value:g}"
