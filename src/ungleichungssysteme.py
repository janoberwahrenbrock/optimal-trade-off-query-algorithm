from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import linprog


Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]


class LinearOptimizationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["optimal", "infeasible", "unbounded"]
    objective_sense: Literal["min", "max"]
    solver_status_code: int | None = None
    solver_message: str | None = None
    optimal_value: float | None = None


class Ungleichungssystem(BaseModel):
    ungleichungen_linke_seite: Matrix = Field(default_factory=list)
    ungleichungen_rechte_seite: Vector = Field(default_factory=list)
    gleichungen_linke_seite: Matrix = Field(default_factory=list)
    gleichungen_rechte_seite: Vector = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Ungleichungssystem:
        if len(self.ungleichungen_linke_seite) != len(self.ungleichungen_rechte_seite):
            raise ValueError(
                "ungleichungen_linke_seite and ungleichungen_rechte_seite must have the same number of rows"
            )

        if len(self.gleichungen_linke_seite) != len(self.gleichungen_rechte_seite):
            raise ValueError(
                "gleichungen_linke_seite and gleichungen_rechte_seite must have the same number of rows"
            )

        n_variablen = self.anzahl_variablen

        for row in self.ungleichungen_linke_seite:
            if len(row) != n_variablen:
                raise ValueError(
                    "all rows in ungleichungen_linke_seite must have the same length"
                )

        for row in self.gleichungen_linke_seite:
            if len(row) != n_variablen:
                raise ValueError(
                    "all rows in gleichungen_linke_seite must have the same length"
                )

        return self

    @property
    def anzahl_variablen(self) -> int:
        if self.gleichungen_linke_seite:
            return len(self.gleichungen_linke_seite[0])
        if self.ungleichungen_linke_seite:
            return len(self.ungleichungen_linke_seite[0])
        return 0

    def add_ungleichung(self, linke_seite: Vector, rechte_seite: float) -> None:
        self._validate_linke_seite(linke_seite)
        self.ungleichungen_linke_seite.append(linke_seite.copy())
        self.ungleichungen_rechte_seite.append(rechte_seite)

    def add_gleichung(self, linke_seite: Vector, rechte_seite: float) -> None:
        self._validate_linke_seite(linke_seite)
        self.gleichungen_linke_seite.append(linke_seite.copy())
        self.gleichungen_rechte_seite.append(rechte_seite)

    def add_ungleichungssystem(self, other: Ungleichungssystem) -> None:
        if self.anzahl_variablen != 0 and other.anzahl_variablen != 0:
            if self.anzahl_variablen != other.anzahl_variablen:
                raise ValueError("other must have the same number of variables")

        for linke_seite, rechte_seite in zip(
            other.ungleichungen_linke_seite,
            other.ungleichungen_rechte_seite,
        ):
            self.add_ungleichung(linke_seite, rechte_seite)

        for linke_seite, rechte_seite in zip(
            other.gleichungen_linke_seite,
            other.gleichungen_rechte_seite,
        ):
            self.add_gleichung(linke_seite, rechte_seite)

    def minimize(self, zielfunktion: Vector) -> LinearOptimizationResult:
        self._validate_zielfunktion(zielfunktion)

        if self.anzahl_variablen <= 0:
            return LinearOptimizationResult(
                status="infeasible",
                objective_sense="min",
                solver_message="system has no variables",
            )

        result = linprog(
            c=zielfunktion,
            A_ub=self.ungleichungen_linke_seite or None,
            b_ub=self.ungleichungen_rechte_seite or None,
            A_eq=self.gleichungen_linke_seite or None,
            b_eq=self.gleichungen_rechte_seite or None,
            bounds=[(None, None)] * self.anzahl_variablen,
            method="highs",
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

        if result.status == 2:
            return LinearOptimizationResult(
                status="infeasible",
                objective_sense="min",
                solver_status_code=int(result.status),
                solver_message=result.message,
            )

        if result.status == 3:
            return LinearOptimizationResult(
                status="unbounded",
                objective_sense="min",
                solver_status_code=int(result.status),
                solver_message=result.message,
            )

        raise RuntimeError(f"linprog failed: {result.message}")

    def maximize(self, zielfunktion: Vector) -> LinearOptimizationResult:
        minimization_result = self.minimize([-value for value in zielfunktion])

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
        if self.anzahl_variablen <= 0:
            return False

        return self.minimize([0.0] * self.anzahl_variablen).status == "optimal"

    def find_feasible_point(self) -> Vector:
        if self.anzahl_variablen <= 0:
            raise ValueError("system has no variables")

        result = linprog(
            c=[0.0] * self.anzahl_variablen,
            A_ub=self.ungleichungen_linke_seite or None,
            b_ub=self.ungleichungen_rechte_seite or None,
            A_eq=self.gleichungen_linke_seite or None,
            b_eq=self.gleichungen_rechte_seite or None,
            bounds=[(None, None)] * self.anzahl_variablen,
            method="highs",
        )

        if result.success:
            if result.x is None:
                raise RuntimeError("linprog returned success without a solution")

            return result.x.astype(float).tolist()

        if result.status == 2:
            raise ValueError("system is infeasible")

        if result.status == 3:
            raise ValueError("system is unbounded")

        raise RuntimeError(f"linprog failed: {result.message}")

    def to_latex(self, variablenname: str = "w") -> str:
        zeilen: list[str] = []

        for linke_seite, rechte_seite in zip(
            self.ungleichungen_linke_seite,
            self.ungleichungen_rechte_seite,
        ):
            zeilen.append(
                f"{self._format_linke_seite_latex(linke_seite, variablenname)}"
                f" \\le {self._format_zahl_latex(rechte_seite)}"
            )

        for linke_seite, rechte_seite in zip(
            self.gleichungen_linke_seite,
            self.gleichungen_rechte_seite,
        ):
            zeilen.append(
                f"{self._format_linke_seite_latex(linke_seite, variablenname)}"
                f" = {self._format_zahl_latex(rechte_seite)}"
            )

        if not zeilen:
            return r"\left\{\begin{aligned}0 = 0\end{aligned}\right."

        return r"\left\{\begin{aligned}" + r" \\ ".join(zeilen) + r"\end{aligned}\right."

    def _validate_linke_seite(self, linke_seite: Vector) -> None:
        if not linke_seite:
            raise ValueError("linke_seite must not be empty")

        if self.anzahl_variablen == 0:
            return

        if len(linke_seite) != self.anzahl_variablen:
            raise ValueError("linke_seite must match the number of variables")

    def _validate_zielfunktion(self, zielfunktion: Vector) -> None:
        if self.anzahl_variablen <= 0:
            return

        if len(zielfunktion) != self.anzahl_variablen:
            raise ValueError("zielfunktion must match the number of variables")

    def _format_linke_seite_latex(self, linke_seite: Vector, variablenname: str) -> str:
        terme: list[str] = []

        for index, koeffizient in enumerate(linke_seite, start=1):
            if koeffizient == 0:
                continue

            variablen_term = rf"{variablenname}_{{{index}}}"
            betrag = abs(koeffizient)

            if betrag == 1:
                term_ohne_vorzeichen = variablen_term
            else:
                term_ohne_vorzeichen = f"{self._format_zahl_latex(betrag)} {variablen_term}"

            if not terme:
                if koeffizient < 0:
                    terme.append(f"-{term_ohne_vorzeichen}")
                else:
                    terme.append(term_ohne_vorzeichen)
                continue

            if koeffizient < 0:
                terme.append(f"- {term_ohne_vorzeichen}")
            else:
                terme.append(f"+ {term_ohne_vorzeichen}")

        if not terme:
            return "0"

        return " ".join(terme)

    def _format_zahl_latex(self, value: float) -> str:
        return f"{value:g}"
