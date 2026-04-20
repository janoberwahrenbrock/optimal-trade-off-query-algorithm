from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, FiniteFloat, NonNegativeInt, model_validator


Alternative: TypeAlias = list[float]
Matrix: TypeAlias = list[Alternative]
QueryOperator: TypeAlias = Literal["<", ">", "="]
TerminationReason: TypeAlias = Literal[
    "one_remaining_candidate",
    "same_utility_values",
    "no_informative_query",
]
SharePrecision: TypeAlias = Literal["exact", "estimated"]
NonNegativeFiniteFloat: TypeAlias = Annotated[FiniteFloat, Ge(0)]
UnitFiniteFloat: TypeAlias = Annotated[FiniteFloat, Ge(0), Le(1)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class AlternativenMatrix(_FrozenModel):
    entries: Matrix

    @model_validator(mode="after")
    def validate_entries(self) -> AlternativenMatrix:
        if not self.entries:
            raise ValueError("entries must not be empty")

        column_count = len(self.entries[0])
        if column_count == 0:
            raise ValueError("entries must contain at least one goal column")

        for row_index, row in enumerate(self.entries):
            if len(row) != column_count:
                raise ValueError("all rows in entries must have the same length")
            for column_index, value in enumerate(row):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"entries[{row_index}][{column_index}] must be between 0 and 1"
                    )

        return self

    def get_anzahl_zeilen(self) -> int:
        return len(self.entries)

    def get_anzahl_spalten(self) -> int:
        return len(self.entries[0])

    def get_alternative(self, index: int) -> Alternative:
        if not 0 <= index < len(self.entries):
            raise IndexError("index is out of range")
        return self.entries[index].copy()


class AnsweredQuery(_FrozenModel):
    ziel_index_a: NonNegativeInt
    ziel_index_b: NonNegativeInt
    value: NonNegativeFiniteFloat
    operator: QueryOperator

    def __str__(self) -> str:
        return (
            f"Ziel[{self.ziel_index_a}] {self.operator} "
            f"{self.value:g} * Ziel[{self.ziel_index_b}]"
        )


class TerminationResult(_FrozenModel):
    reason: TerminationReason
    share_precision: SharePrecision
    remaining_candidates: list[NonNegativeInt]
    optimality_shares: dict[NonNegativeInt, UnitFiniteFloat]

    @model_validator(mode="after")
    def validate_termination_result(self) -> TerminationResult:
        if not self.remaining_candidates:
            raise ValueError("remaining_candidates must not be empty")

        if len(set(self.remaining_candidates)) != len(self.remaining_candidates):
            raise ValueError("remaining_candidates must not contain duplicates")

        remaining_candidates_set = set(self.remaining_candidates)
        optimality_share_keys = set(self.optimality_shares.keys())
        if optimality_share_keys != remaining_candidates_set:
            raise ValueError(
                "optimality_shares keys must match remaining_candidates exactly"
            )

        return self


class Query(_FrozenModel):
    ziel_index_a: NonNegativeInt
    ziel_index_b: NonNegativeInt
    value: NonNegativeFiniteFloat

    def answer(self, operator: QueryOperator) -> AnsweredQuery:
        return AnsweredQuery(
            ziel_index_a=self.ziel_index_a,
            ziel_index_b=self.ziel_index_b,
            value=self.value,
            operator=operator,
        )

    def __str__(self) -> str:
        return f"Ziel[{self.ziel_index_a}] ? {self.value:g} * Ziel[{self.ziel_index_b}]"
