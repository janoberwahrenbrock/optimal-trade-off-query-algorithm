from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from annotated_types import Ge
from pydantic import BaseModel, ConfigDict, FiniteFloat, NonNegativeInt, model_validator


QueryOperator: TypeAlias = Literal["<", ">", "="]
NonNegativeFiniteFloat: TypeAlias = Annotated[FiniteFloat, Ge(0)]


class AnsweredQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    ziel_index_a: NonNegativeInt
    ziel_index_b: NonNegativeInt
    value: NonNegativeFiniteFloat
    operator: QueryOperator

    @model_validator(mode="after")
    def validate_ziel_indices(self) -> AnsweredQuery:
        if self.ziel_index_a == self.ziel_index_b:
            raise ValueError("ziel_index_a and ziel_index_b must be different")

        return self

    def __str__(self) -> str:
        return (
            f"Ziel[{self.ziel_index_a}] {self.operator} "
            f"{self.value:g} * Ziel[{self.ziel_index_b}]"
        )
