from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from .answered_query import AnsweredQuery, NonNegativeFiniteFloat, QueryOperator


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    ziel_index_a: NonNegativeInt
    ziel_index_b: NonNegativeInt
    value: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_ziel_indices(self) -> Query:
        if self.ziel_index_a == self.ziel_index_b:
            raise ValueError("ziel_index_a and ziel_index_b must be different")

        return self

    def answer(self, operator: QueryOperator) -> AnsweredQuery:
        return AnsweredQuery(
            ziel_index_a=self.ziel_index_a,
            ziel_index_b=self.ziel_index_b,
            value=self.value,
            operator=operator,
        )

    def __str__(self) -> str:
        return f"Ziel[{self.ziel_index_a}] ? {self.value:g} * Ziel[{self.ziel_index_b}]"
