from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator


Alternative: TypeAlias = list[float]
Matrix: TypeAlias = list[Alternative]


class AlternativenMatrix(BaseModel):
    model_config = ConfigDict(frozen=True)

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
