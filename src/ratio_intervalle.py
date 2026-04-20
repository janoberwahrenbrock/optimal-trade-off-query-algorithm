from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from .build_ungleichungssysteme import build_normalized_optimal_region
from .io_models import AlternativenMatrix, AnsweredQuery
from .ungleichungssysteme import LinearOptimizationResult


class RatioInterval(BaseModel):
    model_config = ConfigDict(frozen=True)

    lower: LinearOptimizationResult
    upper: LinearOptimizationResult


class ZielpaarIntervalle(BaseModel):
    model_config = ConfigDict(frozen=True)

    ziel_index_a: NonNegativeInt
    ziel_index_b: NonNegativeInt
    intervalle_pro_kandidat: dict[NonNegativeInt, RatioInterval]

    @model_validator(mode="after")
    def validate_ziel_indices(self) -> ZielpaarIntervalle:
        if self.ziel_index_a == self.ziel_index_b:
            raise ValueError("ziel_index_a and ziel_index_b must be different")
        return self


def get_zielpaare(anzahl_ziele: int) -> list[tuple[int, int]]:
    if anzahl_ziele < 0:
        raise ValueError("anzahl_ziele must not be negative")

    return [
        (ziel_index_a, ziel_index_b)
        for ziel_index_a in range(anzahl_ziele)
        for ziel_index_b in range(anzahl_ziele)
        if ziel_index_a != ziel_index_b
    ]


def compute_ratio_interval(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    alternative_index: int,
    ziel_index_a: int,
    ziel_index_b: int,
) -> RatioInterval:
    anzahl_ziele = alternativen_matrix.get_anzahl_spalten()

    if not 0 <= ziel_index_a < anzahl_ziele:
        raise IndexError("ziel_index_a is out of range")

    if not 0 <= ziel_index_b < anzahl_ziele:
        raise IndexError("ziel_index_b is out of range")

    if ziel_index_a == ziel_index_b:
        raise ValueError("ziel_index_a and ziel_index_b must be different")

    normalized_region = build_normalized_optimal_region(
        alternativen_matrix=alternativen_matrix,
        answered_queries=answered_queries,
        alternative_index=alternative_index,
        normierungs_ziel_index=ziel_index_b,
    )

    zielfunktion = [0.0] * anzahl_ziele
    zielfunktion[ziel_index_a] = 1.0

    return RatioInterval(
        lower=normalized_region.minimize(zielfunktion),
        upper=normalized_region.maximize(zielfunktion),
    )


def compute_ratio_intervals_for_pair(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    kandidatenmenge: list[int],
    ziel_index_a: int,
    ziel_index_b: int,
) -> ZielpaarIntervalle:
    intervalle_pro_kandidat: dict[int, RatioInterval] = {}

    for candidate_index in kandidatenmenge:
        intervalle_pro_kandidat[candidate_index] = compute_ratio_interval(
            alternativen_matrix=alternativen_matrix,
            answered_queries=answered_queries,
            alternative_index=candidate_index,
            ziel_index_a=ziel_index_a,
            ziel_index_b=ziel_index_b,
        )

    return ZielpaarIntervalle(
        ziel_index_a=ziel_index_a,
        ziel_index_b=ziel_index_b,
        intervalle_pro_kandidat=intervalle_pro_kandidat,
    )


def compute_all_ratio_intervals(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    kandidatenmenge: list[int],
) -> list[ZielpaarIntervalle]:
    zielpaar_intervalle: list[ZielpaarIntervalle] = []

    for ziel_index_a, ziel_index_b in get_zielpaare(alternativen_matrix.get_anzahl_spalten()):
        zielpaar_intervalle.append(
            compute_ratio_intervals_for_pair(
                alternativen_matrix=alternativen_matrix,
                answered_queries=answered_queries,
                kandidatenmenge=kandidatenmenge,
                ziel_index_a=ziel_index_a,
                ziel_index_b=ziel_index_b,
            )
        )

    return zielpaar_intervalle
