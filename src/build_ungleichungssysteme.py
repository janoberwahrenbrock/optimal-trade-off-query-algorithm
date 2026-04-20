from __future__ import annotations

from .io_models import AlternativenMatrix, AnsweredQuery
from .ungleichungssysteme import Ungleichungssystem, Vector


def build_W(anzahl_ziele: int, answered_queries: list[AnsweredQuery]) -> Ungleichungssystem:
    if anzahl_ziele <= 0:
        raise ValueError("anzahl_ziele must be positive")

    system = Ungleichungssystem()
    _add_nichtnegativitaetsbedingungen(system, anzahl_ziele)
    system.add_gleichung([1.0] * anzahl_ziele, 1.0)
    _add_answered_query_nebenbedingungen(system, answered_queries, anzahl_ziele)
    return system


def build_optimal_region_in_W(
    alternativen_matrix: AlternativenMatrix,
    W: Ungleichungssystem,
    alternative_index: int,
) -> Ungleichungssystem:
    if not 0 <= alternative_index < alternativen_matrix.get_anzahl_zeilen():
        raise IndexError("alternative_index is out of range")

    anzahl_ziele = alternativen_matrix.get_anzahl_spalten()
    if W.anzahl_variablen not in {0, anzahl_ziele}:
        raise ValueError("W must have the same number of variables as the number of goals")

    optimal_region = Ungleichungssystem()
    optimal_region.add_ungleichungssystem(W)
    _add_optimalitaetsbedingungen(optimal_region, alternativen_matrix, alternative_index)
    return optimal_region


def build_normalized_optimal_region(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    alternative_index: int,
    normierungs_ziel_index: int,
) -> Ungleichungssystem:
    anzahl_ziele = alternativen_matrix.get_anzahl_spalten()

    if not 0 <= alternative_index < alternativen_matrix.get_anzahl_zeilen():
        raise IndexError("alternative_index is out of range")

    if not 0 <= normierungs_ziel_index < anzahl_ziele:
        raise IndexError("normierungs_ziel_index is out of range")

    normalized_region = Ungleichungssystem()
    _add_nichtnegativitaetsbedingungen(normalized_region, anzahl_ziele)

    normierungs_zeile = [0.0] * anzahl_ziele
    normierungs_zeile[normierungs_ziel_index] = 1.0
    normalized_region.add_gleichung(normierungs_zeile, 1.0)

    _add_answered_query_nebenbedingungen(
        normalized_region,
        answered_queries,
        anzahl_ziele,
    )
    _add_optimalitaetsbedingungen(
        normalized_region,
        alternativen_matrix,
        alternative_index,
    )
    return normalized_region


def build_utility_difference_zielfunktion(
    alternativen_matrix: AlternativenMatrix,
    alternative_index_a: int,
    alternative_index_b: int,
) -> Vector:
    if not 0 <= alternative_index_a < alternativen_matrix.get_anzahl_zeilen():
        raise IndexError("alternative_index_a is out of range")

    if not 0 <= alternative_index_b < alternativen_matrix.get_anzahl_zeilen():
        raise IndexError("alternative_index_b is out of range")

    alternative_a = alternativen_matrix.get_alternative(alternative_index_a)
    alternative_b = alternativen_matrix.get_alternative(alternative_index_b)
    return [
        nutzenwert_a - nutzenwert_b
        for nutzenwert_a, nutzenwert_b in zip(alternative_a, alternative_b)
    ]


def _add_nichtnegativitaetsbedingungen(
    system: Ungleichungssystem,
    anzahl_ziele: int,
) -> None:
    for ziel_index in range(anzahl_ziele):
        linke_seite = [0.0] * anzahl_ziele
        linke_seite[ziel_index] = -1.0
        system.add_ungleichung(linke_seite, 0.0)


def _add_answered_query_nebenbedingungen(
    system: Ungleichungssystem,
    answered_queries: list[AnsweredQuery],
    anzahl_ziele: int,
) -> None:
    for answered_query in answered_queries:
        _validate_answered_query(answered_query, anzahl_ziele)

        linke_seite, rechte_seite, ist_gleichung = _build_query_nebenbedingung(
            answered_query,
            anzahl_ziele,
        )

        if ist_gleichung:
            system.add_gleichung(linke_seite, rechte_seite)
        else:
            system.add_ungleichung(linke_seite, rechte_seite)


def _add_optimalitaetsbedingungen(
    system: Ungleichungssystem,
    alternativen_matrix: AlternativenMatrix,
    alternative_index: int,
) -> None:
    ziel_alternative = alternativen_matrix.get_alternative(alternative_index)

    for andere_alternative_index in range(alternativen_matrix.get_anzahl_zeilen()):
        if andere_alternative_index == alternative_index:
            continue

        andere_alternative = alternativen_matrix.get_alternative(andere_alternative_index)
        linke_seite = [
            anderer_nutzenwert - ziel_nutzenwert
            for anderer_nutzenwert, ziel_nutzenwert in zip(
                andere_alternative,
                ziel_alternative,
            )
        ]
        system.add_ungleichung(linke_seite, 0.0)


def _validate_answered_query(answered_query: AnsweredQuery, anzahl_ziele: int) -> None:
    if answered_query.ziel_index_a >= anzahl_ziele:
        raise IndexError("answered_query.ziel_index_a is out of range")

    if answered_query.ziel_index_b >= anzahl_ziele:
        raise IndexError("answered_query.ziel_index_b is out of range")

    if answered_query.ziel_index_a == answered_query.ziel_index_b:
        raise ValueError("answered_query must compare two different goals")


def _build_query_nebenbedingung(
    answered_query: AnsweredQuery,
    anzahl_ziele: int,
) -> tuple[Vector, float, bool]:
    linke_seite = [0.0] * anzahl_ziele
    ziel_index_a = answered_query.ziel_index_a
    ziel_index_b = answered_query.ziel_index_b
    value = answered_query.value

    if answered_query.operator == ">":
        # w_a - value * w_b >= 0  <=>  -w_a + value * w_b <= 0
        linke_seite[ziel_index_a] = -1.0
        linke_seite[ziel_index_b] = value
        return linke_seite, 0.0, False

    if answered_query.operator == "<":
        # w_a - value * w_b <= 0
        linke_seite[ziel_index_a] = 1.0
        linke_seite[ziel_index_b] = -value
        return linke_seite, 0.0, False

    if answered_query.operator == "=":
        # w_a - value * w_b = 0
        linke_seite[ziel_index_a] = 1.0
        linke_seite[ziel_index_b] = -value
        return linke_seite, 0.0, True

    raise ValueError(f"unknown operator: {answered_query.operator}")
