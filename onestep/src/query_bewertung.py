from __future__ import annotations

import math
from dataclasses import dataclass

from .build_ungleichungssysteme import build_W
from .io_models import AnsweredQuery, Query
from .ratio_intervalle import ZielpaarIntervalle
from .sampling import sample_points_from_ungleichungssystem


ZielpaarIntervalleLookup = dict[tuple[int, int], ZielpaarIntervalle]


@dataclass(frozen=True)
class QueryInfo:
    query: Query
    kandidatenmenge_nach_kleiner_antwort: set[int]
    kandidatenmenge_nach_groesser_antwort: set[int]
    p_kleiner: float
    p_groesser: float
    expected_kandidatenanzahl: float


def is_point_kleiner_als_query(point: list[float], query: Query) -> bool:
    _validate_point_matches_query(point, query)
    return point[query.ziel_index_a] < query.value * point[query.ziel_index_b]


def is_point_groesser_als_query(point: list[float], query: Query) -> bool:
    _validate_point_matches_query(point, query)
    return point[query.ziel_index_a] > query.value * point[query.ziel_index_b]


def estimate_query_probabilities(
    samples: list[list[float]],
    query: Query,
) -> tuple[float, float]:
    if not samples:
        raise ValueError("samples must not be empty")

    anzahl_kleiner = 0
    anzahl_groesser = 0

    for point in samples:
        if is_point_kleiner_als_query(point, query):
            anzahl_kleiner += 1
        elif is_point_groesser_als_query(point, query):
            anzahl_groesser += 1

    anzahl_beruecksichtigter_samples = anzahl_kleiner + anzahl_groesser
    if anzahl_beruecksichtigter_samples == 0:
        raise ValueError("all sampled points lie on the query boundary")

    p_kleiner = anzahl_kleiner / anzahl_beruecksichtigter_samples
    p_groesser = anzahl_groesser / anzahl_beruecksichtigter_samples
    return p_kleiner, p_groesser


def sample_points_from_W(
    anzahl_ziele: int,
    answered_queries: list[AnsweredQuery],
    num_samples: int = 1000,
    burn_in: int = 200,
    thinning: int = 5,
    seed: int | None = None,
) -> list[list[float]]:
    W = build_W(anzahl_ziele, answered_queries)
    return sample_points_from_ungleichungssystem(
        system=W,
        num_samples=num_samples,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed,
    )


def build_zielpaar_intervalle_lookup(
    zielpaar_intervalle_liste: list[ZielpaarIntervalle],
) -> ZielpaarIntervalleLookup:
    zielpaar_intervalle_lookup: ZielpaarIntervalleLookup = {}

    for zielpaar_intervalle in zielpaar_intervalle_liste:
        zielpaar = (zielpaar_intervalle.ziel_index_a, zielpaar_intervalle.ziel_index_b)
        if zielpaar in zielpaar_intervalle_lookup:
            raise ValueError("zielpaar_intervalle_liste must not contain duplicate goal pairs")
        zielpaar_intervalle_lookup[zielpaar] = zielpaar_intervalle

    return zielpaar_intervalle_lookup


def compute_kandidatenmengen_nach_query(
    query: Query,
    zielpaar_intervalle_lookup: ZielpaarIntervalleLookup,
) -> tuple[set[int], set[int]]:
    zielpaar_intervalle = _get_zielpaar_intervalle_for_query(
        query=query,
        zielpaar_intervalle_lookup=zielpaar_intervalle_lookup,
    )

    kandidatenmenge_nach_kleiner_antwort: set[int] = set()
    kandidatenmenge_nach_groesser_antwort: set[int] = set()

    for candidate_index, ratio_interval in zielpaar_intervalle.intervalle_pro_kandidat.items():
        lower = ratio_interval.lower
        upper = ratio_interval.upper

        if lower.status == "unbounded":
            raise ValueError("lower ratio bound must not be unbounded")

        if lower.status == "optimal":
            if lower.optimal_value is None:
                raise RuntimeError("optimal lower ratio bound has no optimal_value")

            if query.value > lower.optimal_value:
                kandidatenmenge_nach_kleiner_antwort.add(int(candidate_index))

        if upper.status == "optimal":
            if upper.optimal_value is None:
                raise RuntimeError("optimal upper ratio bound has no optimal_value")

            if query.value < upper.optimal_value:
                kandidatenmenge_nach_groesser_antwort.add(int(candidate_index))
        elif upper.status == "unbounded":
            kandidatenmenge_nach_groesser_antwort.add(int(candidate_index))

    return kandidatenmenge_nach_kleiner_antwort, kandidatenmenge_nach_groesser_antwort


def compute_anzahl_kandidaten_nach_query(
    query: Query,
    zielpaar_intervalle_lookup: ZielpaarIntervalleLookup,
) -> tuple[int, int]:
    (
        kandidatenmenge_nach_kleiner_antwort,
        kandidatenmenge_nach_groesser_antwort,
    ) = compute_kandidatenmengen_nach_query(
        query=query,
        zielpaar_intervalle_lookup=zielpaar_intervalle_lookup,
    )

    return (
        len(kandidatenmenge_nach_kleiner_antwort),
        len(kandidatenmenge_nach_groesser_antwort),
    )


def compute_expected_kandidatenanzahl(
    query: Query,
    samples: list[list[float]],
    zielpaar_intervalle_lookup: ZielpaarIntervalleLookup,
) -> float:
    query_info = compute_query_info(
        query=query,
        samples=samples,
        zielpaar_intervalle_lookup=zielpaar_intervalle_lookup,
    )
    if query_info is None:
        raise ValueError("all sampled points lie on the query boundary")

    return query_info.expected_kandidatenanzahl


def compute_query_info(
    query: Query,
    samples: list[list[float]],
    zielpaar_intervalle_lookup: ZielpaarIntervalleLookup,
) -> QueryInfo | None:
    try:
        p_kleiner, p_groesser = estimate_query_probabilities(samples, query)
    except ValueError as exc:
        if str(exc) == "all sampled points lie on the query boundary":
            return None
        raise

    (
        kandidatenmenge_nach_kleiner_antwort,
        kandidatenmenge_nach_groesser_antwort,
    ) = compute_kandidatenmengen_nach_query(
        query=query,
        zielpaar_intervalle_lookup=zielpaar_intervalle_lookup,
    )

    expected_kandidatenanzahl = (
        p_kleiner * len(kandidatenmenge_nach_kleiner_antwort)
        + p_groesser * len(kandidatenmenge_nach_groesser_antwort)
    )

    return QueryInfo(
        query=query,
        kandidatenmenge_nach_kleiner_antwort=kandidatenmenge_nach_kleiner_antwort,
        kandidatenmenge_nach_groesser_antwort=kandidatenmenge_nach_groesser_antwort,
        p_kleiner=p_kleiner,
        p_groesser=p_groesser,
        expected_kandidatenanzahl=expected_kandidatenanzahl,
    )


def filter_already_answered_queries(
    queries: list[Query],
    answered_queries: list[AnsweredQuery],
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-9,
) -> list[Query]:
    return [
        query
        for query in queries
        if not is_query_already_answered(
            query=query,
            answered_queries=answered_queries,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    ]


def is_query_already_answered(
    query: Query,
    answered_queries: list[AnsweredQuery],
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-9,
) -> bool:
    for answered_query in answered_queries:
        if (
            query.ziel_index_a == answered_query.ziel_index_a
            and query.ziel_index_b == answered_query.ziel_index_b
            and math.isclose(
                float(query.value),
                float(answered_query.value),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        ):
            return True

        if (
            query.value > 0.0
            and answered_query.value > 0.0
            and query.ziel_index_a == answered_query.ziel_index_b
            and query.ziel_index_b == answered_query.ziel_index_a
            and math.isclose(
                float(query.value),
                1.0 / float(answered_query.value),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        ):
            return True

    return False


def is_query_informative(
    query_info: QueryInfo,
    kandidatenmenge: set[int],
) -> bool:
    return (
        query_info.p_kleiner > 0.0
        and query_info.kandidatenmenge_nach_kleiner_antwort < kandidatenmenge
    ) or (
        query_info.p_groesser > 0.0
        and query_info.kandidatenmenge_nach_groesser_antwort < kandidatenmenge
    )


def filter_informative_query_infos(
    query_infos: list[QueryInfo],
    kandidatenmenge: set[int],
) -> list[QueryInfo]:
    return [
        query_info
        for query_info in query_infos
        if is_query_informative(query_info, kandidatenmenge)
    ]


def _validate_point_matches_query(point: list[float], query: Query) -> None:
    if query.ziel_index_a >= len(point):
        raise IndexError("query.ziel_index_a is out of range for the sampled point")

    if query.ziel_index_b >= len(point):
        raise IndexError("query.ziel_index_b is out of range for the sampled point")


def _get_zielpaar_intervalle_for_query(
    query: Query,
    zielpaar_intervalle_lookup: ZielpaarIntervalleLookup,
) -> ZielpaarIntervalle:
    zielpaar = (query.ziel_index_a, query.ziel_index_b)
    if zielpaar not in zielpaar_intervalle_lookup:
        raise ValueError("query has no matching goal-pair intervals")

    return zielpaar_intervalle_lookup[zielpaar]
