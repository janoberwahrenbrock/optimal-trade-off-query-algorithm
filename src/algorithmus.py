from __future__ import annotations

from typing import TypeAlias

from .build_ungleichungssysteme import build_W, build_optimal_region_in_W
from .io_models import (
    AnsweredQuery,
    AlternativenMatrix,
    Query,
    TerminationResult,
)
from .query_bewertung import (
    build_zielpaar_intervalle_lookup,
    compute_query_info,
    filter_already_answered_queries,
    filter_informative_query_infos,
)
from .query_kandidaten import compute_all_query_kandidaten
from .ratio_intervalle import compute_all_ratio_intervals
from .sampling import estimate_optimality_shares, sample_points_from_ungleichungssystem
from .termination import (
    all_candidates_have_same_utility_values_in_W,
    build_no_informative_query_termination_result,
    build_one_remaining_candidate_termination_result,
    build_same_utility_termination_result,
)


AlgorithmusOutput: TypeAlias = Query | TerminationResult


def run_algorithmus(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    seed: int | None = None,
) -> AlgorithmusOutput:
    anzahl_ziele = alternativen_matrix.get_anzahl_spalten()
    W = build_W(anzahl_ziele, answered_queries)

    if not W.is_feasible():
        raise ValueError("W is infeasible")

    kandidatenmenge: list[int] = []
    for alternative_index in range(alternativen_matrix.get_anzahl_zeilen()):
        optimal_region = build_optimal_region_in_W(
            alternativen_matrix,
            W,
            alternative_index,
        )
        if optimal_region.is_feasible():
            kandidatenmenge.append(alternative_index)

    if not kandidatenmenge:
        raise ValueError("no candidates found in W")

    if len(kandidatenmenge) == 1:
        return build_one_remaining_candidate_termination_result(kandidatenmenge[0])

    if all_candidates_have_same_utility_values_in_W(
        alternativen_matrix=alternativen_matrix,
        W=W,
        kandidatenmenge=kandidatenmenge,
    ):
        return build_same_utility_termination_result(kandidatenmenge)

    zielpaar_intervalle = compute_all_ratio_intervals(
        alternativen_matrix=alternativen_matrix,
        answered_queries=answered_queries,
        kandidatenmenge=kandidatenmenge,
    )
    query_kandidaten = compute_all_query_kandidaten(zielpaar_intervalle)
    query_kandidaten = filter_already_answered_queries(query_kandidaten, answered_queries)

    samples = sample_points_from_ungleichungssystem(
        W,
        num_samples=1000,
        burn_in=200,
        thinning=5,
        seed=seed,
    )
    zielpaar_intervalle_lookup = build_zielpaar_intervalle_lookup(zielpaar_intervalle)

    query_infos = []

    for query in query_kandidaten:
        query_info = compute_query_info(
            query=query,
            samples=samples,
            zielpaar_intervalle_lookup=zielpaar_intervalle_lookup,
        )
        if query_info is not None:
            query_infos.append(query_info)

    informative_query_infos = filter_informative_query_infos(
        query_infos=query_infos,
        kandidatenmenge=set(kandidatenmenge),
    )

    if not informative_query_infos:
        optimality_shares = estimate_optimality_shares(
            alternativen_matrix=alternativen_matrix,
            samples=samples,
            remaining_candidates=kandidatenmenge,
        )
        return build_no_informative_query_termination_result(
            kandidatenmenge=kandidatenmenge,
            optimality_shares=optimality_shares,
        )

    beste_query_info = min(
        informative_query_infos,
        key=lambda query_info: query_info.expected_kandidatenanzahl,
    )
    return beste_query_info.query
