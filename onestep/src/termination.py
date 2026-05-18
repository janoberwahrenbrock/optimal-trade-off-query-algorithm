from __future__ import annotations

from .build_ungleichungssysteme import build_utility_difference_zielfunktion
from .io_models import AlternativenMatrix, TerminationResult
from .ungleichungssysteme import Ungleichungssystem


def all_candidates_have_same_utility_values_in_W(
    alternativen_matrix: AlternativenMatrix,
    W: Ungleichungssystem,
    kandidatenmenge: list[int],
) -> bool:
    if not kandidatenmenge:
        return False

    referenz_index = kandidatenmenge[0]

    for candidate_index in kandidatenmenge[1:]:
        utility_difference_zielfunktion = build_utility_difference_zielfunktion(
            alternativen_matrix=alternativen_matrix,
            alternative_index_a=candidate_index,
            alternative_index_b=referenz_index,
        )
        minimization_result = W.minimize(utility_difference_zielfunktion)
        maximization_result = W.maximize(utility_difference_zielfunktion)

        if minimization_result.status != "optimal" or maximization_result.status != "optimal":
            raise ValueError(
                "utility difference optimization in W must be feasible and bounded"
            )

        if minimization_result.optimal_value is None or maximization_result.optimal_value is None:
            raise RuntimeError(
                "optimal utility difference optimization result has no optimal_value"
            )

        if abs(minimization_result.optimal_value) > 1e-9:
            return False

        if abs(maximization_result.optimal_value) > 1e-9:
            return False

    return True


def build_one_remaining_candidate_termination_result(
    candidate_index: int,
) -> TerminationResult:
    return TerminationResult(
        reason="one_remaining_candidate",
        share_precision="exact",
        remaining_candidates=[candidate_index],
        optimality_shares={candidate_index: 1.0},
    )


def build_same_utility_termination_result(
    kandidatenmenge: list[int],
) -> TerminationResult:
    return TerminationResult(
        reason="same_utility_values",
        share_precision="exact",
        remaining_candidates=kandidatenmenge,
        optimality_shares={candidate_index: 1.0 for candidate_index in kandidatenmenge},
    )


def build_no_informative_query_termination_result(
    kandidatenmenge: list[int],
    optimality_shares: dict[int, float],
) -> TerminationResult:
    return TerminationResult(
        reason="no_informative_query",
        share_precision="estimated",
        remaining_candidates=kandidatenmenge,
        optimality_shares=optimality_shares,
    )
