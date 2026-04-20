from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .io_models import Query
from .ratio_intervalle import RatioInterval, ZielpaarIntervalle



QUERY_EPSILON = 1e-3
# Abstand, um fertige Query-Werte von alpha bzw. beta weg ins offene Intervall zu verschieben.

RAW_QUERY_TOL = 1e-12
# Interne Toleranz zum Zusammenfassen fast gleicher Roh-/Query-Werte innerhalb eines festen Zielpaars.

QUERY_DEDUP_ABS_TOL = 1e-12
# Absolute Toleranz für die globale Deduplizierung äquivalenter Spiegel-Query-Kandidaten.

QUERY_DEDUP_REL_TOL = 1e-9
# Relative Toleranz für die globale Deduplizierung äquivalenter Spiegel-Query-Kandidaten.



@dataclass(frozen=True)
class _RawQueryCandidate:
    ziel_index_a: int
    ziel_index_b: int
    breakpoint_value: float
    side: Literal["left", "right"]


def compute_all_query_kandidaten(
    zielpaar_intervalle_liste: list[ZielpaarIntervalle],
    epsilon: float = QUERY_EPSILON,
    dedup_abs_tol: float = QUERY_DEDUP_ABS_TOL,
    dedup_rel_tol: float = QUERY_DEDUP_REL_TOL,
) -> list[Query]:
    _validate_epsilon(epsilon)
    _validate_dedup_tolerances(dedup_abs_tol, dedup_rel_tol)

    raw_query_candidates: list[_RawQueryCandidate] = []

    for zielpaar_intervalle in zielpaar_intervalle_liste:
        raw_query_candidates.extend(
            _extract_raw_query_candidates_for_pair(
                zielpaar_intervalle=zielpaar_intervalle,
                epsilon=epsilon,
            )
        )

    deduplicated_raw_query_candidates = _deduplicate_raw_query_candidates(
        raw_query_candidates=raw_query_candidates,
        epsilon=epsilon,
        abs_tol=dedup_abs_tol,
        rel_tol=dedup_rel_tol,
    )
    query_kandidaten = [
        _build_query_from_raw_candidate(raw_query_candidate, epsilon)
        for raw_query_candidate in deduplicated_raw_query_candidates
    ]
    query_kandidaten.sort(
        key=lambda query: (
            int(query.ziel_index_a),
            int(query.ziel_index_b),
            float(query.value),
        )
    )
    return query_kandidaten


def _validate_epsilon(epsilon: float) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")


def _validate_dedup_tolerances(abs_tol: float, rel_tol: float) -> None:
    if abs_tol < 0:
        raise ValueError("abs_tol must not be negative")

    if rel_tol < 0:
        raise ValueError("rel_tol must not be negative")


def _extract_raw_query_candidates_for_pair(
    zielpaar_intervalle: ZielpaarIntervalle,
    epsilon: float,
) -> list[_RawQueryCandidate]:
    raw_query_candidates: list[_RawQueryCandidate] = []

    for ratio_interval in zielpaar_intervalle.intervalle_pro_kandidat.values():
        _extend_raw_query_candidates_from_ratio_interval(
            raw_query_candidates=raw_query_candidates,
            ratio_interval=ratio_interval,
            ziel_index_a=int(zielpaar_intervalle.ziel_index_a),
            ziel_index_b=int(zielpaar_intervalle.ziel_index_b),
            epsilon=epsilon,
        )

    raw_query_candidates.sort(
        key=lambda candidate: (
            candidate.side,
            candidate.breakpoint_value,
        )
    )

    deduplicated_raw_query_candidates: list[_RawQueryCandidate] = []
    for raw_query_candidate in raw_query_candidates:
        if not deduplicated_raw_query_candidates or not _have_same_raw_candidate_form(
            deduplicated_raw_query_candidates[-1],
            raw_query_candidate,
            abs_tol=RAW_QUERY_TOL,
            rel_tol=0.0,
        ):
            deduplicated_raw_query_candidates.append(raw_query_candidate)

    return deduplicated_raw_query_candidates


def _extend_raw_query_candidates_from_ratio_interval(
    raw_query_candidates: list[_RawQueryCandidate],
    ratio_interval: RatioInterval,
    ziel_index_a: int,
    ziel_index_b: int,
    epsilon: float,
) -> None:
    lower = ratio_interval.lower
    upper = ratio_interval.upper

    if lower.status == "unbounded":
        raise ValueError("lower ratio bound must not be unbounded")

    if lower.status == "optimal":
        if lower.optimal_value is None:
            raise RuntimeError("optimal lower ratio bound has no optimal_value")

        if lower.optimal_value > epsilon:
            raw_query_candidates.append(
                _RawQueryCandidate(
                    ziel_index_a=ziel_index_a,
                    ziel_index_b=ziel_index_b,
                    breakpoint_value=float(lower.optimal_value),
                    side="left",
                )
            )

    if upper.status == "optimal":
        if upper.optimal_value is None:
            raise RuntimeError("optimal upper ratio bound has no optimal_value")

        raw_query_candidates.append(
            _RawQueryCandidate(
                ziel_index_a=ziel_index_a,
                ziel_index_b=ziel_index_b,
                breakpoint_value=float(upper.optimal_value),
                side="right",
            )
        )


def _deduplicate_raw_query_candidates(
    raw_query_candidates: list[_RawQueryCandidate],
    epsilon: float,
    abs_tol: float,
    rel_tol: float,
) -> list[_RawQueryCandidate]:
    canonical_raw_query_candidates = [
        _canonicalize_raw_query_candidate(
            raw_query_candidate=raw_query_candidate,
            epsilon=epsilon,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        for raw_query_candidate in raw_query_candidates
    ]
    canonical_raw_query_candidates.sort(
        key=lambda candidate: (
            candidate.ziel_index_a,
            candidate.ziel_index_b,
            candidate.side,
            candidate.breakpoint_value,
        )
    )

    deduplicated_raw_query_candidates: list[_RawQueryCandidate] = []
    for raw_query_candidate in canonical_raw_query_candidates:
        if not deduplicated_raw_query_candidates or not _have_same_raw_candidate_form(
            deduplicated_raw_query_candidates[-1],
            raw_query_candidate,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ):
            deduplicated_raw_query_candidates.append(raw_query_candidate)

    return deduplicated_raw_query_candidates


def _canonicalize_raw_query_candidate(
    raw_query_candidate: _RawQueryCandidate,
    epsilon: float,
    abs_tol: float,
    rel_tol: float,
) -> _RawQueryCandidate:
    if raw_query_candidate.breakpoint_value <= 0.0:
        return raw_query_candidate

    query = _build_query_from_raw_candidate(raw_query_candidate, epsilon)
    mirrored_raw_query_candidate = _build_mirrored_raw_query_candidate(raw_query_candidate)
    mirrored_query = _build_query_from_raw_candidate(mirrored_raw_query_candidate, epsilon)

    query_value = float(query.value)
    mirrored_query_value = float(mirrored_query.value)

    if query_value >= 1.0 and mirrored_query_value < 1.0:
        return raw_query_candidate

    if mirrored_query_value >= 1.0 and query_value < 1.0:
        return mirrored_raw_query_candidate

    if math.isclose(query_value, mirrored_query_value, abs_tol=abs_tol, rel_tol=rel_tol):
        return min(
            raw_query_candidate,
            mirrored_raw_query_candidate,
            key=lambda candidate: (
                candidate.ziel_index_a,
                candidate.ziel_index_b,
                candidate.side,
                candidate.breakpoint_value,
            ),
        )

    return max(
        raw_query_candidate,
        mirrored_raw_query_candidate,
        key=lambda candidate: float(_build_query_from_raw_candidate(candidate, epsilon).value),
    )


def _build_mirrored_raw_query_candidate(
    raw_query_candidate: _RawQueryCandidate,
) -> _RawQueryCandidate:
    mirrored_side: Literal["left", "right"] = "right"
    if raw_query_candidate.side == "right":
        mirrored_side = "left"

    return _RawQueryCandidate(
        ziel_index_a=raw_query_candidate.ziel_index_b,
        ziel_index_b=raw_query_candidate.ziel_index_a,
        breakpoint_value=1.0 / raw_query_candidate.breakpoint_value,
        side=mirrored_side,
    )


def _build_query_from_raw_candidate(
    raw_query_candidate: _RawQueryCandidate,
    epsilon: float,
) -> Query:
    if raw_query_candidate.side == "left":
        query_value = raw_query_candidate.breakpoint_value - epsilon
    else:
        query_value = raw_query_candidate.breakpoint_value + epsilon

    return Query(
        ziel_index_a=raw_query_candidate.ziel_index_a,
        ziel_index_b=raw_query_candidate.ziel_index_b,
        value=query_value,
    )


def _have_same_raw_candidate_form(
    raw_query_candidate_a: _RawQueryCandidate,
    raw_query_candidate_b: _RawQueryCandidate,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    return (
        raw_query_candidate_a.ziel_index_a == raw_query_candidate_b.ziel_index_a
        and raw_query_candidate_a.ziel_index_b == raw_query_candidate_b.ziel_index_b
        and raw_query_candidate_a.side == raw_query_candidate_b.side
        and math.isclose(
            raw_query_candidate_a.breakpoint_value,
            raw_query_candidate_b.breakpoint_value,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    )
