from __future__ import annotations

from .models import Query, QueryOperator


ANSWER_OPTIONS: tuple[QueryOperator, ...] = ("<", "=", ">")


def classify_query_answer(
    weights: list[float],
    query: Query,
    equality_tol: float = 0.0,
) -> QueryOperator:
    _validate_equality_tol(equality_tol)
    _validate_weights_match_query(weights, query)

    difference = (
        float(weights[int(query.ziel_index_a)])
        - float(query.value) * float(weights[int(query.ziel_index_b)])
    )

    if abs(difference) <= equality_tol:
        return "="

    if difference < 0.0:
        return "<"

    return ">"


def estimate_query_answer_probabilities(
    query: Query,
    samples: list[list[float]],
    equality_tol: float = 0.0,
) -> dict[QueryOperator, float]:
    _validate_equality_tol(equality_tol)

    if not samples:
        raise ValueError("samples must not be empty")

    answer_counts = {answer: 0 for answer in ANSWER_OPTIONS}

    for weights in samples:
        answer = classify_query_answer(
            weights=weights,
            query=query,
            equality_tol=equality_tol,
        )
        answer_counts[answer] += 1

    sample_count = len(samples)
    return {
        answer: answer_counts[answer] / sample_count
        for answer in ANSWER_OPTIONS
    }


def estimate_query_answer_probability(
    query: Query,
    answer: QueryOperator,
    samples: list[list[float]],
    equality_tol: float = 0.0,
) -> float:
    if answer not in ANSWER_OPTIONS:
        raise ValueError(f"unknown answer: {answer}")

    return estimate_query_answer_probabilities(
        query=query,
        samples=samples,
        equality_tol=equality_tol,
    )[answer]


def _validate_weights_match_query(weights: list[float], query: Query) -> None:
    if query.ziel_index_a >= len(weights):
        raise IndexError("query.ziel_index_a is out of range for weights")

    if query.ziel_index_b >= len(weights):
        raise IndexError("query.ziel_index_b is out of range for weights")


def _validate_equality_tol(equality_tol: float) -> None:
    if equality_tol < 0.0:
        raise ValueError("equality_tol must not be negative")
