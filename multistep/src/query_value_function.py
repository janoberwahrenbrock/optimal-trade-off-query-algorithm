from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import compute_candidate_set, estimate_candidate_set_from_samples
from .models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from .query_probability import (
    ANSWER_OPTIONS,
    classify_query_answer,
    estimate_query_answer_probabilities,
)
from .weight_space import build_weight_space


@dataclass(frozen=True)
class QueryAnswerEvaluation:
    answer: QueryOperator
    probability: float
    candidate_count: int
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class QueryValueEvaluation:
    query: Query
    expected_candidate_count: float
    answer_evaluations: tuple[QueryAnswerEvaluation, ...]


def compute_sample_ratio_range(
    samples: list[list[float]],
    numerator_index: int,
    denominator_index: int,
    denominator_tol: float = 1e-12,
) -> tuple[float, float]:
    if not samples:
        raise ValueError("samples must not be empty")

    if numerator_index == denominator_index:
        raise ValueError("numerator_index and denominator_index must be different")

    if denominator_tol < 0.0:
        raise ValueError("denominator_tol must not be negative")

    ratios: list[float] = []
    for sample in samples:
        if numerator_index >= len(sample):
            raise IndexError("numerator_index is out of range for samples")

        if denominator_index >= len(sample):
            raise IndexError("denominator_index is out of range for samples")

        denominator = float(sample[denominator_index])
        if denominator > denominator_tol:
            ratios.append(float(sample[numerator_index]) / denominator)

    if not ratios:
        raise ValueError("no sample has a denominator above denominator_tol")

    return min(ratios), max(ratios)


def build_linear_query_values(
    lower: float,
    upper: float,
    steps: int,
) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")

    if lower > upper:
        raise ValueError("lower must not be greater than upper")

    if lower == upper:
        return [float(lower)]

    return np.linspace(lower, upper, steps, dtype=float).tolist()


def get_ordered_goal_pairs(goal_count: int) -> list[tuple[int, int]]:
    if goal_count <= 0:
        raise ValueError("goal_count must be positive")

    return [
        (goal_index_a, goal_index_b)
        for goal_index_a in range(goal_count)
        for goal_index_b in range(goal_count)
        if goal_index_a != goal_index_b
    ]


def filter_samples_for_query_answer(
    samples: list[list[float]],
    query: Query,
    answer: QueryOperator,
    equality_tol: float = 0.0,
) -> list[list[float]]:
    if answer not in ANSWER_OPTIONS:
        raise ValueError(f"unknown answer: {answer}")

    return [
        sample
        for sample in samples
        if classify_query_answer(
            weights=sample,
            query=query,
            equality_tol=equality_tol,
        )
        == answer
    ]


def evaluate_query_value(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    query: Query,
    samples: list[list[float]],
    equality_tol: float = 0.0,
) -> QueryValueEvaluation:
    probabilities = estimate_query_answer_probabilities(
        query=query,
        samples=samples,
        equality_tol=equality_tol,
    )

    answer_evaluations: list[QueryAnswerEvaluation] = []
    expected_candidate_count = 0.0

    for answer in ANSWER_OPTIONS:
        child_answered_queries = answered_queries + [query.answer(answer)]
        child_weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=child_answered_queries,
        )

        if child_weight_space.is_feasible():
            candidates = tuple(
                compute_candidate_set(
                    alternatives=alternatives,
                    weight_space=child_weight_space,
                )
            )
        else:
            candidates = ()

        probability = probabilities[answer]
        candidate_count = len(candidates)
        expected_candidate_count += probability * candidate_count
        answer_evaluations.append(
            QueryAnswerEvaluation(
                answer=answer,
                probability=probability,
                candidate_count=candidate_count,
                candidates=candidates,
            )
        )

    return QueryValueEvaluation(
        query=query,
        expected_candidate_count=expected_candidate_count,
        answer_evaluations=tuple(answer_evaluations),
    )


def evaluate_query_value_curve_for_goal_pair(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    samples: list[list[float]],
    goal_pair: tuple[int, int],
    query_values: list[float],
    equality_tol: float = 0.0,
) -> list[QueryValueEvaluation]:
    numerator_index, denominator_index = goal_pair
    evaluations: list[QueryValueEvaluation] = []

    for query_value in query_values:
        query = Query(
            ziel_index_a=numerator_index,
            ziel_index_b=denominator_index,
            value=float(query_value),
        )
        evaluations.append(
            evaluate_query_value(
                alternatives=alternatives,
                answered_queries=answered_queries,
                query=query,
                samples=samples,
                equality_tol=equality_tol,
            )
        )

    return evaluations


def estimate_query_value_from_samples(
    alternatives: AlternativenMatrix,
    query: Query,
    samples: list[list[float]],
    equality_tol: float = 0.0,
    utility_tol: float = 1e-9,
) -> QueryValueEvaluation:
    if not samples:
        raise ValueError("samples must not be empty")

    probabilities = estimate_query_answer_probabilities(
        query=query,
        samples=samples,
        equality_tol=equality_tol,
    )

    answer_evaluations: list[QueryAnswerEvaluation] = []
    expected_candidate_count = 0.0

    for answer in ANSWER_OPTIONS:
        answer_samples = filter_samples_for_query_answer(
            samples=samples,
            query=query,
            answer=answer,
            equality_tol=equality_tol,
        )
        if answer_samples:
            candidates = tuple(
                estimate_candidate_set_from_samples(
                    alternatives=alternatives,
                    samples=answer_samples,
                    utility_tol=utility_tol,
                )
            )
        else:
            candidates = ()

        probability = probabilities[answer]
        candidate_count = len(candidates)
        expected_candidate_count += probability * candidate_count
        answer_evaluations.append(
            QueryAnswerEvaluation(
                answer=answer,
                probability=probability,
                candidate_count=candidate_count,
                candidates=candidates,
            )
        )

    return QueryValueEvaluation(
        query=query,
        expected_candidate_count=expected_candidate_count,
        answer_evaluations=tuple(answer_evaluations),
    )


def find_best_estimated_one_step_query_for_samples(
    alternatives: AlternativenMatrix,
    samples: list[list[float]],
    query_value_steps: int,
    max_query_value: float,
    equality_tol: float = 0.0,
    utility_tol: float = 1e-9,
) -> QueryValueEvaluation:
    if not samples:
        raise ValueError("samples must not be empty")

    if max_query_value < 0.0:
        raise ValueError("max_query_value must not be negative")

    best_evaluation: QueryValueEvaluation | None = None
    query_values = build_linear_query_values(
        lower=0.0,
        upper=max_query_value,
        steps=query_value_steps,
    )

    for goal_pair in get_ordered_goal_pairs(alternatives.get_anzahl_spalten()):
        for query_value in query_values:
            query = Query(
                ziel_index_a=goal_pair[0],
                ziel_index_b=goal_pair[1],
                value=float(query_value),
            )
            evaluation = estimate_query_value_from_samples(
                alternatives=alternatives,
                query=query,
                samples=samples,
                equality_tol=equality_tol,
                utility_tol=utility_tol,
            )
            if (
                best_evaluation is None
                or evaluation.expected_candidate_count
                < best_evaluation.expected_candidate_count
            ):
                best_evaluation = evaluation

    if best_evaluation is None:
        raise RuntimeError("no query evaluation was produced")

    return best_evaluation


def estimate_depth_two_query_value_from_samples(
    alternatives: AlternativenMatrix,
    query: Query,
    samples: list[list[float]],
    inner_query_value_steps: int,
    max_query_value: float,
    equality_tol: float = 0.0,
    utility_tol: float = 1e-9,
) -> QueryValueEvaluation:
    if not samples:
        raise ValueError("samples must not be empty")

    probabilities = estimate_query_answer_probabilities(
        query=query,
        samples=samples,
        equality_tol=equality_tol,
    )
    answer_evaluations: list[QueryAnswerEvaluation] = []
    expected_candidate_count = 0.0

    for answer in ANSWER_OPTIONS:
        answer_samples = filter_samples_for_query_answer(
            samples=samples,
            query=query,
            answer=answer,
            equality_tol=equality_tol,
        )
        probability = probabilities[answer]

        if answer_samples:
            continuation = find_best_estimated_one_step_query_for_samples(
                alternatives=alternatives,
                samples=answer_samples,
                query_value_steps=inner_query_value_steps,
                max_query_value=max_query_value,
                equality_tol=equality_tol,
                utility_tol=utility_tol,
            )
            candidate_count = continuation.expected_candidate_count
        else:
            candidate_count = 0.0

        expected_candidate_count += probability * candidate_count
        answer_evaluations.append(
            QueryAnswerEvaluation(
                answer=answer,
                probability=probability,
                candidate_count=int(round(candidate_count)),
                candidates=(),
            )
        )

    return QueryValueEvaluation(
        query=query,
        expected_candidate_count=expected_candidate_count,
        answer_evaluations=tuple(answer_evaluations),
    )


def estimate_depth_two_value_curve_for_goal_pair_from_samples(
    alternatives: AlternativenMatrix,
    samples: list[list[float]],
    goal_pair: tuple[int, int],
    query_values: list[float],
    inner_query_value_steps: int,
    max_query_value: float,
    equality_tol: float = 0.0,
    utility_tol: float = 1e-9,
) -> list[QueryValueEvaluation]:
    evaluations: list[QueryValueEvaluation] = []

    for query_value in query_values:
        query = Query(
            ziel_index_a=goal_pair[0],
            ziel_index_b=goal_pair[1],
            value=float(query_value),
        )
        evaluations.append(
            estimate_depth_two_query_value_from_samples(
                alternatives=alternatives,
                query=query,
                samples=samples,
                inner_query_value_steps=inner_query_value_steps,
                max_query_value=max_query_value,
                equality_tol=equality_tol,
                utility_tol=utility_tol,
            )
        )

    return evaluations
