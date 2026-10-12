from __future__ import annotations

from unittest.mock import patch

import numpy as np

import multistep.src.ratio_intervals as ratio_intervals_module
from multistep.src.models import AlternativenMatrix, AnsweredQuery
from multistep.src.ratio_intervals import (
    _compute_ratio_intervals_from_vertices,
    _compute_prepared_ratio_intervals_from_vertices,
    compute_all_ratio_interval_analysis,
)
from multistep.src.weight_space import build_weight_space
from multistep.tests import test_batched_ratio_intervals as reference_tests
from multistep.tests.e04_ratio_reference import compute_e04_reference_intervals


class VectorizedRatioIntervalsTest(reference_tests.BatchedRatioIntervalsTest):
    def assert_reference_intervals(self, vertices, goals, tolerance):
        super().assert_reference_intervals(vertices, goals, tolerance)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected = compute_e04_reference_intervals(vertices, goals, tolerance)
            actual = _compute_ratio_intervals_from_vertices(vertices, goals, tolerance)
        self.assertEqual(list(actual), list(expected))
        for pair in expected:
            self.assertEqual(
                reference_tests._hexify(actual[pair].model_dump()),
                reference_tests._hexify(expected[pair].model_dump()),
                msg=str(pair),
            )

    def test_full_candidate_analysis_preserves_volumes_messages_and_pair_order(self):
        super().test_full_candidate_analysis_preserves_volumes_messages_and_pair_order()
        generator = np.random.default_rng(321104)
        for seed_index in range(3):
            alternatives = AlternativenMatrix(entries=generator.random((5, 5)).tolist())
            for answers in (
                [],
                [AnsweredQuery(ziel_index_a=0, ziel_index_b=1, value=0.75, operator="<")],
                [AnsweredQuery(ziel_index_a=0, ziel_index_b=1, value=1.25, operator=">"),
                 AnsweredQuery(ziel_index_a=2, ziel_index_b=3, value=0.8, operator="<")],
                [AnsweredQuery(ziel_index_a=0, ziel_index_b=1, value=1.0, operator="=")],
            ):
                with self.subTest(seed=seed_index, answers=answers, reference="e04"):
                    kwargs = dict(alternatives=alternatives, weight_space=build_weight_space(5, answers),
                                  candidates=[0, 1, 2, 3, 4], include_candidate_volumes=True)
                    with patch(
                        "multistep.src.ratio_intervals._compute_ratio_intervals_from_vertices",
                        side_effect=compute_e04_reference_intervals,
                    ):
                        expected = compute_all_ratio_interval_analysis(**kwargs)
                    actual = compute_all_ratio_interval_analysis(**kwargs)
                    self.assertEqual(reference_tests._analysis_signature(actual),
                                     reference_tests._analysis_signature(expected))

    def test_where_mask_avoids_zero_denominator_and_diagonal_division(self):
        vertices = np.asarray([
            [0.0, -0.0, 1.0, 0.0, 0.5],
            [0.5, 0.0, 0.0, 0.0, 0.5],
        ])
        with np.errstate(divide="raise", invalid="raise"):
            self.assert_reference_intervals(vertices, 5, 1e-10)

    def test_temporary_matrix_memory_limit_uses_unchanged_fallback(self):
        vertices = np.ones((3, 5), dtype=np.float64)
        with patch.object(ratio_intervals_module, "_MAX_BATCHED_RATIO_ELEMENTS", 50):
            with patch(
                "multistep.src.ratio_intervals._compute_prepared_ratio_intervals_from_vertices",
                wraps=_compute_prepared_ratio_intervals_from_vertices,
            ) as prepared:
                _compute_ratio_intervals_from_vertices(vertices, 5, 1e-10)
                self.assertEqual(prepared.call_count, 1)
            self.assert_reference_intervals(vertices, 5, 1e-10)

    def test_dispatch_threshold_neighbors_preserve_bits_and_all_pair_objects(self):
        generator = np.random.default_rng(310551)
        for goals in (3, 5, 8):
            threshold = 4 * goals * (goals - 1)
            for rows in (threshold - 1, threshold, threshold + 1):
                vertices = generator.random((rows, goals))
                vertices[generator.random(vertices.shape) < 0.1] = -0.0
                with self.subTest(goals=goals, rows=rows, threshold=threshold):
                    with patch(
                        "multistep.src.ratio_intervals._compute_prepared_ratio_intervals_from_vertices",
                        wraps=_compute_prepared_ratio_intervals_from_vertices,
                    ) as prepared:
                        _compute_ratio_intervals_from_vertices(vertices, goals, 1e-10)
                    self.assertEqual(prepared.call_count, int(rows > threshold))
                    self.assert_reference_intervals(vertices, goals, 1e-10)
