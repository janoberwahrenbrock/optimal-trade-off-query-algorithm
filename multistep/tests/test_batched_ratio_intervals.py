from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from multistep.src.models import AlternativenMatrix, AnsweredQuery
from multistep.src.ratio_intervals import (
    RatioInterval,
    _compute_ratio_interval_from_vertices,
    _compute_ratio_intervals_from_vertices,
    compute_all_ratio_interval_analysis,
    get_ordered_goal_pairs,
)
from multistep.src.weight_space import build_weight_space


def _reference_batch(
    vertices: np.ndarray, goal_count: int, tolerance: float
) -> dict[tuple[int, int], RatioInterval]:
    return {
        (goal_index_a, goal_index_b): _compute_ratio_interval_from_vertices(
            vertices, goal_index_a, goal_index_b, tolerance
        )
        for goal_index_a, goal_index_b in get_ordered_goal_pairs(goal_count)
    }


def _hexify(value):
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {key: _hexify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hexify(item) for item in value]
    return value


def _analysis_signature(analysis):
    return _hexify({
        "ratio_intervals": [pair.model_dump() for pair in analysis.ratio_intervals],
        "candidate_volumes": analysis.candidate_volumes,
        "candidate_volume_shares": analysis.candidate_volume_shares,
    })


class BatchedRatioIntervalsTest(unittest.TestCase):
    def assert_reference_intervals(self, vertices, goals, tolerance):
        before = vertices.tobytes()
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected = _reference_batch(vertices, goals, tolerance)
            actual = _compute_ratio_intervals_from_vertices(vertices, goals, tolerance)
        self.assertEqual(list(actual), get_ordered_goal_pairs(goals))
        self.assertEqual(list(actual), list(expected))
        for pair in expected:
            self.assertIsInstance(actual[pair], RatioInterval)
            self.assertEqual(
                _hexify(actual[pair].model_dump()),
                _hexify(expected[pair].model_dump()),
                msg=str(pair),
            )
        self.assertEqual(vertices.tobytes(), before)

    def test_random_geometries_include_zero_columns_negative_values_and_duplicates(self):
        generator = np.random.default_rng(2026101204)
        for goals in (1, 2, 3, 5, 8):
            for rows in (1, 2, 5, 16, 80):
                for tolerance in (1e-3, 1e-10, 1e-14):
                    with self.subTest(goals=goals, rows=rows, tolerance=tolerance):
                        vertices = generator.normal(size=(rows, goals))
                        vertices[generator.random(vertices.shape) < 0.15] = 0.0
                        if rows > 1:
                            vertices[-1] = vertices[0]
                        self.assert_reference_intervals(vertices, goals, tolerance)

    def test_tolerance_boundaries_signed_zero_and_subnormal_values(self):
        for tolerance in (1e-3, 1e-10, 1e-14):
            boundaries = np.asarray([
                -0.0, 0.0, np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0),
                tolerance, np.nextafter(tolerance, 0.0),
                np.nextafter(tolerance, np.inf), -tolerance, 1.0,
            ])
            vertices = np.column_stack([np.roll(boundaries, index) for index in range(5)])
            self.assert_reference_intervals(vertices, 5, tolerance)

    def test_random_finite_float_bits_and_overflowing_direct_ratios(self):
        generator = np.random.default_rng(839104)
        bits = generator.integers(0, np.iinfo(np.uint64).max, size=(128, 5), dtype=np.uint64)
        bits &= np.uint64(0xFFEFFFFFFFFFFFFF)
        vertices = bits.view(np.float64)
        self.assertTrue(np.all(np.isfinite(vertices)))
        for tolerance in (1e-10, 1e-100, 1e-300):
            self.assert_reference_intervals(vertices, 5, tolerance)

    def test_strided_reversed_and_fortran_geometries(self):
        generator = np.random.default_rng(912)
        base = generator.normal(size=(40, 15))
        for vertices in (base[:, ::3], base[::-1, ::-3], np.asfortranarray(base[:, ::3])):
            self.assert_reference_intervals(vertices, 5, 1e-10)

    def test_denominator_zero_and_unbounded_statuses_preserve_full_objects(self):
        vertices = np.asarray([
            [-0.0, 1.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.5, 0.0],
            [0.0, 0.5, 0.5, 0.0, -0.0],
        ])
        self.assert_reference_intervals(vertices, 5, 1e-10)
        intervals = _compute_ratio_intervals_from_vertices(vertices, 5, 1e-10)
        self.assertEqual(intervals[(1, 0)].lower.status, "infeasible")
        self.assertEqual(intervals[(1, 0)].lower.solver_message,
                         "ratio denominator is zero throughout the candidate region")
        self.assertEqual(intervals[(1, 2)].upper.status, "unbounded")
        self.assertEqual(intervals[(0, 1)].lower.optimal_value.hex(), "0x0.0p+0")

    def test_unusual_dtypes_shapes_subclasses_and_nonfinite_values_use_fallback(self):
        class ArraySubclass(np.ndarray):
            pass

        ordinary = np.asarray([[0.0, 1.0, 0.5, 0.2, 0.7], [0.5, 0.0, 0.5, 0.8, 0.3]])
        nonfinite = ordinary.copy()
        nonfinite[0, 2] = np.nan
        nonfinite[1, 4] = np.inf
        for vertices in (
            ordinary.astype(np.float32), ordinary.astype(np.int64), ordinary.astype(">f8"),
            ordinary.view(ArraySubclass), nonfinite, np.empty((0, 5)),
            np.column_stack((ordinary, ordinary[:, 0])), ordinary[:, :, None],
        ):
            with self.subTest(dtype=str(vertices.dtype), shape=vertices.shape):
                self.assert_reference_intervals(vertices, 5, 1e-10)
                with np.errstate(invalid="ignore"):
                    with patch(
                        "multistep.src.ratio_intervals._compute_ratio_interval_from_vertices",
                        wraps=_compute_ratio_interval_from_vertices,
                    ) as original:
                        _compute_ratio_intervals_from_vertices(vertices, 5, 1e-10)
                self.assertEqual(original.call_count, 20)

    def test_one_dimensional_fallback_preserves_original_exception(self):
        vertices = np.asarray([0.0, 1.0, 0.5, 0.2, 0.7])
        with self.assertRaises(IndexError):
            _reference_batch(vertices, 5, 1e-10)
        with self.assertRaises(IndexError):
            _compute_ratio_intervals_from_vertices(vertices, 5, 1e-10)

    def test_finite_native_float64_computes_all_twenty_pairs_without_fallback(self):
        vertices = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.1, 0.4, 0.3, 0.5]])
        with patch(
            "multistep.src.ratio_intervals._compute_ratio_interval_from_vertices",
            side_effect=AssertionError("finite float64 should use batched path"),
        ):
            actual = _compute_ratio_intervals_from_vertices(vertices, 5, 1e-10)
        self.assertEqual(list(actual), get_ordered_goal_pairs(5))
        self.assertEqual(len(actual), 20)

    def test_full_candidate_analysis_preserves_volumes_messages_and_pair_order(self):
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
                with self.subTest(seed=seed_index, answers=answers):
                    weight_space = build_weight_space(5, answers)
                    kwargs = dict(alternatives=alternatives, weight_space=weight_space,
                                  candidates=[0, 1, 2, 3, 4], include_candidate_volumes=True)
                    with patch(
                        "multistep.src.ratio_intervals._compute_ratio_intervals_from_vertices",
                        side_effect=_reference_batch,
                    ):
                        expected = compute_all_ratio_interval_analysis(**kwargs)
                    actual = compute_all_ratio_interval_analysis(**kwargs)
                    self.assertEqual(_analysis_signature(actual), _analysis_signature(expected))


if __name__ == "__main__":
    unittest.main()
