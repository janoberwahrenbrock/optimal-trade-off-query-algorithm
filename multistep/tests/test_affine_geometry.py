from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy.linalg import null_space

from multistep.src import affine_geometry, polytope_geometry, polytope_volume
from multistep.src.affine_geometry import (
    AffineDecomposition,
    affine_geometry_cache_info,
    clear_affine_geometry_cache,
    get_affine_decomposition,
)
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import Query
from multistep.src.polytope_geometry import enumerate_polytope_vertices
from multistep.src.polytope_volume import (
    clear_polytope_volume_cache,
    compute_exact_query_answer_probabilities,
    compute_polytope_intrinsic_volume,
    compute_polytope_intrinsic_volume_from_vertices,
    polytope_affine_cache_info,
)
from multistep.src.weight_space import build_weight_space


def _original_decomposition(
    variable_count: int,
    matrix: np.ndarray | None,
    right_side: np.ndarray | None,
) -> AffineDecomposition:
    if matrix is None:
        return AffineDecomposition(
            np.zeros(variable_count), np.eye(variable_count), 0.0
        )
    point, *_ = np.linalg.lstsq(matrix, right_side, rcond=None)
    residual = float(np.max(np.abs(matrix @ point - right_side), initial=0.0))
    return AffineDecomposition(point, null_space(matrix), residual)


class AffineGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_polytope_volume_cache()

    def test_geometry_and_volume_share_one_original_decomposition(self) -> None:
        system = build_weight_space(5, [])
        with patch.object(
            affine_geometry.np.linalg,
            "lstsq",
            wraps=np.linalg.lstsq,
        ) as least_squares, patch.object(
            affine_geometry,
            "null_space",
            wraps=null_space,
        ) as kernel:
            polytope = enumerate_polytope_vertices(system)
            self.assertGreater(
                compute_polytope_intrinsic_volume_from_vertices(system, polytope),
                0.0,
            )
        self.assertEqual(least_squares.call_count, 1)
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(polytope_affine_cache_info().misses, 1)
        self.assertEqual(polytope_affine_cache_info().hits, 1)

    def test_inequalities_do_not_change_equality_cache_key(self) -> None:
        system = build_weight_space(3, [])
        compute_polytope_intrinsic_volume(system)
        system.add_inequality([1.0, 0.0, 0.0], 0.5)
        compute_polytope_intrinsic_volume(system)
        info = affine_geometry_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 3)

    def test_rhs_coefficients_and_dimension_are_exact_key_components(self) -> None:
        matrix = np.asarray([[1.0, 1.0, 1.0]])
        right_side = np.asarray([1.0])
        first = get_affine_decomposition(3, matrix, right_side)
        same = get_affine_decomposition(3, matrix.copy(), right_side.copy())
        self.assertIs(first, same)
        right_side[0] = np.nextafter(1.0, 2.0)
        self.assertIsNot(first, get_affine_decomposition(3, matrix, right_side))
        matrix[0, 1] = np.nextafter(1.0, 2.0)
        self.assertIsNot(first, get_affine_decomposition(3, matrix, np.asarray([1.0])))
        self.assertEqual(get_affine_decomposition(2, None, None).basis.shape, (2, 2))
        self.assertEqual(get_affine_decomposition(3, None, None).basis.shape, (3, 3))
        self.assertEqual(affine_geometry_cache_info().misses, 5)

    def test_equalities_keep_independent_tolerance_decisions_on_cache_hit(self) -> None:
        system = LinearConstraintSystem(
            inequalities_left_side=[[-1.0], [1.0]],
            inequalities_right_side=[0.0, 2.0],
            equalities_left_side=[[1.0], [1.0]],
            equalities_right_side=[1.0, 1.0 + 1e-8],
        )
        strict = enumerate_polytope_vertices(system, tolerance=1e-10)
        permissive = enumerate_polytope_vertices(system, tolerance=1e-8)
        self.assertEqual(strict.status, "infeasible")
        self.assertEqual(permissive.status, "point")
        self.assertEqual(affine_geometry_cache_info().misses, 1)
        self.assertEqual(affine_geometry_cache_info().hits, 1)

    def test_arrays_cannot_be_mutated_and_keep_original_strides(self) -> None:
        matrix = np.ones((1, 5))
        right_side = np.asarray([1.0])
        actual = get_affine_decomposition(5, matrix, right_side)
        expected = _original_decomposition(5, matrix, right_side)
        np.testing.assert_array_equal(actual.particular_point, expected.particular_point)
        np.testing.assert_array_equal(actual.basis, expected.basis)
        self.assertEqual(actual.basis.strides, expected.basis.strides)
        for array in (actual.particular_point, actual.basis):
            with self.assertRaises(ValueError):
                array.flat[0] = 7.0
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        matrix[0, 0], right_side[0] = 2.0, 2.0
        np.testing.assert_array_equal(actual.particular_point, expected.particular_point)
        np.testing.assert_array_equal(actual.basis, expected.basis)

    def test_point_result_remains_independent_of_cache(self) -> None:
        system = LinearConstraintSystem(
            inequalities_left_side=[[-1.0]],
            inequalities_right_side=[0.0],
            equalities_left_side=[[1.0]],
            equalities_right_side=[0.5],
        )
        first = enumerate_polytope_vertices(system)
        first.vertices[0, 0] = 123.0
        second = enumerate_polytope_vertices(system)
        self.assertEqual(second.status, "point")
        self.assertEqual(second.vertices[0, 0], 0.5)

    def test_lru_entry_bound_and_cache_reset(self) -> None:
        with patch.object(affine_geometry, "_AFFINE_CACHE_MAX_SIZE", 2):
            first = get_affine_decomposition(2, None, None)
            get_affine_decomposition(3, None, None)
            get_affine_decomposition(2, None, None)
            get_affine_decomposition(4, None, None)
            self.assertEqual(affine_geometry_cache_info().currsize, 2)
            self.assertIs(first, get_affine_decomposition(2, None, None))
            get_affine_decomposition(3, None, None)
            self.assertEqual(affine_geometry_cache_info().misses, 4)
        clear_polytope_volume_cache()
        info = affine_geometry_cache_info()
        self.assertEqual((info.hits, info.misses, info.currsize, info.currbytes), (0, 0, 0, 0))

    def test_byte_bound_and_oversized_decomposition(self) -> None:
        with patch.object(affine_geometry, "_AFFINE_CACHE_MAX_BYTES", 100):
            get_affine_decomposition(2, None, None)
            get_affine_decomposition(3, None, None)
            info = affine_geometry_cache_info()
            self.assertEqual(info.currsize, 1)
            self.assertLessEqual(info.currbytes, 100)
            get_affine_decomposition(4, None, None)
            self.assertEqual(affine_geometry_cache_info().currsize, 1)
            self.assertLessEqual(affine_geometry_cache_info().currbytes, 100)

    def test_cached_geometry_volume_and_probabilities_are_bitwise_equal(self) -> None:
        for goal_count in (3, 5):
            system = build_weight_space(
                goal_count,
                [
                    Query(ziel_index_a=0, ziel_index_b=1, value=0.4).answer(">"),
                    Query(ziel_index_a=1, ziel_index_b=2, value=2.0).answer("<"),
                ],
            )
            for ratio in (0.1, 0.5, 1.0, 2.0, 10.0):
                query = Query(ziel_index_a=0, ziel_index_b=2, value=ratio)
                with self.subTest(goals=goal_count, ratio=ratio):
                    clear_polytope_volume_cache()
                    with ExitStack() as stack:
                        for module in (polytope_geometry, polytope_volume):
                            stack.enter_context(
                                patch.object(module, "get_affine_decomposition", side_effect=_original_decomposition)
                            )
                        original_vertices = enumerate_polytope_vertices(system)
                        original_volume = compute_polytope_intrinsic_volume(system)
                        original_probability = compute_exact_query_answer_probabilities(system, query)
                    clear_polytope_volume_cache()
                    cached_vertices = enumerate_polytope_vertices(system)
                    cached_volume = compute_polytope_intrinsic_volume(system)
                    cached_probability = compute_exact_query_answer_probabilities(system, query)
                    self.assertEqual(cached_vertices.status, original_vertices.status)
                    np.testing.assert_array_equal(cached_vertices.vertices, original_vertices.vertices)
                    self.assertEqual(cached_volume.hex(), original_volume.hex())
                    self.assertEqual(
                        [cached_probability[operator].hex() for operator in ("<", "=", ">")],
                        [original_probability[operator].hex() for operator in ("<", "=", ">")],
                    )


if __name__ == "__main__":
    unittest.main()
