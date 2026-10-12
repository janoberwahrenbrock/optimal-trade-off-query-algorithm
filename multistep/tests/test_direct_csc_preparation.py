from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_array
import scipy.sparse._sputils as sputils

from multistep.src import reusable_highs as backend
from multistep.src.linear_programming import LINPROG_OPTIONS, run_linprog_with_retries


def numeric_signature(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array.shape, tuple(float(item).hex() for item in array.flat)


class DirectCscPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        backend.clear_reusable_highs_cache()

    def assert_csc_identical(self, a_ub, a_eq):
        expected = csc_array(np.vstack((a_ub, a_eq)))
        actual = backend._constraint_matrix_csc(a_ub, a_eq)
        for field in ("data", "indices", "indptr"):
            left, right = getattr(actual, field), getattr(expected, field)
            self.assertEqual(left.dtype, right.dtype, field)
            self.assertEqual(left.shape, right.shape, field)
            self.assertEqual(left.strides, right.strides, field)
            self.assertEqual(left.tobytes(), right.tobytes(), field)
            self.assertTrue(left.flags.c_contiguous, field)
        return actual

    def assert_result_identical(self, actual, expected):
        for field in ("success", "status", "message", "nit", "crossover_nit"):
            self.assertEqual(actual[field], expected[field], field)
        for field in ("x", "fun", "slack", "con"):
            self.assertEqual(numeric_signature(actual[field]),
                             numeric_signature(expected[field]), field)
        for field in ("ineqlin", "eqlin", "lower", "upper"):
            for part in ("residual", "marginals"):
                self.assertEqual(numeric_signature(actual[field][part]),
                                 numeric_signature(expected[field][part]), (field, part))

    def test_empty_zero_and_signed_zero_arrays_match_every_byte(self) -> None:
        for rows, cols in ((0, 5), (5, 0), (0, 0), (3, 5), (1, 1)):
            matrix = np.zeros((rows, cols), dtype=float)
            matrix.flat[::2] = -0.0
            for split in (0, rows // 2, rows):
                self.assert_csc_identical(matrix[:split], matrix[split:])
        matrix = np.asarray([[0., -0., 2., 0.], [-3., 0., -0., 0.],
                             [0., 4., -5., -0.]], dtype=float)
        actual = self.assert_csc_identical(matrix[:2], matrix[2:])
        self.assertEqual(actual.indptr.tolist(), [0, 1, 2, 4, 4])
        self.assertEqual(actual.indices.tolist(), [1, 2, 0, 2])
        self.assertEqual(actual.data.tolist(), [-3., 4., 2., -5.])

    def test_random_matrices_and_strides_preserve_column_and_row_order(self) -> None:
        rng = np.random.default_rng(6006)
        for index in range(70):
            rows, cols = index % 19, 1 + index % 8
            matrix = rng.normal(size=(rows, cols))
            matrix[rng.uniform(size=matrix.shape) < .6] = 0.
            for source in (matrix, np.asfortranarray(matrix), matrix[::-1, ::-1]):
                self.assert_csc_identical(source[:rows // 2], source[rows // 2:])
        readonly = np.asarray([[1., 0.], [0., -2.]])
        readonly.setflags(write=False)
        self.assert_csc_identical(readonly[:1], readonly[1:])

    def test_coefficient_bits_and_non_float64_fallback_remain_exact(self) -> None:
        tiny = np.nextafter(0., 1.)
        matrix = np.asarray([[tiny, -tiny, 1e-300, -1e300],
                             [np.nextafter(1., 2.), 0., -0., -1e-300]])
        self.assert_csc_identical(matrix[:1], matrix[1:])
        for dtype in (np.float32, np.int8, np.int64, np.dtype(">f8")):
            matrix = np.asarray([[1, 0, -2], [0, 3, 0]], dtype=dtype)
            actual = self.assert_csc_identical(matrix[:1], matrix[1:])
            if np.vstack((matrix[:1], matrix[1:])).dtype != np.dtype(np.float64):
                self.assertIsInstance(actual, csc_array)

    def test_missing_dtype_helper_keeps_previous_csc_path(self) -> None:
        matrix = np.asarray([[1., 0.], [0., -2.]])
        with patch.object(backend, "_get_index_dtype", None):
            actual = self.assert_csc_identical(matrix[:1], matrix[1:])
            result = backend.try_reusable_highs(dict(
                c=[-1., 0.], A_ub=[[1., 0.]], b_ub=[1.], options=LINPROG_OPTIONS,
            ))
        self.assertIsInstance(actual, csc_array)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(backend.reusable_highs_cache_info().fallbacks, 0)

    def test_int64_policy_matches_scipy_without_large_allocations(self) -> None:
        # Force the same installed policy in both paths. No giant matrix is
        # needed to exercise the dtype-dependent array preparation.
        with patch.object(backend, "_get_index_dtype", return_value=np.int64), \
                patch.object(sputils, "get_index_dtype", return_value=np.int64):
            for matrix in (np.empty((0, 4)), np.eye(4), np.zeros((3, 4))):
                actual = self.assert_csc_identical(matrix[:1], matrix[1:])
                self.assertEqual(actual.indices.dtype, np.dtype(np.int64))
                self.assertEqual(actual.indptr.dtype, np.dtype(np.int64))
            kwargs = dict(c=[1., -2.], A_eq=[[1., 1.]], b_eq=[1.],
                          options=LINPROG_OPTIONS)
            actual = backend.try_reusable_highs(kwargs)
            self.assertIsNotNone(actual)
            self.assert_result_identical(actual, linprog(**kwargs))
        self.assertEqual(sputils.get_index_dtype(maxval=2**31 - 1), np.int32)
        self.assertEqual(sputils.get_index_dtype(maxval=2**31), np.int64)

    def test_random_model_misses_hits_and_native_recycling_match_public_solver(self) -> None:
        rng = np.random.default_rng(606)
        for _ in range(40):
            matrix = rng.normal(size=(7, 5))
            rhs = matrix @ np.full(5, .2) + .3
            for objective in (rng.normal(size=5), rng.normal(size=5)):
                kwargs = dict(c=objective, A_ub=matrix, b_ub=rhs,
                              A_eq=np.ones((1, 5)), b_eq=[1.], bounds=[(0., 1.)] * 5,
                              method="highs", options=LINPROG_OPTIONS)
                actual = backend.try_reusable_highs(kwargs)
                self.assertIsNotNone(actual)
                self.assert_result_identical(actual, linprog(**kwargs))
        info = backend.reusable_highs_cache_info()
        self.assertEqual(info.model_misses, 40)
        self.assertEqual(info.model_hits, 40)
        self.assertEqual(info.solver_creations, backend._MAX_MODELS)
        self.assertEqual(info.fallbacks, 0)

    def test_solver_methods_presolve_and_failure_statuses_keep_every_result_bit(self) -> None:
        models = [
            dict(c=[1.], A_ub=[[1.], [-1.]], b_ub=[0., -1.], bounds=[(None, None)]),
            dict(c=[-1.], bounds=[(None, None)]),
            dict(c=[1.], A_eq=[[1.]], b_eq=[1e-9], bounds=[(None, None)]),
            dict(c=[1., 0.], A_eq=[[1., 1.], [1., 0.]], b_eq=[1., 0.]),
        ]
        for method in ("highs", "highs-ds", "highs-ipm"):
            for presolve in (True, False):
                for model in models:
                    kwargs = dict(**model, method=method,
                                  options={**LINPROG_OPTIONS, "presolve": presolve})
                    actual = backend.try_reusable_highs(kwargs)
                    self.assertIsNotNone(actual)
                    self.assert_result_identical(actual, linprog(**kwargs))

    def test_input_validation_and_preparation_error_keep_public_fallback(self) -> None:
        for invalid in (
            dict(c=[np.nan]),
            dict(c=[1.], A_ub=[[np.inf]], b_ub=[1.]),
            dict(c=[1.], A_ub=[[1.]], b_ub=[np.nan]),
            dict(c=[1., 2.], A_eq=[[1.]], b_eq=[1.]),
        ):
            with self.assertRaises(ValueError):
                run_linprog_with_retries(**invalid)
        backend.clear_reusable_highs_cache()
        kwargs = dict(c=[1.], A_eq=[[1.]], b_eq=[1.], options=LINPROG_OPTIONS)
        backend.try_reusable_highs(dict(c=[1.], options=LINPROG_OPTIONS))
        self.assertEqual(backend.reusable_highs_cache_info().currsize, 1)
        with patch.object(backend, "_constraint_matrix_csc", side_effect=RuntimeError("setup error")):
            actual = run_linprog_with_retries(**kwargs)
        self.assert_result_identical(actual, linprog(**kwargs))
        self.assertEqual(backend.reusable_highs_cache_info().currsize, 0)
        self.assertEqual(backend.reusable_highs_cache_info().fallbacks, 1)


if __name__ == "__main__":
    unittest.main()
