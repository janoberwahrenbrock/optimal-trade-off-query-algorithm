from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_array

from multistep.src import reusable_highs as backend
from multistep.src.linear_programming import (
    LINPROG_OPTIONS, run_linprog_with_retries,
)


class ReusableHighsTests(unittest.TestCase):
    def setUp(self) -> None:
        backend.clear_reusable_highs_cache()

    def compare(self, **kwargs):
        kwargs = {"method": "highs", "options": LINPROG_OPTIONS, **kwargs}
        expected = linprog(**kwargs)
        actual = backend.try_reusable_highs(kwargs)
        self.assertIsNotNone(actual)
        self.assertEqual(actual.status, expected.status)
        self.assertEqual(actual.success, expected.success)
        self.assertEqual(actual.message, expected.message)
        self.assertEqual(actual.nit, expected.nit)
        self.assertEqual(actual.crossover_nit, expected.crossover_nit)
        for field in ("x", "fun", "slack", "con"):
            np.testing.assert_array_equal(actual[field], expected[field])
        for field in ("ineqlin", "eqlin", "lower", "upper"):
            for part in ("residual", "marginals"):
                np.testing.assert_array_equal(actual[field][part], expected[field][part])
        return actual

    def test_min_max_and_repeated_objectives_preserve_cold_results(self) -> None:
        for _ in range(3):
            for objective in ([1., 0., 0.], [-1., 0., 0.], [0., 0., 0.], [1., -2., 0.]):
                self.compare(c=objective, A_eq=[[1., 1., 1.]], b_eq=[1.])
        info = backend.reusable_highs_cache_info()
        self.assertEqual(info.model_hits, 11)
        self.assertEqual(info.solver_creations, 1)
        self.assertEqual(info.fallbacks, 0)

    def test_center_lps_match_for_many_models_and_recycle_solver(self) -> None:
        rng = np.random.default_rng(194)
        for _ in range(70):
            matrix = np.vstack((np.eye(3), -np.eye(3), rng.normal(size=(4, 3))))
            norms = np.linalg.norm(matrix, axis=1)
            rhs = rng.uniform(.2, 2., len(matrix))
            self.compare(c=[0., 0., 0., -1.], A_ub=np.column_stack((matrix, norms)),
                         b_ub=rhs, bounds=[(None, None)] * 3 + [(0., None)])
        info = backend.reusable_highs_cache_info()
        self.assertEqual(info.model_misses, 70)
        self.assertLessEqual(info.currsize, info.maxsize)
        self.assertLessEqual(info.currbytes, info.maxbytes)
        self.assertEqual(info.solver_creations, info.maxsize)
        self.assertEqual(info.fallbacks, 0)

    def test_infeasible_unbounded_and_near_boundary_match(self) -> None:
        models = [
            dict(c=[1.], A_ub=[[1.], [-1.]], b_ub=[0., -1.], bounds=[(None, None)]),
            dict(c=[-1.], bounds=[(None, None)]),
            dict(c=[1.], A_eq=[[1.]], b_eq=[1e-9], bounds=[(None, None)]),
            dict(c=[1., 0.], A_eq=[[1., 1.], [1., 0.]], b_eq=[1., 0.]),
        ]
        for method in ("highs", "highs-ds", "highs-ipm"):
            for presolve in (True, False):
                for model in models:
                    self.compare(**model, method=method,
                                 options={**LINPROG_OPTIONS, "presolve": presolve})

    def test_constraint_and_bound_mutations_never_reuse_stale_model(self) -> None:
        matrix, rhs = np.asarray([[1.]]), np.asarray([1.])
        bounds = np.asarray([[0., 4.]])
        for index in range(3):
            self.compare(c=[-1.], A_ub=matrix, b_ub=rhs, bounds=bounds)
            rhs[0] += 1.
            matrix[0, 0] += .1
            bounds[0, 1] -= .3
        self.assertEqual(backend.reusable_highs_cache_info().model_hits, 0)

    def test_unsupported_inputs_and_options_use_public_fallback(self) -> None:
        for kwargs in (
            {"c": [1.], "integrality": [1]},
            {"c": [1.], "options": {"time_limit": 1.}},
            {"c": [1.], "A_ub": csc_array([[1.]]), "b_ub": [1.]},
            {"c": [1.], "options": {"disp": True}},
        ):
            self.assertIsNone(backend.try_reusable_highs(kwargs))
        with patch.object(backend, "_AVAILABLE", False):
            self.assertIsNone(backend.try_reusable_highs({"c": [1.]}))
        self.assertEqual(backend.reusable_highs_cache_info().fallbacks, 5)

    def test_native_error_clears_cache_and_falls_back(self) -> None:
        with patch.object(backend, "_solve", side_effect=RuntimeError("native failure")):
            result = run_linprog_with_retries(c=[1.], options=LINPROG_OPTIONS)
        self.assertTrue(result.success)
        self.assertEqual(backend.reusable_highs_cache_info().fallbacks, 1)
        self.assertEqual(backend.reusable_highs_cache_info().currsize, 0)

    def test_invalid_model_retains_public_validation(self) -> None:
        with self.assertRaises(ValueError):
            run_linprog_with_retries(c=[np.nan], options=LINPROG_OPTIONS)

    def test_infeasible_retry_uses_separate_options_and_same_status(self) -> None:
        result = run_linprog_with_retries(c=[1.], A_ub=[[1.], [-1.]],
                                         b_ub=[0., -1.], options=LINPROG_OPTIONS)
        self.assertEqual(result.status, 2)
        info = backend.reusable_highs_cache_info()
        self.assertEqual(info.calls, 2)
        self.assertEqual(info.model_misses, 2)
        self.assertEqual(info.fallbacks, 0)

    def test_threads_keep_models_and_counters_separate(self) -> None:
        import threading
        barrier = threading.Barrier(2)

        def worker(value):
            backend.clear_reusable_highs_cache()
            barrier.wait()
            self.compare(c=[-1.], A_ub=[[1.]], b_ub=[value])
            self.compare(c=[1.], A_ub=[[1.]], b_ub=[value])
            return backend.reusable_highs_cache_info()

        with ThreadPoolExecutor(max_workers=2) as executor:
            infos = list(executor.map(worker, (1., 2.)))
        self.assertTrue(all(info.calls == 2 and info.model_hits == 1 for info in infos))
        self.assertEqual(backend.reusable_highs_cache_info().calls, 0)

    def test_entry_and_byte_limits_do_not_affect_results(self) -> None:
        with patch.object(backend, "_MAX_MODELS", 2), patch.object(backend, "_MAX_MODEL_BYTES", 100):
            for value in (1., 2., 3., 1.):
                self.compare(c=[-1., -2.], A_eq=[[1., 1.]], b_eq=[value])
            info = backend.reusable_highs_cache_info()
            self.assertLessEqual(info.currsize, 2)
            self.assertLessEqual(info.currbytes, 100)

    def test_process_identity_change_drops_inherited_native_state(self) -> None:
        self.compare(c=[1.])
        backend._state().pid = -1
        self.assertEqual(backend.reusable_highs_cache_info().calls, 0)
        self.assertEqual(backend.reusable_highs_cache_info().currsize, 0)
        self.compare(c=[-1.], bounds=[(0., 1.)])
        self.assertEqual(backend.reusable_highs_cache_info().solver_creations, 1)


if __name__ == "__main__":
    unittest.main()
