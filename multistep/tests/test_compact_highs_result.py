from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_array

from multistep.src import linear_programming as lp_api
from multistep.src import reusable_highs as backend


def numeric_signature(value):
    """Compare every primal bit, including the sign of zero."""
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array.shape, tuple(float(item).hex() for item in array.flat)


class CompactHighsResultTests(unittest.TestCase):
    def setUp(self) -> None:
        backend.clear_reusable_highs_cache()

    def assert_primal_identical(self, actual, expected) -> None:
        for field in ("success", "status", "message", "nit", "crossover_nit"):
            self.assertEqual(actual[field], expected[field], field)
        for field in ("x", "fun", "slack", "con"):
            self.assertEqual(numeric_signature(actual[field]),
                             numeric_signature(expected[field]), field)

    def test_recycled_cold_models_preserve_all_primal_bits(self) -> None:
        rng = np.random.default_rng(741)
        # More models than the native cache holds exercise instance recycling;
        # two objectives per model also exercise template reloads on cache hits.
        for _ in range(40):
            matrix = rng.normal(size=(4, 5))
            for objective in (rng.normal(size=5), rng.normal(size=5)):
                kwargs = dict(c=objective, A_ub=matrix, b_ub=np.full(4, 5.),
                              A_eq=np.ones((1, 5)), b_eq=[1.],
                              bounds=[(0., 1.)] * 5, method="highs",
                              options=lp_api.LINPROG_OPTIONS)
                expected = linprog(**kwargs)
                full = backend.try_reusable_highs(kwargs)
                compact = backend.try_reusable_highs(kwargs, compact_result=True)
                self.assertIsNotNone(full)
                self.assertIsNotNone(compact)
                self.assert_primal_identical(full, expected)
                self.assert_primal_identical(compact, expected)
                self.assertIn("lower", full)
                self.assertNotIn("lower", compact)
        info = backend.reusable_highs_cache_info()
        self.assertGreater(info.solver_reuses, 0)
        self.assertEqual(info.fallbacks, 0)

    def test_methods_presolve_and_failure_statuses_preserve_bits(self) -> None:
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
                                  options={**lp_api.LINPROG_OPTIONS,
                                           "presolve": presolve})
                    expected = linprog(**kwargs)
                    compact = backend.try_reusable_highs(kwargs, compact_result=True)
                    self.assertIsNotNone(compact)
                    self.assert_primal_identical(compact, expected)

    def test_compact_path_keeps_validation_and_never_reads_duals_or_basis(self) -> None:
        class PrimalOnlySolution:
            col_value = [1., 0.]
            row_value = [1.]

            @property
            def row_dual(self):
                raise AssertionError("compact result must not read row duals")

            @property
            def col_dual(self):
                raise AssertionError("compact result must not read column duals")

        solver = Mock()
        solver.getModelStatus.return_value = backend._h.HighsModelStatus.kOptimal
        solver.getInfo.return_value = SimpleNamespace(
            objective_function_value=1., simplex_iteration_count=1,
            ipm_iteration_count=0, crossover_iteration_count=0,
        )
        solver.modelStatusToString.return_value = "Optimal"
        solver.getSolution.return_value = PrimalOnlySolution()
        solver.getBasis.side_effect = AssertionError("compact result must not read basis")
        cleaned = backend._clean_inputs(backend._LPProblem(
            [1., 0.], None, None, [[1., 1.]], [1.], None, None, None,
        ))
        with patch.object(backend, "_check_result", wraps=backend._check_result) as check:
            result = backend._result(solver, cleaned, compact_result=True)
        self.assertTrue(result.success)
        check.assert_called_once()
        self.assertEqual(check.call_args.args[6], 1e-9)
        solver.getBasis.assert_not_called()
        self.assertEqual(set(result), {"x", "fun", "slack", "con", "status",
                                       "message", "success", "nit", "crossover_nit"})

        # The primal feasibility check must still downgrade a nominal native
        # optimum whose equality residual is invalid.
        solver.getSolution.return_value.row_value = [2.]
        invalid = backend._result(solver, cleaned, compact_result=True)
        self.assertFalse(invalid.success)
        self.assertEqual(invalid.status, 4)

    def test_default_public_result_remains_complete(self) -> None:
        result = lp_api.run_linprog_with_retries(
            c=[1., -2.], A_eq=[[1., 1.]], b_eq=[1.],
            options=lp_api.LINPROG_OPTIONS,
        )
        for field in ("ineqlin", "eqlin", "lower", "upper"):
            self.assertEqual(set(result[field]), {"residual", "marginals"})

    def test_public_override_and_native_fallback_never_receive_private_flag(self) -> None:
        kwargs = dict(c=[1.], options=lp_api.LINPROG_OPTIONS)
        expected = linprog(**kwargs)
        with patch.object(lp_api, "linprog", return_value=expected) as public:
            actual = lp_api.run_linprog_with_retries(
                _internal_compact_result=True, **kwargs,
            )
        self.assertIs(actual, expected)
        public.assert_called_once_with(**kwargs)
        with patch.object(backend, "_AVAILABLE", False), \
                patch.object(lp_api, "_suppress_native_solver_output") as suppress:
            # Keep the original linprog identity, exercising the native
            # availability fallback rather than the explicit override branch.
            actual = lp_api.run_linprog_with_retries(
                _internal_compact_result=True, **kwargs,
            )
        self.assert_primal_identical(actual, expected)
        self.assertIn("lower", actual)
        suppress.assert_called_once()

    def test_invalid_inputs_and_unsupported_options_retain_public_behavior(self) -> None:
        for invalid in (
            dict(c=[np.nan]),
            dict(c=[1., 2.], A_ub=[[1.]], b_ub=[1.]),
            dict(c=[1.], A_eq=[[np.inf]], b_eq=[1.]),
        ):
            with self.assertRaises(ValueError):
                lp_api.run_linprog_with_retries(_internal_compact_result=True, **invalid)
        kwargs = dict(c=[-1.], A_ub=csc_array([[1.]]), b_ub=[1.],
                      options={**lp_api.LINPROG_OPTIONS, "time_limit": 1.})
        result = lp_api.run_linprog_with_retries(_internal_compact_result=True, **kwargs)
        self.assert_primal_identical(result, linprog(**kwargs))
        self.assertIn("lower", result)

    def test_native_error_drops_cache_and_returns_complete_public_fallback(self) -> None:
        kwargs = dict(c=[1.], options=lp_api.LINPROG_OPTIONS)
        backend.try_reusable_highs(kwargs)
        self.assertEqual(backend.reusable_highs_cache_info().currsize, 1)
        with patch.object(backend, "_solve", side_effect=RuntimeError("native error")):
            actual = lp_api.run_linprog_with_retries(
                _internal_compact_result=True, **kwargs,
            )
        self.assert_primal_identical(actual, linprog(**kwargs))
        self.assertIn("lower", actual)
        info = backend.reusable_highs_cache_info()
        self.assertEqual(info.currsize, 0)
        self.assertEqual(info.fallbacks, 1)

    def test_constraint_mutations_and_infeasible_retry_remain_exact(self) -> None:
        matrix, rhs = np.asarray([[1.]]), np.asarray([1.])
        bounds = np.asarray([[0., 4.]])
        for _ in range(3):
            kwargs = dict(c=[-1.], A_ub=matrix, b_ub=rhs, bounds=bounds,
                          options=lp_api.LINPROG_OPTIONS)
            actual = lp_api.run_linprog_with_retries(_internal_compact_result=True, **kwargs)
            self.assert_primal_identical(actual, linprog(**kwargs))
            rhs[0] += 1.
            matrix[0, 0] += .1
            bounds[0, 1] -= .3
        self.assertEqual(backend.reusable_highs_cache_info().model_hits, 0)
        backend.clear_reusable_highs_cache()
        kwargs = dict(c=[1.], A_ub=[[1.], [-1.]], b_ub=[0., -1.],
                      options=lp_api.LINPROG_OPTIONS)
        compact = lp_api.run_linprog_with_retries(_internal_compact_result=True, **kwargs)
        full = lp_api.run_linprog_with_retries(**kwargs)
        self.assert_primal_identical(compact, full)
        self.assertEqual(compact.status, 2)
        self.assertEqual(backend.reusable_highs_cache_info().calls, 4)


if __name__ == "__main__":
    unittest.main()
