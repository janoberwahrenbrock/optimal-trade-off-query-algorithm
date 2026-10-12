from __future__ import annotations

import ast
import hashlib
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
from scipy.optimize import linprog

from multistep.src import continuous_lp_validation as validator
from multistep.src import linear_programming as lp_api
from multistep.src import reusable_highs as backend


def outcome(function, args, mode="ignore"):
    before = [(a.shape, a.dtype.str, a.strides, a.tobytes())
              for a in args if isinstance(a, np.ndarray)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all=mode):
            try:
                result = ("return", function(*args))
            except Exception as error:
                result = ("error", type(error), str(error))
    after = [(a.shape, a.dtype.str, a.strides, a.tobytes())
             for a in args if isinstance(a, np.ndarray)]
    assert before == after
    return result, [(type(w.message), str(w.message)) for w in caught]


class ContinuousLPValidationTests(unittest.TestCase):
    def setUp(self):
        backend.clear_reusable_highs_cache()
        self.args = (np.ones(5), 0.25, 0, np.ones(15), np.zeros(1),
                     np.tile([-np.inf, np.inf], (5, 1)), 1e-9, "native", None)

    def check(self, *args):
        return validator.check_private_continuous_result(backend._check_result, *args)

    def assert_primal_equal(self, actual, expected):
        for field in ("success", "status", "message", "nit", "crossover_nit"):
            self.assertEqual(actual[field], expected[field], field)
        for field in ("x", "fun", "slack", "con"):
            a, b = actual[field], expected[field]
            if a is None or b is None:
                self.assertIs(a, b, field)
            else:
                a, b = np.asarray(a), np.asarray(b)
                self.assertEqual((a.shape, a.dtype.str, a.tobytes()),
                                 (b.shape, b.dtype.str, b.tobytes()), field)

    def test_source_verification_failure_preserves_actual_checker(self):
        self.assertFalse(validator._source_is_supported(None))
        with patch.object(validator.inspect, "getsource", side_effect=RuntimeError("source read")):
            self.assertFalse(validator._source_is_supported(backend._check_result))
        with patch.object(validator.inspect, "getsource", return_value="def changed(): pass\n"):
            self.assertFalse(validator._source_is_supported(backend._check_result))
        with patch.object(validator.inspect, "getsource", return_value=None):
            self.assertFalse(validator._source_is_supported(backend._check_result))
        checker = Mock(wraps=backend._check_result)
        with patch.object(validator, "_SOURCE_VERIFIED", False), \
                patch.object(validator, "_continuous_check_result", side_effect=AssertionError("fast")):
            self.assertEqual(validator.check_private_continuous_result(checker, *self.args),
                             backend._check_result(*self.args))
        checker.assert_called_once_with(*self.args)

    def test_checker_and_isclose_overrides_remain_observable(self):
        checker = Mock(wraps=backend._check_result)
        validator.check_private_continuous_result(checker, *self.args)
        checker.assert_called_once_with(*self.args)
        with patch.object(np, "isclose", side_effect=RuntimeError("isclose override")) as hook:
            with self.assertRaisesRegex(RuntimeError, "isclose override"):
                self.check(*self.args)
        hook.assert_called_once()

    def test_in_place_checker_code_override_uses_original_function(self):
        checker = backend._check_result
        original_code = checker.__code__

        def replacement(*args):
            return 4, "in-place checker override"

        try:
            checker.__code__ = replacement.__code__
            with patch.object(validator, "_continuous_check_result",
                              side_effect=AssertionError("fast path bypassed code override")) as fast:
                self.assertEqual(self.check(*self.args), (4, "in-place checker override"))
            fast.assert_not_called()
        finally:
            checker.__code__ = original_code

    def test_scipy_checker_numpy_alias_override_uses_original_function(self):
        class NumpyOverride:
            def __getattr__(self, name):
                if name == "sqrt":
                    raise RuntimeError("SciPy NumPy alias override")
                return getattr(np, name)

        with patch.dict(backend._check_result.__globals__, {"np": NumpyOverride()}), \
                patch.object(validator, "_continuous_check_result",
                             side_effect=AssertionError("fast path bypassed alias override")) as fast:
            with self.assertRaisesRegex(RuntimeError, "SciPy NumPy alias override"):
                self.check(*self.args)
        fast.assert_not_called()

    def test_only_explicit_internal_compact_native_call_uses_specialization(self):
        kwargs = dict(c=[1., -2.], A_eq=[[1., 1.]], b_eq=[1.],
                      options=lp_api.LINPROG_OPTIONS)
        with patch.object(backend, "check_private_continuous_result",
                          wraps=backend.check_private_continuous_result) as check:
            full = lp_api.run_linprog_with_retries(**kwargs)
            direct_compact = backend.try_reusable_highs(kwargs, compact_result=True)
            self.assertEqual(check.call_count, 0)
            internal = lp_api.run_linprog_with_retries(_internal_compact_result=True, **kwargs)
            self.assertEqual(check.call_count, 1)
        expected = linprog(**kwargs)
        for result in (full, direct_compact, internal):
            self.assert_primal_equal(result, expected)
        self.assertIn("lower", full)
        self.assertNotIn("lower", internal)

    def test_dead_allowance_is_the_only_removed_source(self):
        if not validator._SOURCE_VERIFIED:
            self.skipTest("unknown SciPy checker preserves original path")
        source = inspect.getsource(backend._check_result)
        self.assertIn(hashlib.sha256(source.encode()).hexdigest(),
                      validator._SUPPORTED_SOURCE_SHA256)
        expected = source.replace("def _check_result(", "def _continuous_check_result(", 1)
        expected = expected.replace("        if integrality is None:\n            integrality = 0\n", "", 1)
        expected = expected.replace(
            "        # When integrality is 2 or 3, x must be within bounds OR take value 0\n"
            "        valid_bounds |= (integrality > 1) & np.isclose(x, 0, atol=tol)\n", "", 1)
        expected_tree = ast.parse(expected).body[0]
        actual_tree = ast.parse(inspect.getsource(validator._continuous_check_result)).body[0]
        # The new private helper has its own accurate concise docstring; the
        # original installed SciPy function/source pin remains unchanged.
        expected_tree.body.pop(0)
        actual_tree.body.pop(0)
        self.assertEqual(ast.dump(actual_tree, include_attributes=False),
                         ast.dump(expected_tree, include_attributes=False))
        with patch.object(validator, "_continuous_check_result",
                          wraps=validator._continuous_check_result) as fast:
            self.assertEqual(self.check(*self.args), backend._check_result(*self.args))
        fast.assert_called_once()

    def test_independent_nonfinite_fields_and_boundary_status_messages(self):
        tolerance = float(np.sqrt(1e-9) * 10)
        special = (0., -0., tolerance, np.nextafter(tolerance, 0.),
                   np.nextafter(tolerance, np.inf), -tolerance,
                   np.finfo(float).max, -np.finfo(float).max,
                   np.inf, -np.inf, np.nan,
                   np.array([0x7ff0000000000001], dtype=np.uint64).view(float)[0])
        for value in special:
            for field in (0, 1, 3, 4, 5):
                for status in (0, 2, 3, 4):
                    args = list(self.args)
                    args[2] = status
                    args[0], args[3], args[4], args[5] = (
                        self.args[0].copy(), self.args[3].copy(),
                        self.args[4].copy(), self.args[5].copy())
                    if field == 1:
                        args[field] = float(value)
                    elif field == 5:
                        args[field][0, 0] = value
                    else:
                        args[field][0] = value
                    for mode in ("ignore", "warn", "raise"):
                        self.assertEqual(outcome(self.check, args, mode),
                                         outcome(backend._check_result, args, mode),
                                         (field, status, mode))

    def test_unusual_arrays_shapes_and_integrality_use_original_dispatch(self):
        cases = []
        for field in (0, 3, 4, 5):
            for conversion in (lambda x: x.astype(np.float32), lambda x: x[::-1],
                               lambda x: x.tolist(), lambda x: x.view(np.matrix)):
                args = list(self.args)
                args[field] = conversion(args[field])
                cases.append(args)
        for field, value in ((0, None), (0, np.ones((1, 5))),
                             (5, np.ones((5, 3))), (1, np.float64(.25)),
                             (2, np.int64(0)), (6, .1), (8, 0), (8, 1), (8, 2),
                             (8, np.array([0, 1, 2, 3, 0]))):
            args = list(self.args)
            args[field] = value
            cases.append(args)
        for args in cases:
            with patch.object(validator, "_continuous_check_result",
                              side_effect=AssertionError("unusual result entered fast path")) as fast:
                self.assertEqual(outcome(self.check, args), outcome(backend._check_result, args))
            fast.assert_not_called()

    def test_public_solver_override_retains_entire_retry_sequence(self):
        results = [SimpleNamespace(success=False, status=4, message="unknown"),
                   SimpleNamespace(success=False, status=4, message="unknown"),
                   SimpleNamespace(success=False, status=4, message="unknown"),
                   SimpleNamespace(success=False, status=3, message="unbounded")]
        with patch.object(lp_api, "linprog", side_effect=results) as public:
            actual = lp_api.run_linprog_with_retries(
                _internal_compact_result=True, c=[1.], method="highs",
                options=lp_api.LINPROG_OPTIONS)
        self.assertIs(actual, results[-1])
        calls = public.call_args_list
        self.assertEqual([c.kwargs["method"] for c in calls],
                         ["highs", "highs", "highs-ds", "highs-ipm"])
        self.assertEqual(calls[0].kwargs["options"], lp_api.LINPROG_OPTIONS)
        self.assertTrue(all(c.kwargs["options"] == lp_api.LINPROG_OPTIONS_WITHOUT_PRESOLVE
                            for c in calls[1:]))
        self.assertTrue(all(set(c.kwargs) == {"c", "method", "options"} for c in calls))


if __name__ == "__main__":
    unittest.main()
