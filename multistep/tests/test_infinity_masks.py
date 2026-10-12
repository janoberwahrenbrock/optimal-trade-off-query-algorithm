from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from multistep.src import reusable_highs as backend
from multistep.src.linear_programming import LINPROG_OPTIONS


def _legacy_replace(array):
    array[np.isposinf(array)] = backend._h.kHighsInf
    array[np.isneginf(array)] = -backend._h.kHighsInf


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return ('array', value.shape, value.dtype.str, value.strides, value.tobytes())
    if isinstance(value, dict):
        return {key: _snapshot(part) for key, part in value.items()}
    if isinstance(value, float):
        return ('float', value.hex())
    if isinstance(value, np.generic):
        return ('scalar', value.dtype.str, value.tobytes())
    return value


def _outcome(function, array):
    try:
        function(array)
        return ('ok',)
    except Exception as error:
        return ('error', type(error), str(error))


class InfinityMaskTests(unittest.TestCase):
    def tearDown(self):
        backend.clear_reusable_highs_cache()

    def test_4000_float64_masks_and_replacements_preserve_bits_and_layout(self):
        rng = np.random.default_rng(507132)
        with np.errstate(all='raise'), patch.object(backend._h, 'kHighsInf', 1e30):
            for index in range(4000):
                rows, columns = int(rng.integers(0, 12)), int(rng.integers(0, 8))
                raw = rng.integers(0, 2**64, size=(rows, columns), dtype=np.uint64)
                if raw.size:
                    special = np.asarray([
                        0, 0x8000000000000000, 0x7ff0000000000000,
                        0xfff0000000000000, 0x7ff8000000000001,
                        0xfff8000000000001, 0x7ff0000000000001,
                        0xfff0000000000001, 1, 0x8000000000000001,
                    ], dtype=np.uint64)
                    count = min(raw.size, len(special))
                    raw.reshape(-1)[:count] = special[:count]
                values = raw.view(np.float64)
                if index % 3 == 0:
                    original = np.asfortranarray(values)
                    actual = np.asfortranarray(values)
                else:
                    original, actual = values.copy(), values.copy()
                if index % 3 == 1:
                    original, actual = original[::-1, ::2], actual[::-1, ::2]
                with self.subTest(index=index):
                    self.assertEqual(np.isposinf(original).tobytes(),
                                     (original == np.inf).tobytes())
                    self.assertEqual(np.isneginf(original).tobytes(),
                                     (original == -np.inf).tobytes())
                    expected_layout = (original.shape, original.dtype.str, original.strides)
                    _legacy_replace(original)
                    backend._replace_highs_infinities(actual)
                    self.assertEqual((actual.shape, actual.dtype.str, actual.strides),
                                     expected_layout)
                    self.assertEqual(actual.tobytes(), original.tobytes())

    def test_scalar_empty_readonly_and_exotic_dtypes_keep_original_errors(self):
        for values in (
            np.asarray(np.inf), np.asarray(-np.inf), np.empty(0),
            np.asarray([np.inf, -np.inf, -0.0], dtype=np.float32),
            np.asarray([np.inf, -np.inf, -0.0], dtype='>f8'),
            np.asarray([0, 1, -1], dtype=np.int64),
            np.asarray([complex(1, 2)]), np.asarray(['x']), np.asarray([None], dtype=object),
        ):
            for readonly in (False, True):
                original, actual = values.copy(), values.copy()
                if readonly:
                    original.flags.writeable = actual.flags.writeable = False
                self.assertEqual(_outcome(backend._replace_highs_infinities, actual),
                                 _outcome(_legacy_replace, original))
                self.assertEqual(actual.tobytes(), original.tobytes())

    def test_subclass_array_function_dispatch_is_retained(self):
        class DispatchArray(np.ndarray):
            def __array_function__(self, function, types, args, kwargs):
                if function in (np.isposinf, np.isneginf):
                    self.calls.append(function.__name__)
                    result = np.zeros(self.shape, dtype=bool)
                    result[0 if function is np.isposinf else -1] = True
                    return result
                return super().__array_function__(function, types, args, kwargs)
        expected = np.asarray([1.0, 2.0, 3.0]).view(DispatchArray)
        actual = np.asarray([1.0, 2.0, 3.0]).view(DispatchArray)
        expected.calls, actual.calls = [], []
        with patch.object(backend._h, 'kHighsInf', 1e30):
            _legacy_replace(expected)
            backend._replace_highs_infinities(actual)
        self.assertEqual(actual.calls, expected.calls)
        self.assertEqual(actual.calls, ['isposinf', 'isneginf'])
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_positive_assignment_precedes_negative_mask_generation(self):
        with patch.object(backend._h, 'kHighsInf', -np.inf):
            expected = np.asarray([np.inf, -np.inf, -0.0, 0.0])
            actual = expected.copy()
            _legacy_replace(expected)
            backend._replace_highs_infinities(actual)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertTrue(np.isposinf(actual[:2]).all())

    def test_clean_inputs_guarantee_plain_float64_and_do_not_mutate_callers(self):
        cases = [
            dict(c=[-1.0], A_ub=[[1.0]], b_ub=[1.0], bounds=[(None, None)]),
            dict(c=np.asarray([-1.0], dtype=np.float32),
                 A_ub=np.asarray([[1.0]], dtype='>f8'),
                 b_ub=np.asarray([1.0], dtype=np.float32),
                 bounds=np.asarray([[-np.inf, np.inf]], dtype=np.float32)),
            dict(c=[1.0, -2.0], A_eq=[[1.0, 1.0]], b_eq=[1.0],
                 bounds=[(np.nan, np.nan), (-0.0, None)]),
        ]
        replacement = backend._replace_highs_infinities
        seen = []
        def record(array):
            self.assertIs(type(array), np.ndarray)
            self.assertEqual(array.dtype, np.dtype(float))
            self.assertFalse(np.isnan(array).any())
            seen.append(array.shape)
            replacement(array)
        for kwargs in cases:
            before = _snapshot(kwargs)
            backend.clear_reusable_highs_cache()
            with patch.object(backend, '_replace_highs_infinities', record):
                self.assertIsNotNone(backend.try_reusable_highs(kwargs))
            self.assertEqual(_snapshot(kwargs), before)
        self.assertEqual(len(seen), 12)

    def test_full_results_statuses_cache_lifecycle_and_mutations_match_legacy(self):
        rng = np.random.default_rng(507132)
        models = [
            dict(c=[1.0], A_ub=[[1.0], [-1.0]], b_ub=[0.0, -1.0], bounds=[(None, None)]),
            dict(c=[-1.0], bounds=[(None, None)]),
            dict(c=[1.0], A_eq=[[1.0]], b_eq=[1e-9], bounds=[(None, None)]),
            dict(c=[1.0, 0.0], A_eq=[[1.0, 1.0], [1.0, 0.0]], b_eq=[1.0, 0.0]),
        ]
        for _ in range(40):
            matrix = np.vstack((np.eye(4), -np.eye(4), rng.normal(size=(4, 4))))
            models.append(dict(c=rng.normal(size=4), A_ub=matrix,
                               b_ub=rng.uniform(0.2, 2.0, len(matrix)),
                               bounds=[(None, None)] * 4))
        comparisons = 0
        for method in ('highs', 'highs-ds', 'highs-ipm'):
            for presolve in (True, False):
                cases = [dict(model, method=method,
                              options={**LINPROG_OPTIONS, 'presolve': presolve})
                         for model in models]
                # Repeated objectives must use exactly the same cold reset path.
                cases += cases[:4]
                reference = self._solve_cases(cases, _legacy_replace)
                actual = self._solve_cases(cases, backend._replace_highs_infinities)
                self.assertEqual(actual, reference)
                comparisons += len(cases)
        self.assertEqual(comparisons, 288)
        matrix, rhs, bounds = np.asarray([[1.0]]), np.asarray([1.0]), np.asarray([[0.0, 4.0]])
        for _ in range(3):
            kwargs = dict(c=[-1.0], A_ub=matrix, b_ub=rhs, bounds=bounds)
            self.assertEqual(self._solve_cases([kwargs], backend._replace_highs_infinities),
                             self._solve_cases([kwargs], _legacy_replace))
            matrix[0, 0] += 0.1
            rhs[0] += 1.0
            bounds[0, 1] -= 0.3

    @staticmethod
    def _solve_cases(cases, replacement):
        backend.clear_reusable_highs_cache()
        results = []
        with patch.object(backend, '_replace_highs_infinities', replacement):
            for kwargs in cases:
                result = backend.try_reusable_highs(kwargs)
                if result is None:
                    raise AssertionError('valid models must retain the native path')
                results.append(_snapshot(result))
        return results, tuple(backend.reusable_highs_cache_info())


if __name__ == '__main__':
    unittest.main()
