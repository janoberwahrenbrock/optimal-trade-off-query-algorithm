from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from multistep.src.polytope_geometry import _deduplicate_rows


def _reference_deduplicate_rows(values: np.ndarray, tolerance: float) -> np.ndarray:
    decimals = max(0, int(-np.log10(tolerance)))
    _, indices = np.unique(np.round(values, decimals=decimals), axis=0, return_index=True)
    return values[np.sort(indices)]


class PolytopeRowDeduplicationTest(unittest.TestCase):
    def assert_reference_bits(self, values: np.ndarray, tolerance: float) -> None:
        before = values.tobytes()
        with np.errstate(over="ignore", invalid="ignore"):
            expected = _reference_deduplicate_rows(values, tolerance)
            actual = _deduplicate_rows(values, tolerance)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(values.tobytes(), before)

    def test_random_duplicates_preserve_first_original_row_bits_and_order(self) -> None:
        generator = np.random.default_rng(20261012)
        for goal_count in (1, 2, 3, 5, 8):
            for row_count in (0, 1, 2, 7, 32, 101):
                for tolerance in (1e-3, 1e-10, 1e-14):
                    with self.subTest(goals=goal_count, rows=row_count, tolerance=tolerance):
                        originals = generator.normal(size=(row_count, goal_count))
                        duplicate_indices = generator.integers(
                            0, max(row_count, 1), size=row_count
                        )
                        duplicates = originals[duplicate_indices]
                        near_duplicates = duplicates + tolerance * 0.01
                        values = np.concatenate((originals, duplicates, near_duplicates))
                        generator.shuffle(values, axis=0)
                        self.assert_reference_bits(values, tolerance)

    def test_rounding_boundaries_include_adjacent_float_values(self) -> None:
        for decimals in (0, 3, 10, 14):
            half_values = np.asarray([0.5, 1.5, -0.5, -1.5]) * 10.0 ** -decimals
            values = np.concatenate((
                np.nextafter(half_values, -np.inf),
                half_values,
                np.nextafter(half_values, np.inf),
                half_values,
            ))
            self.assert_reference_bits(np.column_stack((values, -values)), 10.0 ** -decimals)

    def test_signed_zero_compares_equal_without_changing_original_signs(self) -> None:
        values = np.asarray([
            [-0.0, 0.0, -1e-12],
            [0.0, -0.0, 1e-12],
            [1.0, -0.0, 2.0],
            [1.0, 0.0, 2.0],
        ])
        self.assert_reference_bits(values, 1e-10)
        result = _deduplicate_rows(values, 1e-10)
        self.assertEqual(result.tobytes(), values[[0, 2]].tobytes())

    def test_noncontiguous_reversed_and_fortran_arrays(self) -> None:
        generator = np.random.default_rng(2203)
        base = generator.normal(size=(40, 12))
        base[20:] = base[:20]
        for values in (base[:, ::2], base[::-1, ::-2], np.asfortranarray(base)):
            self.assert_reference_bits(values, 1e-10)

    def test_finite_float_bit_patterns_and_rounding_overflow(self) -> None:
        generator = np.random.default_rng(822031)
        bits = generator.integers(0, np.iinfo(np.uint64).max, size=(80, 5), dtype=np.uint64)
        bits &= np.uint64(0xFFEFFFFFFFFFFFFF)
        values = bits.view(np.float64)
        values = np.concatenate((values, values[::-1]))
        self.assertTrue(np.all(np.isfinite(values)))
        for tolerance in (1.0, 1e-10, 1e-14):
            self.assert_reference_bits(values, tolerance)

    def test_nonfinite_rows_keep_numpy_axis_zero_nan_semantics(self) -> None:
        bits = np.asarray([
            [0x7FF8000000000001, 0x3FF0000000000000],
            [0x7FF8000000000001, 0x3FF0000000000000],
            [0x7FF8000000000002, 0x3FF0000000000000],
            [0x7FF0000000000000, 0x0000000000000000],
            [0x7FF0000000000000, 0x8000000000000000],
            [0xFFF0000000000000, 0x0000000000000000],
        ], dtype=np.uint64)
        self.assert_reference_bits(bits.view(np.float64), 1e-10)

    def test_other_dtypes_and_unusual_dimensions_use_legacy_semantics(self) -> None:
        ordinary = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        for values in (
            ordinary.astype(np.float32),
            ordinary.astype(np.int64),
            ordinary.astype(np.complex128),
            ordinary.astype(">f8"),
            ordinary[:, 0],
            ordinary.reshape(3, 1, 2),
            np.empty((0, 0)),
            np.empty((3, 0)),
        ):
            with self.subTest(dtype=str(values.dtype), shape=values.shape):
                self.assert_reference_bits(values, 1e-10)

    def test_only_finite_native_float64_matrices_use_byte_keys(self) -> None:
        original_unique = np.unique
        for values, expected_axis in (
            (np.asarray([[0.0, 1.0], [-0.0, 1.0]]), None),
            (np.asarray([[np.nan, 1.0], [np.nan, 1.0]]), 0),
            (np.asarray([[np.inf, 1.0], [np.inf, 1.0]]), 0),
            (np.asarray([[1.0, 2.0]], dtype=np.float32), 0),
            (np.asarray([1.0, 1.0]), 0),
            (np.empty((2, 0)), 0),
        ):
            with self.subTest(values=values, axis=expected_axis):
                with patch(
                    "multistep.src.polytope_geometry.np.unique", wraps=original_unique
                ) as unique:
                    _deduplicate_rows(values, 1e-10)
                self.assertEqual(unique.call_args.kwargs.get("axis"), expected_axis)


if __name__ == "__main__":
    unittest.main()
