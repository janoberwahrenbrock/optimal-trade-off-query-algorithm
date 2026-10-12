from __future__ import annotations

"""Reuse exact equality decompositions across geometric calculations."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import NamedTuple

import numpy as np
from scipy.linalg import null_space


@dataclass(frozen=True)
class AffineDecomposition:
    particular_point: np.ndarray
    basis: np.ndarray
    equality_residual: float


class AffineCacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int
    maxbytes: int
    currbytes: int


_EqualityKey = tuple[
    int,
    tuple[tuple[int, ...], str] | None,
    bytes | None,
    tuple[tuple[int, ...], str] | None,
    bytes | None,
]
_AFFINE_CACHE_MAX_SIZE = 4_096
_AFFINE_CACHE_MAX_BYTES = 8 * 1024 * 1024
_affine_cache: OrderedDict[_EqualityKey, AffineDecomposition] = OrderedDict()
_affine_cache_lock = RLock()
_affine_cache_bytes = 0
_affine_cache_hits = 0
_affine_cache_misses = 0


def get_affine_decomposition(
    variable_count: int,
    equality_matrix: np.ndarray | None,
    equality_right_side: np.ndarray | None,
) -> AffineDecomposition:
    """Return the original least-squares point and orthonormal null space.

    The signature retains all equality coefficients, their right-hand sides,
    and the ambient dimension. No rounding, scaling, or rank approximation is
    added. Equality feasibility is deliberately checked by the caller using
    its requested tolerance; the decomposition itself has no tolerance input.
    """

    global _affine_cache_hits, _affine_cache_misses, _affine_cache_bytes
    if equality_matrix is None:
        key: _EqualityKey = (variable_count, None, None, None, None)
    else:
        assert equality_right_side is not None
        key = (
            variable_count,
            (equality_matrix.shape, equality_matrix.dtype.str),
            equality_matrix.tobytes(),
            (equality_right_side.shape, equality_right_side.dtype.str),
            equality_right_side.tobytes(),
        )
    with _affine_cache_lock:
        cached = _affine_cache.get(key)
        if cached is not None:
            _affine_cache.move_to_end(key)
            _affine_cache_hits += 1
            return cached
        _affine_cache_misses += 1

    result = _compute_affine_decomposition(
        variable_count, equality_matrix, equality_right_side
    )
    size = _entry_bytes(key, result)
    if size > _AFFINE_CACHE_MAX_BYTES:
        return result
    with _affine_cache_lock:
        # Another caller may have filled the same entry while we computed it.
        cached = _affine_cache.get(key)
        if cached is not None:
            _affine_cache.move_to_end(key)
            return cached
        _affine_cache[key] = result
        _affine_cache_bytes += size
        while (
            len(_affine_cache) > _AFFINE_CACHE_MAX_SIZE
            or _affine_cache_bytes > _AFFINE_CACHE_MAX_BYTES
        ):
            evicted_key, evicted = _affine_cache.popitem(last=False)
            _affine_cache_bytes -= _entry_bytes(evicted_key, evicted)
    return result


def _compute_affine_decomposition(
    variable_count: int,
    equality_matrix: np.ndarray | None,
    equality_right_side: np.ndarray | None,
) -> AffineDecomposition:
    if equality_matrix is None:
        point = np.zeros(variable_count, dtype=float)
        basis = np.eye(variable_count, dtype=float)
        residual = 0.0
    else:
        assert equality_right_side is not None
        point, *_ = np.linalg.lstsq(
            equality_matrix, equality_right_side, rcond=None
        )
        residual = float(
            np.max(np.abs(equality_matrix @ point - equality_right_side), initial=0.0)
        )
        basis = null_space(equality_matrix)
    # Bytes backing prevents callers from enabling writeability on a cached
    # array, while keeping exactly the values produced by the original math.
    point = _immutable_array(point)
    basis = _immutable_array(basis)
    return AffineDecomposition(point, basis, residual)


def clear_affine_geometry_cache() -> None:
    global _affine_cache_bytes, _affine_cache_hits, _affine_cache_misses
    with _affine_cache_lock:
        _affine_cache.clear()
        _affine_cache_bytes = 0
        _affine_cache_hits = 0
        _affine_cache_misses = 0


def affine_geometry_cache_info() -> AffineCacheInfo:
    """Report reuse; bytes include equality signatures and cached array data."""

    with _affine_cache_lock:
        return AffineCacheInfo(
            _affine_cache_hits,
            _affine_cache_misses,
            _AFFINE_CACHE_MAX_SIZE,
            len(_affine_cache),
            _AFFINE_CACHE_MAX_BYTES,
            _affine_cache_bytes,
        )


def _entry_bytes(key: _EqualityKey, result: AffineDecomposition) -> int:
    return (
        len(key[2] or b"")
        + len(key[4] or b"")
        + len(result.particular_point.base)
        + len(result.basis.base)
    )


def _immutable_array(array: np.ndarray) -> np.ndarray:
    # null_space may return a strided slice rather than a contiguous matrix.
    # Retain those strides so NumPy chooses the same matrix-multiplication
    # path and floating-point summation order as before caching.
    extent = sum(
        (dimension - 1) * abs(stride)
        for dimension, stride in zip(array.shape, array.strides)
        if dimension > 0
    )
    size = extent + array.dtype.itemsize if array.size else 0
    offset = sum(
        (dimension - 1) * -stride
        for dimension, stride in zip(array.shape, array.strides)
        if dimension > 0 and stride < 0
    )
    storage = bytearray(size)
    writable = np.ndarray(
        array.shape, dtype=array.dtype, buffer=storage,
        offset=offset, strides=array.strides,
    )
    writable[...] = array
    return np.ndarray(
        array.shape, dtype=array.dtype, buffer=bytes(storage),
        offset=offset, strides=array.strides,
    )
