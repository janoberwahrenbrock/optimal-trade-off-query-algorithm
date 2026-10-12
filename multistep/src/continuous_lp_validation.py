"""Private continuous-LP primal validation with exact SciPy fallback.

The specialized checker is used only with the caller's explicit private opt-in
and ordinary native arrays. All actual primal validation is retained. The only
removed operation computes a semi-continuous allowance masked by scalar False.
Source drift, checker overrides, helper overrides, and unusual results retain
the caller's actual original checker.

Checker body derived from SciPy's BSD-3-Clause _linprog_util._check_result.
The SciPy license is retained at SCIPY_CONTINUOUS_VALIDATION_LICENSE.txt.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Any

import numpy as np
try:
    from scipy.optimize._linprog_util import _check_result as _IMPORTED_CHECK_RESULT
except (ImportError, AttributeError):
    # The existing native adapter remains optional on incompatible SciPy.
    _IMPORTED_CHECK_RESULT = None

_F64 = np.dtype(np.float64)
_ORIGINAL_ISCLOSE = np.isclose
_ORIGINAL_CHECKER_CODE = getattr(_IMPORTED_CHECK_RESULT, "__code__", None)
_SUPPORTED_SOURCE_SHA256 = {
    "599e4d0380edaa1fb91d5de7d657266c36de3f92c0534811bece19c305cfa17b",
}


def _source_is_supported(checker: Any) -> bool:
    try:
        source = inspect.getsource(checker)
        return hashlib.sha256(source.encode()).hexdigest() in _SUPPORTED_SOURCE_SHA256
    except Exception:
        return False


_SOURCE_VERIFIED = _source_is_supported(_IMPORTED_CHECK_RESULT)


def check_private_continuous_result(checker, x, fun, status, slack, con,
                                    bounds, tol, message, integrality):
    if (
        not _SOURCE_VERIFIED
        or checker is not _IMPORTED_CHECK_RESULT
        or checker.__code__ is not _ORIGINAL_CHECKER_CODE
        or checker.__globals__.get("np") is not np
        or np.isclose is not _ORIGINAL_ISCLOSE
        or integrality is not None
        or type(status) is not int
        or type(message) is not str
        or type(fun) is not float
        or type(tol) is not float
        or tol != 1e-9
        or type(x) is not np.ndarray
        or x.dtype != _F64
        or x.ndim != 1
        or not x.flags.c_contiguous
        or x.strides[0] < 0
        or type(slack) is not np.ndarray
        or slack.dtype != _F64
        or slack.ndim != 1
        or not slack.flags.c_contiguous
        or slack.strides[0] < 0
        or type(con) is not np.ndarray
        or con.dtype != _F64
        or con.ndim != 1
        or not con.flags.c_contiguous
        or con.strides[0] < 0
        or type(bounds) is not np.ndarray
        or bounds.dtype != _F64
        or bounds.shape != (len(x), 2)
        or not bounds.flags.c_contiguous
        or bounds.strides[0] < 0
        or bounds.strides[1] < 0
    ):
        return checker(x, fun, status, slack, con, bounds, tol, message, integrality)
    return _continuous_check_result(x, fun, status, slack, con, bounds,
                                    tol, message, integrality)


def _continuous_check_result(x, fun, status, slack, con, bounds, tol, message,
                  integrality):
    """Retain SciPy primal validation for a private continuous LP result.

    Bounds, nonnegative inequality slack and zero equality residuals are
    checked within the unchanged tolerance. Only the dead semi-continuous
    zero-value allowance is omitted; status and message rules remain exact.
    """
    # Somewhat arbitrary
    tol = np.sqrt(tol) * 10

    if x is None:
        # HiGHS does not provide x if infeasible/unbounded
        if status == 0:  # Observed with HiGHS Simplex Primal
            status = 4
            message = ("The solver did not provide a solution nor did it "
                       "report a failure. Please submit a bug report.")
        return status, message

    contains_nans = (
        np.isnan(x).any()
        or np.isnan(fun)
        or np.isnan(slack).any()
        or np.isnan(con).any()
    )

    if contains_nans:
        is_feasible = False
    else:
        valid_bounds = (x >= bounds[:, 0] - tol) & (x <= bounds[:, 1] + tol)
        invalid_bounds = not np.all(valid_bounds)

        invalid_slack = status != 3 and (slack < -tol).any()
        invalid_con = status != 3 and (np.abs(con) > tol).any()
        is_feasible = not (invalid_bounds or invalid_slack or invalid_con)

    if status == 0 and not is_feasible:
        status = 4
        message = ("The solution does not satisfy the constraints within the "
                   "required tolerance of " + f"{tol:.2E}" + ", yet "
                   "no errors were raised and there is no certificate of "
                   "infeasibility or unboundedness. Check whether "
                   "the slack and constraint residuals are acceptable; "
                   "if not, consider enabling presolve, adjusting the "
                   "tolerance option(s), and/or using a different method. "
                   "Please consider submitting a bug report.")
    elif status == 2 and is_feasible:
        # Occurs if the simplex method exits after phase one with a very
        # nearly basic feasible solution. Postsolving can make the solution
        # basic, however, this solution is NOT optimal
        status = 4
        message = ("The solution is feasible, but the solver did not report "
                   "that the solution was optimal. Please try a different "
                   "method.")

    return status, message
