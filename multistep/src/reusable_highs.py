"""Cold-solve HiGHS reuse for the small continuous LPs used by this project.

SciPy's bundled binding is used so the solver version stays identical to
``linprog``. This optional adapter deliberately accepts only the options used
here. Everything else, and incompatible SciPy installations, falls back to the
public API. Bases and solutions are cleared before every solve; this is setup
reuse, not a change to numerical warm-start behaviour.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import threading
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csc_array, issparse

from .continuous_lp_validation import check_private_continuous_result

try:
    from scipy.sparse import get_index_dtype as _get_index_dtype
except ImportError:
    # Sparse-array preparation is optional independently of the native adapter.
    _get_index_dtype = None

try:
    from scipy.optimize._highspy import _core as _h
    from scipy.optimize._linprog_highs import _highs_to_scipy_status_message
    from scipy.optimize._linprog_util import _LPProblem, _clean_inputs, _check_result
    _AVAILABLE = all(hasattr(_h._Highs, name) for name in (
        "clearSolver", "clearModel", "passOptions", "passModel",
        "run", "getInfo", "getSolution", "getBasis", "getModelStatus",
    ))
except (ImportError, AttributeError):
    _AVAILABLE = False

_MAX_MODELS = 32
_MAX_MODEL_BYTES = 4 * 1024 * 1024
_ORIGINAL_CLEAN_INPUTS = globals().get("_clean_inputs")
_LOCAL = threading.local()
_SUPPORTED_OPTIONS = {"primal_feasibility_tolerance", "dual_feasibility_tolerance",
                      "disp", "presolve"}
_SUPPORTED_KWARGS = {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds", "method", "options"}


class ReusableHighsCacheInfo(NamedTuple):
    calls: int
    model_hits: int
    model_misses: int
    solver_creations: int
    solver_reuses: int
    fallbacks: int
    currsize: int
    currbytes: int
    maxsize: int
    maxbytes: int


@dataclass
class _Model:
    solver: Any
    options_key: tuple[Any, ...]
    size: int
    template: Any


class _CSCData(NamedTuple):
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray


class _State:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.models: OrderedDict[Any, _Model] = OrderedDict()
        self.options: OrderedDict[Any, Any] = OrderedDict()
        self.bytes = 0
        self.calls = self.model_hits = self.model_misses = 0
        self.solver_creations = self.solver_reuses = self.fallbacks = 0


def _state() -> _State:
    state = getattr(_LOCAL, "state", None)
    if state is None or state.pid != os.getpid():
        state = _LOCAL.state = _State()
    return state


def clear_reusable_highs_cache() -> None:
    _LOCAL.state = _State()


def reusable_highs_cache_info() -> ReusableHighsCacheInfo:
    state = _state()
    return ReusableHighsCacheInfo(
        state.calls, state.model_hits, state.model_misses, state.solver_creations,
        state.solver_reuses, state.fallbacks, len(state.models), state.bytes,
        _MAX_MODELS, _MAX_MODEL_BYTES,
    )


def _supported(kwargs: dict[str, Any]) -> bool:
    if not _AVAILABLE or kwargs.keys() - _SUPPORTED_KWARGS:
        return False
    if kwargs.get("method", "highs") not in {"highs", "highs-ds", "highs-ipm"}:
        return False
    if any(issparse(kwargs.get(name)) for name in ("A_ub", "A_eq")):
        return False
    options = kwargs.get("options")
    if options is None:
        options = {}
    if not isinstance(options, dict) or options.keys() - _SUPPORTED_OPTIONS:
        return False
    if options.get("disp", False) is not False:
        return False
    if type(options.get("presolve", True)) is not bool:
        return False
    for name in ("primal_feasibility_tolerance", "dual_feasibility_tolerance"):
        value = options.get(name, 1e-7)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value) or value < 1e-10):
            return False
    return True


def _require_ok(status: Any) -> None:
    if status == _h.HighsStatus.kError:
        raise RuntimeError("HiGHS setup or execution failed")


def _options(state: _State, key: tuple[Any, ...]) -> Any:
    if key not in state.options:
        method, presolve, primal, dual = key
        options = _h.HighsOptions()
        options.presolve = "on" if presolve else "off"
        if method != "highs":
            options.solver = "simplex" if method == "highs-ds" else "ipm"
        options.primal_feasibility_tolerance = primal
        options.dual_feasibility_tolerance = dual
        options.output_flag = False
        options.log_to_console = False
        options.highs_debug_level = int(_h.HighsDebugLevel.kHighsDebugLevelNone)
        options.simplex_strategy = int(_h.simplex_constants.SimplexStrategy.kSimplexStrategyDual)
        state.options[key] = options
        while len(state.options) > _MAX_MODELS:
            state.options.popitem(last=False)
    state.options.move_to_end(key)
    return state.options[key]


def _array_key(array: np.ndarray) -> tuple[Any, ...]:
    return array.shape, array.dtype.str, array.tobytes()


def _replace_highs_infinities(array: np.ndarray) -> None:
    """Replace the same infinity sentinels after input cleaning, in-place."""
    if type(array) is np.ndarray and array.dtype == np.dtype(float):
        array[array == np.inf] = _h.kHighsInf
        array[array == -np.inf] = -_h.kHighsInf
    else:
        # Preserve NumPy dispatch and exceptions for unusual private inputs.
        array[np.isposinf(array)] = _h.kHighsInf
        array[np.isneginf(array)] = -_h.kHighsInf


def _constraint_matrix_csc(a_ub: np.ndarray, a_eq: np.ndarray) -> Any:
    """Prepare the same CSC arrays after SciPy has validated the dense inputs.

    The previous dense-to-COO-to-CSC conversion contains no duplicates: each
    nonzero is stored once, with ascending row indices within each column.
    Scanning the transpose obtains exactly that order without intermediate
    sparse objects or any arithmetic on matrix coefficients.
    """
    stacked = np.vstack((a_ub, a_eq))
    if _get_index_dtype is None or stacked.dtype != np.dtype(np.float64):
        return csc_array(stacked)
    transposed = stacked.T
    columns, rows = transposed.nonzero()
    index_dtype = _get_index_dtype(maxval=max(*stacked.shape, len(rows)))
    # nonzero coordinates can be strided views, including empty int64 arrays.
    # The previous compressed constructor owns a contiguous indices array.
    indices = np.array(rows, dtype=index_dtype, copy=True)
    indptr = np.empty(stacked.shape[1] + 1, dtype=index_dtype)
    indptr[0] = 0
    counts = np.bincount(columns, minlength=stacked.shape[1])
    np.cumsum(counts, dtype=index_dtype, out=indptr[1:])
    return _CSCData(transposed[columns, rows], indices, indptr)


def _problem(kwargs: dict[str, Any]) -> Any:
    return _LPProblem(
        kwargs["c"], kwargs.get("A_ub"), kwargs.get("b_ub"),
        kwargs.get("A_eq"), kwargs.get("b_eq"), kwargs.get("bounds"), None, None,
    )


def _clean_private_inputs(kwargs: dict[str, Any]) -> Any:
    """Same checks/copies for standard project arrays; no prepared-input cache."""
    if _clean_inputs is not _ORIGINAL_CLEAN_INPUTS:
        return _clean_inputs(_problem(kwargs))
    objective = kwargs["c"]
    if type(objective) is np.ndarray:
        ordinary = (objective.dtype == np.dtype(np.float64)
                    and objective.ndim == 1 and objective.flags.c_contiguous)
    else:
        ordinary = (type(objective) in (list, tuple)
                    and all(type(value) in (float, int) for value in objective))
    if not ordinary:
        return _clean_inputs(_problem(kwargs))
    try:
        c = np.array(objective, dtype=np.float64, copy=True).squeeze()
    except (TypeError, ValueError, OverflowError):
        return _clean_inputs(_problem(kwargs))
    if c.size == 1:
        c = c.reshape(-1)
    if c.ndim != 1 or c.size == 0 or not np.isfinite(c).all():
        return _clean_inputs(_problem(kwargs))
    variable_count = len(c)
    constraints = []
    for matrix_name, rhs_name in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        matrix, rhs = kwargs.get(matrix_name), kwargs.get(rhs_name)
        if matrix is None and rhs is None:
            constraints.extend((np.zeros((0, variable_count)), np.array([], dtype=float)))
            continue
        if not (
            type(matrix) is np.ndarray and type(rhs) is np.ndarray
            and matrix.dtype == rhs.dtype == np.dtype(np.float64)
            and matrix.ndim == 2 and matrix.shape[1] == variable_count
            and rhs.shape == (matrix.shape[0],)
            and matrix.flags.c_contiguous and rhs.flags.c_contiguous
            and np.isfinite(matrix).all() and np.isfinite(rhs).all()
        ):
            return _clean_inputs(_problem(kwargs))
        constraints.extend((matrix.copy(order="K"), rhs.copy(order="K")))
    bounds = kwargs.get("bounds")
    if bounds is None:
        cleaned_bounds = np.empty((variable_count, 2))
        cleaned_bounds[:, 0] = 0.0
        cleaned_bounds[:, 1] = np.inf
    elif type(bounds) is np.ndarray:
        if not (bounds.dtype == np.dtype(np.float64)
                and bounds.shape == (variable_count, 2)
                and bounds.flags.c_contiguous and np.isfinite(bounds).all()):
            return _clean_inputs(_problem(kwargs))
        cleaned_bounds = bounds.copy(order="K")
    elif type(bounds) in (list, tuple) and len(bounds) == variable_count:
        cleaned_bounds = np.empty((variable_count, 2))
        for index, row in enumerate(bounds):
            if type(row) not in (list, tuple) or len(row) != 2:
                return _clean_inputs(_problem(kwargs))
            for column, value in enumerate(row):
                if value is None:
                    cleaned_bounds[index, column] = -np.inf if column == 0 else np.inf
                elif type(value) in (float, int):
                    try:
                        number = float(value)
                    except (OverflowError, ValueError):
                        return _clean_inputs(_problem(kwargs))
                    if not np.isfinite(number):
                        return _clean_inputs(_problem(kwargs))
                    cleaned_bounds[index, column] = number
                else:
                    return _clean_inputs(_problem(kwargs))
    else:
        return _clean_inputs(_problem(kwargs))
    return _LPProblem(c, *constraints, cleaned_bounds, None, None)


def _solve(
    state: _State,
    kwargs: dict[str, Any],
    *,
    compact_result: bool = False,
    _internal_continuous_check: bool = False,
) -> OptimizeResult:
    # Only ordinary private continuous inputs have equivalent direct format
    # preparation. Every generic/invalid input retains the original cleaner.
    lp = _clean_private_inputs(kwargs) if (
        compact_result and _internal_continuous_check
    ) else _clean_inputs(_problem(kwargs))
    c, a_ub, b_ub, a_eq, b_eq, bounds, _, _ = lp
    options = kwargs.get("options") or {}
    options_key = (
        kwargs.get("method", "highs"), options.get("presolve", True),
        float(options.get("primal_feasibility_tolerance", 1e-7)),
        float(options.get("dual_feasibility_tolerance", 1e-7)),
    )
    # Literal array contents prevent stale models after caller-side mutation.
    key = (options_key, *(_array_key(array) for array in (a_ub, b_ub, a_eq, b_eq, bounds)))
    model = state.models.pop(key, None)
    if model is not None:
        state.model_hits += 1
        state.solver_reuses += 1
        # Keep the prepared matrix, but reload the pristine template. Merely
        # clearing the solver basis leaves internal model scaling from an
        # earlier objective and can change rounding or simplex iterations.
        _require_ok(model.solver.clearModel())
        model.template.col_cost_ = c
        _require_ok(model.solver.passModel(model.template))
    else:
        state.model_misses += 1
        matrix = _constraint_matrix_csc(a_ub, a_eq)
        rhs = np.concatenate((b_ub, b_eq))
        lhs = np.concatenate((np.full_like(b_ub, -np.inf), b_eq))
        lower, upper = bounds.T.copy()
        for array in (rhs, lhs, lower, upper):
            _replace_highs_infinities(array)
        size = sum(array.nbytes for array in (a_ub, b_ub, a_eq, b_eq, bounds,
                                             matrix.data, matrix.indices, matrix.indptr))
        # Recycle one native instance with the same options when a new model
        # replaces an old entry. clearModel retains options, unlike clear().
        recyclable = next((old_key for old_key, old in state.models.items()
                            if old.options_key == options_key), None)
        if recyclable is not None and (len(state.models) >= _MAX_MODELS or
                                      state.bytes + size > _MAX_MODEL_BYTES):
            old = state.models.pop(recyclable)
            state.bytes -= old.size
            solver = old.solver
            _require_ok(solver.clearModel())
            state.solver_reuses += 1
        else:
            solver = _h._Highs()
            _require_ok(solver.passOptions(_options(state, options_key)))
            state.solver_creations += 1
        native_lp = _h.HighsLp()
        native_lp.num_col_ = len(c)
        native_lp.num_row_ = len(rhs)
        native_lp.a_matrix_.num_col_ = len(c)
        native_lp.a_matrix_.num_row_ = len(rhs)
        native_lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
        native_lp.col_cost_ = c
        native_lp.col_lower_, native_lp.col_upper_ = lower, upper
        native_lp.row_lower_, native_lp.row_upper_ = lhs, rhs
        native_lp.a_matrix_.start_ = matrix.indptr
        native_lp.a_matrix_.index_ = matrix.indices
        native_lp.a_matrix_.value_ = matrix.data
        _require_ok(solver.passModel(native_lp))
        model = _Model(solver, options_key, size, native_lp)
        state.bytes += size
    # Do not carry bases, presolve state or a solution into the next LP.
    _require_ok(model.solver.clearSolver())
    if model.size <= _MAX_MODEL_BYTES:
        state.models[key] = model
        while len(state.models) > _MAX_MODELS or state.bytes > _MAX_MODEL_BYTES:
            _, old = state.models.popitem(last=False)
            state.bytes -= old.size
    else:
        state.bytes -= model.size
    _require_ok(model.solver.run())
    return _result(model.solver, lp, compact_result=compact_result,
                   _internal_continuous_check=_internal_continuous_check)


def _result(solver: Any, lp: Any, *, compact_result: bool = False,
            _internal_continuous_check: bool = False) -> OptimizeResult:
    c, _, b_ub, _, b_eq, bounds, _, _ = lp
    native_status = solver.getModelStatus()
    info = solver.getInfo()
    native_message = solver.modelStatusToString(native_status)
    x = fun = slack = con = marg_ineq = marg_eq = marg_lower = marg_upper = None
    if native_status == _h.HighsModelStatus.kOptimal:
        solution = solver.getSolution()
        x = np.asarray(solution.col_value)
        fun = info.objective_function_value
        all_slack = np.concatenate((b_ub, b_eq)) - solution.row_value
        slack, con = all_slack[:len(b_ub)], all_slack[len(b_ub):]
        if not compact_result:
            basis = solver.getBasis()
            dual = np.asarray(solution.row_dual)
            marg_ineq, marg_eq = dual[:len(b_ub)], dual[len(b_ub):]
            marg_lower, marg_upper = np.zeros(len(c)), np.zeros(len(c))
            for index, basis_status in enumerate(basis.col_status):
                if basis_status == _h.HighsBasisStatus.kLower:
                    marg_lower[index] = solution.col_dual[index]
                elif basis_status == _h.HighsBasisStatus.kUpper:
                    marg_upper[index] = solution.col_dual[index]
    else:
        native_message = (f"model_status is {native_message}; primal_status is "
                          f"{solver.solutionStatusToString(info.primal_solution_status)}")
    status, message = _highs_to_scipy_status_message(native_status, native_message)
    if compact_result and _internal_continuous_check:
        status, message = check_private_continuous_result(
            _check_result, x, fun, status, slack, con, bounds, 1e-9, message, None,
        )
    else:
        status, message = _check_result(x, fun, status, slack, con, bounds, 1e-9, message, None)
    if compact_result:
        # Production callers consume the primal solution and classified status
        # only. Keep SciPy's complete primal validation above, but avoid basis,
        # dual and nested-result work that none of those callers use.
        return OptimizeResult(
            x=x, fun=fun, slack=slack, con=con, status=status, message=message,
            success=status == 0, nit=info.simplex_iteration_count or info.ipm_iteration_count,
            crossover_nit=info.crossover_iteration_count,
        )
    lower, upper = bounds.T
    # Match SciPy's finite sentinel in bound residuals.
    lower = np.where(np.isneginf(lower), -_h.kHighsInf, lower)
    upper = np.where(np.isposinf(upper), _h.kHighsInf, upper)
    return OptimizeResult(
        x=x, fun=fun, slack=slack, con=con, status=status, message=message,
        success=status == 0, nit=info.simplex_iteration_count or info.ipm_iteration_count,
        crossover_nit=info.crossover_iteration_count,
        ineqlin=OptimizeResult(residual=slack, marginals=marg_ineq),
        eqlin=OptimizeResult(residual=con, marginals=marg_eq),
        lower=OptimizeResult(residual=None if x is None else x - lower, marginals=marg_lower),
        upper=OptimizeResult(residual=None if x is None else upper - x, marginals=marg_upper),
    )


def try_reusable_highs(
    kwargs: dict[str, Any],
    *,
    compact_result: bool = False,
    _internal_continuous_check: bool = False,
) -> OptimizeResult | None:
    """Return None when the original SciPy path should handle this call."""
    state = _state()
    state.calls += 1
    try:
        if not _supported(kwargs):
            state.fallbacks += 1
            return None
        return _solve(state, kwargs, compact_result=compact_result,
                      _internal_continuous_check=_internal_continuous_check)
    except Exception:
        # A version/API/model error must not make the optional fast path a
        # correctness dependency. Drop possibly modified native objects.
        state.models.clear()
        state.bytes = 0
        state.fallbacks += 1
        return None
