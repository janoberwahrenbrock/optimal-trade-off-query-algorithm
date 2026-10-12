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


def _solve(state: _State, kwargs: dict[str, Any]) -> OptimizeResult:
    # Keep SciPy's validation and default bounds, including rejection of NaNs.
    lp = _clean_inputs(_LPProblem(
        kwargs["c"], kwargs.get("A_ub"), kwargs.get("b_ub"),
        kwargs.get("A_eq"), kwargs.get("b_eq"), kwargs.get("bounds"), None, None,
    ))
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
        matrix = csc_array(np.vstack((a_ub, a_eq)))
        rhs = np.concatenate((b_ub, b_eq))
        lhs = np.concatenate((np.full_like(b_ub, -np.inf), b_eq))
        lower, upper = bounds.T.copy()
        for array in (rhs, lhs, lower, upper):
            array[np.isposinf(array)] = _h.kHighsInf
            array[np.isneginf(array)] = -_h.kHighsInf
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
    return _result(model.solver, lp)


def _result(solver: Any, lp: Any) -> OptimizeResult:
    c, _, b_ub, _, b_eq, bounds, _, _ = lp
    native_status = solver.getModelStatus()
    info = solver.getInfo()
    native_message = solver.modelStatusToString(native_status)
    x = fun = slack = con = marg_ineq = marg_eq = marg_lower = marg_upper = None
    if native_status == _h.HighsModelStatus.kOptimal:
        solution, basis = solver.getSolution(), solver.getBasis()
        x = np.asarray(solution.col_value)
        fun = info.objective_function_value
        all_slack = np.concatenate((b_ub, b_eq)) - solution.row_value
        slack, con = all_slack[:len(b_ub)], all_slack[len(b_ub):]
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
    status, message = _check_result(x, fun, status, slack, con, bounds, 1e-9, message, None)
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


def try_reusable_highs(kwargs: dict[str, Any]) -> OptimizeResult | None:
    """Return None when the original SciPy path should handle this call."""
    state = _state()
    state.calls += 1
    try:
        if not _supported(kwargs):
            state.fallbacks += 1
            return None
        return _solve(state, kwargs)
    except Exception:
        # A version/API/model error must not make the optional fast path a
        # correctness dependency. Drop possibly modified native objects.
        state.models.clear()
        state.bytes = 0
        state.fallbacks += 1
        return None
