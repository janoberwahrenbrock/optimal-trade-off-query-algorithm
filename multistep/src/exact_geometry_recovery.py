"""Rare exact-rational recovery for guarded simplex states of dimension <=2.

Successful ordinary geometric calculations retain their original numerical path.
Explicit equalities are eliminated by exact RREF of original binary rational
coefficients. Convex intervals/polygons are clipped using every original row;
no sampling, probability threshold, candidate shortcut or radius cutoff is used.
Unsupported dimensions and unresolved degeneracy remain explicit errors.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
import math
from typing import Any


class RecoveryNotApplicable(ValueError):
    pass


class RecoveryGeometryError(RuntimeError):
    pass


def _fraction(value: float) -> Fraction:
    if not math.isfinite(float(value)):
        raise ValueError("finite coefficients are required")
    return Fraction.from_float(float(value))


def _text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _eliminate_equalities(rows, rhs, variable_count):
    matrix = [list(row) + [value] for row, value in zip(rows, rhs)]
    pivots = []
    active_row = 0
    for column in range(variable_count):
        pivot = next((i for i in range(active_row, len(matrix)) if matrix[i][column]), None)
        if pivot is None:
            continue
        matrix[active_row], matrix[pivot] = matrix[pivot], matrix[active_row]
        divisor = matrix[active_row][column]
        matrix[active_row] = [entry / divisor for entry in matrix[active_row]]
        for i in range(len(matrix)):
            if i != active_row and matrix[i][column]:
                multiplier = matrix[i][column]
                matrix[i] = [a - multiplier * b for a, b in zip(matrix[i], matrix[active_row])]
        pivots.append(column)
        active_row += 1
    if any(not any(row[:-1]) and row[-1] for row in matrix):
        raise RecoveryGeometryError("explicit equalities are exactly inconsistent")
    free = [column for column in range(variable_count) if column not in pivots]
    dimension = len(free)
    constants = [Fraction(0)] * variable_count
    basis = [[Fraction(0)] * dimension for _ in range(variable_count)]
    for j, column in enumerate(free):
        basis[column][j] = Fraction(1)
    for i, column in enumerate(pivots):
        constants[column] = matrix[i][-1]
        basis[column] = [-matrix[i][free_column] for free_column in free]
    return constants, basis, free


def _reduce(row, rhs, constants, basis):
    dimension = len(basis[0])
    reduced = [sum(a * coefficients[j] for a, coefficients in zip(row, basis)) for j in range(dimension)]
    return reduced, rhs - sum(a * value for a, value in zip(row, constants))


def _clip_polygon(polygon, coefficients, rhs):
    if not polygon:
        return []
    result = []
    previous = polygon[-1]
    previous_value = sum(a * x for a, x in zip(coefficients, previous)) - rhs
    for current in polygon:
        current_value = sum(a * x for a, x in zip(coefficients, current)) - rhs
        previous_inside, current_inside = previous_value <= 0, current_value <= 0
        if previous_inside != current_inside:
            amount = previous_value / (previous_value - current_value)
            result.append(tuple(a + amount * (b - a) for a, b in zip(previous, current)))
        if current_inside:
            result.append(current)
        previous, previous_value = current, current_value
    deduplicated = []
    for point in result:
        if not deduplicated or deduplicated[-1] != point:
            deduplicated.append(point)
    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    return deduplicated


def _area(polygon):
    if len(polygon) < 3:
        return Fraction(0)
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(polygon, polygon[1:] + polygon[:1]))) / 2


def _clip_interval(interval, coefficient, rhs):
    lower, upper = interval
    if coefficient > 0:
        upper = min(upper, rhs / coefficient)
    elif coefficient < 0:
        lower = max(lower, rhs / coefficient)
    elif rhs < 0:
        return Fraction(1), Fraction(0)
    return lower, upper


def _intrinsic_volume(measure, basis):
    dimension = len(basis[0])
    if dimension == 1:
        determinant = sum(row[0] ** 2 for row in basis)
    else:
        aa = sum(row[0] ** 2 for row in basis)
        ab = sum(row[0] * row[1] for row in basis)
        bb = sum(row[1] ** 2 for row in basis)
        determinant = aa * bb - ab * ab
    if determinant <= 0:
        raise RecoveryGeometryError("affine basis has no positive metric determinant")
    # Compute the product before conversion, so an underflowed projected area
    # and a large Jacobian cannot falsely turn a representable volume into zero.
    with localcontext() as context:
        context.prec = 80
        projected = Decimal(measure.numerator) / Decimal(measure.denominator)
        squared_jacobian = Decimal(determinant.numerator) / Decimal(determinant.denominator)
        intrinsic = float(projected * squared_jacobian.sqrt())
    return intrinsic, determinant


def recover_query_probabilities(constraint_key, query: dict[str, Any]) -> dict[str, Any]:
    inequalities, inequality_rhs, equalities, equality_rhs = constraint_key
    variable_count = len(equalities[0]) if equalities else len(inequalities[0]) if inequalities else 0
    if not variable_count:
        raise RecoveryNotApplicable("the system has no variables")
    if len(inequalities) != len(inequality_rhs) or len(equalities) != len(equality_rhs):
        raise ValueError("constraint row counts differ")
    if any(len(row) != variable_count for row in inequalities + equalities):
        raise ValueError("constraint dimensions differ")
    A, b = [[_fraction(value) for value in row] for row in inequalities], [_fraction(value) for value in inequality_rhs]
    E, e = [[_fraction(value) for value in row] for row in equalities], [_fraction(value) for value in equality_rhs]
    simplex = any(row[0] and all(value == row[0] for value in row) and rhs == row[0] for row, rhs in zip(E, e))
    nonnegative = all(any(
        row[column] < 0 and rhs <= 0 and all(value == 0 for j, value in enumerate(row) if j != column)
        for row, rhs in zip(A, b)
    ) for column in range(variable_count))
    if not simplex or not nonnegative:
        raise RecoveryNotApplicable("requires explicit sum(weights)=1 and nonnegative weights")
    constants, basis, free = _eliminate_equalities(E, e, variable_count)
    dimension = len(free)
    if dimension > 2:
        raise RecoveryNotApplicable("exact rational prototype supports intrinsic dimension at most two")
    a, c = int(query["ziel_index_a"]), int(query["ziel_index_b"])
    if a == c or not (0 <= a < variable_count and 0 <= c < variable_count):
        raise ValueError("query goal indices are invalid")
    objective = [Fraction(0)] * variable_count
    objective[a], objective[c] = Fraction(1), -_fraction(query["value"])
    cut, cut_rhs = _reduce(objective, Fraction(0), constants, basis)
    if dimension == 0:
        if any(sum(a * w for a, w in zip(row, constants)) > rhs for row, rhs in zip(A, b)):
            raise RecoveryGeometryError("the equality-defined point violates an inequality")
        value = sum(a * w for a, w in zip(objective, constants))
        answer = "<" if value < 0 else ">" if value > 0 else "="
        return {"dimension": 0, "probabilities": {option: float(option == answer) for option in ("<", "=", ">")}, "exact_probabilities": {option: _text(Fraction(option == answer)) for option in ("<", "=", ">")}}
    reduced = [_reduce(row, rhs, constants, basis) for row, rhs in zip(A, b)]
    if dimension == 1:
        parent = Fraction(0), Fraction(1)
        for row, rhs in reduced:
            parent = _clip_interval(parent, row[0], rhs)
        parent_measure = max(Fraction(0), parent[1] - parent[0])
        left = _clip_interval(parent, cut[0], cut_rhs)
        right = _clip_interval(parent, -cut[0], -cut_rhs)
        less = max(Fraction(0), left[1] - left[0])
        greater = max(Fraction(0), right[1] - right[0])
        parent_vertices = [(value,) for value in parent]
    else:
        parent = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(1), Fraction(1)), (Fraction(0), Fraction(1))]
        for row, rhs in reduced:
            parent = _clip_polygon(parent, row, rhs)
        parent_measure = _area(parent)
        less = _area(_clip_polygon(parent, cut, cut_rhs))
        greater = _area(_clip_polygon(parent, [-value for value in cut], -cut_rhs))
        parent_vertices = parent
    if parent_measure <= 0:
        raise RecoveryGeometryError("parent has exactly zero measure in its explicit affine hull")
    if not any(cut) and cut_rhs == 0:
        exact = {"<": Fraction(0), "=": Fraction(1), ">": Fraction(0)}
    else:
        if less + greater != parent_measure:
            raise RecoveryGeometryError("exact child measures do not partition the parent")
        exact = {"<": less / parent_measure, "=": Fraction(0), ">": greater / parent_measure}
    probabilities = {answer: float(value) for answer, value in exact.items()}
    if any(value > 0 and probabilities[answer] == 0 for answer, value in exact.items()):
        raise RecoveryGeometryError("a positive probability cannot be represented by the float result contract")
    intrinsic_volume, metric_determinant = _intrinsic_volume(parent_measure, basis)
    return {
        "dimension": dimension,
        "free_original_weight_coordinates": free,
        "probabilities": probabilities,
        "exact_probabilities": {answer: _text(value) for answer, value in exact.items()},
        "exact_parent_projected_measure": _text(parent_measure),
        "exact_less_projected_measure": _text(less),
        "exact_greater_projected_measure": _text(greater),
        "parent_intrinsic_volume_float": intrinsic_volume,
        "parent_intrinsic_volume_underflows_float": intrinsic_volume == 0,
        "affine_metric_determinant": _text(metric_determinant),
        "parent_vertices_in_free_coordinates": [[_text(value) for value in vertex] for vertex in parent_vertices],
    }


def recover_singleton_after_empty_prefilter(
    child_constraint_key, alternatives, parent_candidate_subset,
    filtered_child_subset, positive_branch_probability,
):
    if filtered_child_subset != []:
        raise RecoveryNotApplicable("requires an originally empty filtered child subset")
    if not math.isfinite(positive_branch_probability) or not 0 < positive_branch_probability <= 1:
        raise RecoveryNotApplicable("requires an already-confirmed positive branch probability")
    candidates = list(parent_candidate_subset or ())
    if not candidates or len(set(candidates)) != len(candidates):
        raise RecoveryNotApplicable("requires a nonempty distinct parent candidate subset")
    if any(type(candidate) is not int or not 0 <= candidate < len(alternatives) for candidate in candidates):
        raise ValueError("parent candidate indices are invalid")
    n = len(alternatives[0])
    if n < 2 or any(len(row) != n for row in alternatives):
        raise ValueError("alternatives must have at least two consistent goal columns")
    parent = recover_query_probabilities(child_constraint_key, {
        "ziel_index_a": 0, "ziel_index_b": 1, "value": 1.,
    })
    dimension = parent["dimension"]
    if dimension not in (1, 2):
        raise RecoveryNotApplicable("singleton proof requires positive intrinsic dimension one or two")
    A, b, E, e = child_constraint_key
    A = [[_fraction(value) for value in row] for row in A]
    b = [_fraction(value) for value in b]
    E = [[_fraction(value) for value in row] for row in E]
    e = [_fraction(value) for value in e]
    constants, basis, free = _eliminate_equalities(E, e, n)
    polygon = [tuple(Fraction(value) for value in point) for point in parent["parent_vertices_in_free_coordinates"]]
    if dimension == 1:
        interval = min(point[0] for point in polygon), max(point[0] for point in polygon)
    measure = Fraction(parent["exact_parent_projected_measure"])
    regions = []
    feasible = []
    for candidate in candidates:
        region = list(polygon) if dimension == 2 else interval
        for other in range(len(alternatives)):
            if other == candidate:
                continue
            # Match the existing utility-winning row construction, including
            # binary rounding of each float subtraction, then eliminate exactly.
            row = [_fraction(float(a) - float(c)) for a, c in zip(alternatives[other], alternatives[candidate])]
            coefficients, rhs = _reduce(row, Fraction(0), constants, basis)
            region = (_clip_polygon(region, coefficients, rhs) if dimension == 2
                      else _clip_interval(region, coefficients[0], rhs))
        closed_feasible = bool(region) if dimension == 2 else region[0] <= region[1]
        region_measure = _area(region) if dimension == 2 else max(Fraction(0), region[1] - region[0])
        regions.append({"candidate": candidate, "closed_feasible": closed_feasible,
                        "exact_projected_measure": _text(region_measure),
                        "exact_share_of_child": _text(region_measure / measure)})
        if closed_feasible:
            feasible.append(candidate)
    if not feasible:
        raise RecoveryGeometryError("positive child has no exact winning region in the parent candidate subset")
    if len(feasible) != 1:
        raise RecoveryNotApplicable("multiple closed-feasible candidates remain; full-state recovery is required")
    winner = feasible[0]
    winning_region = next(region for region in regions if region["candidate"] == winner)
    if Fraction(winning_region["exact_projected_measure"]) != measure:
        raise RecoveryNotApplicable("sole passed candidate does not cover the complete child volume")
    vertices = [
        [constant + sum(a * coordinate for a, coordinate in zip(row, point))
         for constant, row in zip(constants, basis)]
        for point in polygon
    ]
    minimum_margin = None
    for point in vertices:
        winner_utility = sum(_fraction(value) * weight for value, weight in zip(alternatives[winner], point))
        for other, row in enumerate(alternatives):
            if other == winner:
                continue
            margin = winner_utility - sum(_fraction(value) * weight for value, weight in zip(row, point))
            minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
    if minimum_margin is None or minimum_margin <= 0:
        raise RecoveryNotApplicable("global strict dominance is unproved; zero-measure ties must not be discarded")
    intrinsic_volume = parent["parent_intrinsic_volume_float"]
    if not intrinsic_volume > 0:
        raise RecoveryGeometryError("positive child intrinsic volume underflows the float metadata contract")
    return {
        "candidate_analysis_fields": {
            "candidates": [winner],
            "ratio_intervals": None,
            "candidate_volumes": {winner: intrinsic_volume},
            "candidate_volume_shares": {winner: 1.},
        },
        "dimension": dimension,
        "checked_parent_candidates": candidates,
        "exact_candidate_regions": regions,
        "exact_child_projected_measure": _text(measure),
        "strict_global_winner": winner,
        "minimum_exact_utility_margin_at_all_child_vertices": _text(minimum_margin),
        "minimum_utility_margin_float": float(minimum_margin),
        "positive_branch_probability_retained": positive_branch_probability,
        "uses_target_weights": False,
    }


def _system_key(system):
    return (tuple(tuple(row) for row in system.inequalities_left_side),
            tuple(system.inequalities_right_side),
            tuple(tuple(row) for row in system.equalities_left_side),
            tuple(system.equalities_right_side))


def recover_probabilities_from_system(system, query):
    """Called only after one of the existing nonpositive-volume failures."""
    return recover_query_probabilities(_system_key(system), query.model_dump())["probabilities"]


def recover_strict_singleton_from_systems(*, parent_system, child_system,
                                         alternatives, parent_candidate_subset,
                                         filtered_child_subset, probability,
                                         query, answer):
    """Certify a genuine singleton only for an empty positive child prefilter."""
    if filtered_child_subset != [] or not 0 < probability <= 1:
        raise RecoveryNotApplicable("requires empty prefilter and positive branch")
    parent = recover_query_probabilities(_system_key(parent_system), query.model_dump())
    if parent["dimension"] not in (1, 2):
        raise RecoveryNotApplicable("requires positive one/two-dimensional parent")
    if Fraction(parent["exact_probabilities"][answer]) <= 0:
        raise RecoveryGeometryError("the original parent does not support this branch exactly")
    result = recover_singleton_after_empty_prefilter(
        _system_key(child_system), alternatives.entries, parent_candidate_subset,
        filtered_child_subset, probability,
    )
    result["exact_parent_branch_probability"] = parent["exact_probabilities"][answer]
    return result["candidate_analysis_fields"]


def recover_intrinsic_volume(constraint_key):
    """Certify an ambiguous native zero/error in the explicit affine measure.

    A line cut out by inequalities in an explicit two-dimensional affine hull
    still has zero area: this function never replaces area with line length.
    Ordinary positive native volumes never call it.
    """
    inequalities, inequality_rhs, equalities, equality_rhs = constraint_key
    n = len(equalities[0]) if equalities else len(inequalities[0]) if inequalities else 0
    if not n:
        raise RecoveryNotApplicable("the system has no variables")
    # rank(E) <= row_count(E): a full ordinary5D simplex is rejected before
    # any Fraction conversion or clipping, with no change to its native path.
    if n > 3 and len(equalities) < n - 2:
        raise RecoveryNotApplicable("explicit equality rank cannot leave affine dimension at most two")
    if len(inequalities) != len(inequality_rhs) or len(equalities) != len(equality_rhs):
        raise ValueError("constraint row counts differ")
    if any(len(row) != n for row in inequalities + equalities):
        raise ValueError("constraint dimensions differ")
    A = [[_fraction(v) for v in row] for row in inequalities]
    b = [_fraction(v) for v in inequality_rhs]
    E = [[_fraction(v) for v in row] for row in equalities]
    e = [_fraction(v) for v in equality_rhs]
    simplex = any(row[0] and all(v == row[0] for v in row) and rhs == row[0] for row, rhs in zip(E, e))
    nonnegative = all(any(
        row[i] < 0 and rhs <= 0 and all(v == 0 for j, v in enumerate(row) if j != i)
        for row, rhs in zip(A, b)
    ) for i in range(n))
    if not simplex or not nonnegative:
        raise RecoveryNotApplicable("requires explicit sum(weights)=1 and nonnegative weights")
    try:
        constants, basis, free = _eliminate_equalities(E, e, n)
    except RecoveryGeometryError:
        # Exact RREF has certified inconsistent equalities: the region is
        # genuinely empty, and the existing zero-volume convention is retained.
        return {"dimension": None, "exact_projected_measure": "0/1",
                "intrinsic_volume_float": 0., "exactly_inconsistent_equalities": True,
                "vertices_in_free_coordinates": []}
    dimension = len(free)
    if dimension > 2:
        raise RecoveryNotApplicable("exact intrinsic-volume recovery supports affine dimension at most two")
    if dimension == 0:
        # Preserve the existing volume convention for explicit points. Query
        # probabilities have their separate deterministic-point contract.
        return {"dimension": 0, "exact_projected_measure": "0/1",
                "intrinsic_volume_float": 0., "vertices_in_free_coordinates": []}
    reduced = [_reduce(row, rhs, constants, basis) for row, rhs in zip(A, b)]
    if dimension == 1:
        region = Fraction(0), Fraction(1)
        for row, rhs in reduced:
            region = _clip_interval(region, row[0], rhs)
        measure = max(Fraction(0), region[1] - region[0])
        vertices = [(v,) for v in region] if region[0] <= region[1] else []
    else:
        region = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
                  (Fraction(1), Fraction(1)), (Fraction(0), Fraction(1))]
        for row, rhs in reduced:
            region = _clip_polygon(region, row, rhs)
        measure, vertices = _area(region), region
    if measure == 0:
        intrinsic, determinant = 0., None
    else:
        intrinsic, determinant = _intrinsic_volume(measure, basis)
        if not math.isfinite(intrinsic) or intrinsic <= 0:
            raise RecoveryGeometryError("positive exact intrinsic volume is outside the finite positive float contract")
    return {"dimension": dimension, "free_original_weight_coordinates": free,
            "exact_projected_measure": _text(measure),
            "intrinsic_volume_float": intrinsic,
            "affine_metric_determinant": _text(determinant) if determinant is not None else None,
            "vertices_in_free_coordinates": [[_text(v) for v in point] for point in vertices]}


def recover_intrinsic_volume_from_system(system):
    return recover_intrinsic_volume(_system_key(system))["intrinsic_volume_float"]
