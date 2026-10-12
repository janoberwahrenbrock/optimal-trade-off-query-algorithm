from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np

from multistep.src import polytope_volume
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import Query
from multistep.src.weight_space import build_weight_space


def _reference_copy_based_probabilities(
    weight_space: LinearConstraintSystem,
    query: Query,
    tolerance: float,
) -> dict[str, float]:
    """The probability calculation from commit 074fbf9, before branch keys.

    Keep its copy/add/public-volume path as an independent reference; the
    geometric, LP and normalization operations are intentionally unchanged.
    """

    objective = [0.0] * weight_space.variable_count
    objective[int(query.ziel_index_a)] = 1.0
    objective[int(query.ziel_index_b)] = -float(query.value)
    support = polytope_volume._query_support_from_cached_vertices(
        weight_space, objective, tolerance
    )
    if support is None:
        lower = weight_space.minimize(objective)
        upper = weight_space.maximize(objective)
        if lower.status != "optimal" or upper.status != "optimal":
            raise RuntimeError("cannot determine exact query support")
        if lower.optimal_value is None or upper.optimal_value is None:
            raise RuntimeError("query support optimization has no value")
        lower_value, upper_value = float(lower.optimal_value), float(upper.optimal_value)
    else:
        lower_value, upper_value = support
    if lower_value >= -tolerance and upper_value <= tolerance:
        return {"<": 0.0, "=": 1.0, ">": 0.0}
    if lower_value >= -tolerance:
        return {"<": 0.0, "=": 0.0, ">": 1.0}
    if upper_value <= tolerance:
        return {"<": 1.0, "=": 0.0, ">": 0.0}

    total_volume = polytope_volume.compute_polytope_intrinsic_volume(
        weight_space, tolerance=tolerance
    )
    if total_volume <= 0.0:
        raise RuntimeError("weight space has no positive intrinsic volume")
    less_system = LinearConstraintSystem()
    less_system.add_constraint_system(weight_space)
    less_system.add_inequality(objective, 0.0)
    greater_system = LinearConstraintSystem()
    greater_system.add_constraint_system(weight_space)
    greater_system.add_inequality([-value for value in objective], 0.0)
    less_volume = polytope_volume.compute_polytope_intrinsic_volume(
        less_system, tolerance=tolerance
    )
    greater_volume = polytope_volume.compute_polytope_intrinsic_volume(
        greater_system, tolerance=tolerance
    )
    raw = np.asarray([less_volume, greater_volume], dtype=float) / total_volume
    raw = np.maximum(raw, 0.0)
    probability_sum = float(np.sum(raw))
    if probability_sum <= 0.0:
        raise RuntimeError("exact query branches have zero total probability")
    raw /= probability_sum
    return {"<": float(raw[0]), "=": 0.0, ">": float(raw[1])}


def _key_hex(key):
    inequalities, inequality_rhs, equalities, equality_rhs = key
    return (
        tuple(tuple(value.hex() for value in row) for row in inequalities),
        tuple(value.hex() for value in inequality_rhs),
        tuple(tuple(value.hex() for value in row) for row in equalities),
        tuple(value.hex() for value in equality_rhs),
    )


class ProbabilityBranchKeyTests(unittest.TestCase):
    def tearDown(self) -> None:
        polytope_volume.clear_polytope_volume_cache()

    def test_branch_signatures_preserve_order_signed_zero_and_exact_float_bits(self) -> None:
        system = build_weight_space(5, [])
        system.add_inequality([-0.0, 1.0, 0.0, -0.0, 0.0], 0.9)
        system.add_inequality([0.0, 0.0, -0.0, -1.0, 0.0], -0.0)
        query = Query(ziel_index_a=2, ziel_index_b=4, value=1.2345678901234567)
        objective = [0.0, 0.0, 1.0, 0.0, -query.value]
        expected = [polytope_volume._constraint_system_key(system)]
        for row in (objective, [-value for value in objective]):
            branch = LinearConstraintSystem()
            branch.add_constraint_system(system)
            branch.add_inequality(row, 0.0)
            expected.append(polytope_volume._constraint_system_key(branch))
        snapshot = copy.deepcopy(system.model_dump())
        with patch.object(
            polytope_volume,
            "_constraint_system_key",
            wraps=polytope_volume._constraint_system_key,
        ) as key_builder, patch.object(
            polytope_volume,
            "_compute_polytope_intrinsic_volume_cached",
            side_effect=(2.0, 1.0, 1.0),
        ) as volume_solver, patch.object(
            LinearConstraintSystem,
            "add_constraint_system",
            side_effect=AssertionError("branch copies should be unnecessary"),
        ):
            probabilities = polytope_volume.compute_exact_query_answer_probabilities(
                system, query
            )
        self.assertEqual(key_builder.call_count, 1)
        self.assertEqual(probabilities, {"<": 0.5, "=": 0.0, ">": 0.5})
        self.assertEqual(
            [_key_hex(call.args[0]) for call in volume_solver.call_args_list],
            [_key_hex(key) for key in expected],
        )
        self.assertEqual(system.model_dump(), snapshot)

    def test_probabilities_and_cache_behavior_match_copy_based_reference_bitwise(self) -> None:
        rng = np.random.default_rng(20261012)
        states = []
        for goal_count in (3, 5):
            states.append(build_weight_space(goal_count, []))
            target = rng.uniform(0.5, 1.5, size=goal_count)
            target /= target.sum()
            answers = []
            for index in range(goal_count + 2):
                a, b = (int(i) for i in rng.choice(goal_count, 2, replace=False))
                factor = 0.6 if index % 2 == 0 else 1.4
                query = Query(
                    ziel_index_a=a,
                    ziel_index_b=b,
                    value=float(target[a] / target[b] * factor),
                )
                answers.append(query.answer(">" if factor < 1.0 else "<"))
            states.append(build_weight_space(goal_count, answers))
            states.append(build_weight_space(
                goal_count,
                [Query(ziel_index_a=0, ziel_index_b=1, value=2.0).answer("=")],
            ))
        forced = build_weight_space(3, [])
        forced.add_inequality([-1.0, 0.0, 0.0], -0.75)
        states.append(forced)
        states.append(build_weight_space(
            3, [Query(ziel_index_a=0, ziel_index_b=1, value=1.0).answer(">")]
        ))

        comparisons = 0
        for state_index, system in enumerate(states):
            snapshot = copy.deepcopy(system.model_dump())
            pairs = ((0, 1), (system.variable_count - 1, 0))
            for a, b in pairs:
                for ratio in (0.0, 0.001, 0.25, 1.0, 1.0 + 5e-9, 2.0, 1000.0):
                    query = Query(ziel_index_a=a, ziel_index_b=b, value=ratio)
                    for cache_state in ("cold", "warm", "volume_only"):
                        for tolerance in (1e-10, 1e-7):
                            with self.subTest(
                                state=state_index, query=query, cache=cache_state,
                                tolerance=tolerance,
                            ):
                                reference, reference_caches = self._run(
                                    _reference_copy_based_probabilities,
                                    system, query, tolerance, cache_state,
                                )
                                actual, actual_caches = self._run(
                                    polytope_volume.compute_exact_query_answer_probabilities,
                                    system, query, tolerance, cache_state,
                                )
                                self.assertEqual(
                                    {answer: value.hex() for answer, value in actual.items()},
                                    {answer: value.hex() for answer, value in reference.items()},
                                )
                                self.assertEqual(actual_caches, reference_caches)
                                self.assertEqual(system.model_dump(), snapshot)
                                comparisons += 1
        self.assertEqual(comparisons, 672)

    @staticmethod
    def _run(function, system, query, tolerance, cache_state):
        polytope_volume.clear_polytope_volume_cache()
        if cache_state != "cold":
            polytope_volume.compute_polytope_intrinsic_volume(system, tolerance=tolerance)
        if cache_state == "volume_only":
            with polytope_volume._vertex_cache_lock:
                polytope_volume._vertex_cache.clear()
                polytope_volume._vertex_cache_bytes = 0
        result = function(system, query, tolerance)
        return result, (
            polytope_volume.polytope_volume_cache_info(),
            polytope_volume.polytope_vertex_cache_info(),
        )


if __name__ == "__main__":
    unittest.main()
