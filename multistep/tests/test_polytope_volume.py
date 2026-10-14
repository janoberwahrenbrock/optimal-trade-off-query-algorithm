from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from multistep.src import polytope_volume
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import Query
from multistep.src.polytope_geometry import (
    PolytopeVertices,
    enumerate_polytope_vertices,
)
from multistep.src.polytope_volume import (
    clear_polytope_volume_cache,
    compute_exact_query_answer_probabilities,
    compute_polytope_intrinsic_volume,
    polytope_volume_cache_info,
)
from multistep.src.weight_space import build_weight_space


class PolytopeVolumeTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_polytope_volume_cache()

    def test_free_simplex_query_matches_analytic_probability(self) -> None:
        weight_space = build_weight_space(goal_count=7, answered_queries=[])

        for ratio in (0.1, 0.5, 1.0, 2.0, 10.0):
            with self.subTest(ratio=ratio):
                probabilities = compute_exact_query_answer_probabilities(
                    weight_space=weight_space,
                    query=Query(ziel_index_a=0, ziel_index_b=1, value=ratio),
                )
                self.assertAlmostEqual(
                    probabilities["<"],
                    ratio / (1.0 + ratio),
                    places=8,
                )
                self.assertAlmostEqual(
                    probabilities[">"],
                    1.0 / (1.0 + ratio),
                    places=8,
                )
                self.assertEqual(probabilities["="], 0.0)
                self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_volume_results_are_cached_by_constraints(self) -> None:
        weight_space = build_weight_space(goal_count=3, answered_queries=[])

        first = compute_polytope_intrinsic_volume(weight_space)
        second = compute_polytope_intrinsic_volume(weight_space)

        self.assertGreater(first, 0.0)
        self.assertEqual(first, second)
        self.assertEqual(polytope_volume_cache_info().hits, 1)

    def test_cold_forced_answer_keeps_two_lps_without_enumerating_vertices(self) -> None:
        weight_space = build_weight_space(goal_count=3, answered_queries=[])
        weight_space.add_inequality([-1.0, 0.0, 0.0], -0.6)
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        with self._record_support_lps() as calls, patch.object(
            polytope_volume,
            "enumerate_polytope_vertices",
            side_effect=AssertionError("a forced cold answer needs no geometry"),
        ):
            probabilities = compute_exact_query_answer_probabilities(weight_space, query)

        self.assertEqual(probabilities, {"<": 0.0, "=": 0.0, ">": 1.0})
        self.assertEqual(len(calls), 2)

    def test_volume_vertices_avoid_support_lps_for_repeated_queries(self) -> None:
        weight_space = build_weight_space(goal_count=5, answered_queries=[])
        compute_polytope_intrinsic_volume(weight_space)

        for ratio in (0.001, 0.5, 1.0, 2.0, 1000.0):
            query = Query(ziel_index_a=0, ziel_index_b=1, value=ratio)
            with self.subTest(ratio=ratio), self._record_support_lps() as calls:
                probabilities = compute_exact_query_answer_probabilities(weight_space, query)
            self.assertEqual(calls, [])
            self.assertAlmostEqual(probabilities["<"], ratio / (1.0 + ratio), places=8)
            self.assertAlmostEqual(probabilities[">"], 1.0 / (1.0 + ratio), places=8)

    def test_cached_support_and_probabilities_match_lps_on_random_states(self) -> None:
        rng = np.random.default_rng(20261012)
        reused_supports = 0
        for goal_count in (3, 5):
            for state_index in range(3):
                target = rng.uniform(0.5, 1.5, size=goal_count)
                target /= target.sum()
                answers = []
                for _ in range(goal_count + 1):
                    a, b = (int(index) for index in rng.choice(goal_count, 2, replace=False))
                    factor = float(rng.choice((0.6, 1.4)))
                    query = Query(
                        ziel_index_a=a,
                        ziel_index_b=b,
                        value=float(target[a] / target[b] * factor),
                    )
                    answers.append(query.answer(">" if factor < 1.0 else "<"))
                weight_space = build_weight_space(goal_count, answers)
                # Keep the region away from zero weights while retaining the
                # feasible target. Extreme ratios then also test forced answers.
                for a in range(goal_count):
                    row = [0.0] * goal_count
                    row[a] = -1.0
                    weight_space.add_inequality(row, -0.02)
                self.assertGreater(compute_polytope_intrinsic_volume(weight_space), 0.0)

                for query_index, ratio in enumerate((0.001, 0.25, 0.8, 1.0, 2.0, 4.0, 1000.0)):
                    a, b = (int(index) for index in rng.choice(goal_count, 2, replace=False))
                    query = Query(ziel_index_a=a, ziel_index_b=b, value=ratio)
                    objective = [0.0] * goal_count
                    objective[a], objective[b] = 1.0, -ratio
                    with self.subTest(goals=goal_count, state=state_index, query=query_index):
                        lower = weight_space.minimize(objective)
                        upper = weight_space.maximize(objective)
                        support = polytope_volume._query_support_from_cached_vertices(
                            weight_space, objective, 1e-10
                        )
                        if support is not None:
                            reused_supports += 1
                            np.testing.assert_allclose(
                                support,
                                (lower.optimal_value, upper.optimal_value),
                                rtol=1e-8,
                                atol=1e-9,
                            )
                        with patch.object(
                            polytope_volume,
                            "_query_support_from_cached_vertices",
                            return_value=None,
                        ):
                            reference = compute_exact_query_answer_probabilities(weight_space, query)
                        actual = compute_exact_query_answer_probabilities(weight_space, query)
                        np.testing.assert_allclose(
                            [actual[operator] for operator in ("<", "=", ">")],
                            [reference[operator] for operator in ("<", "=", ">")],
                            rtol=1e-9,
                            atol=1e-10,
                        )
        self.assertGreater(reused_supports, 30)

    def test_boundary_and_equality_answers_keep_lp_fallback(self) -> None:
        cases = [
            (build_weight_space(3, []), Query(ziel_index_a=0, ziel_index_b=1, value=0.0), ">", 1),
            (
                build_weight_space(3, [Query(ziel_index_a=0, ziel_index_b=1, value=2.0).answer(">")]),
                Query(ziel_index_a=0, ziel_index_b=1, value=2.0),
                ">",
                1,
            ),
            (
                build_weight_space(3, [Query(ziel_index_a=0, ziel_index_b=1, value=1.0).answer("=")]),
                Query(ziel_index_a=0, ziel_index_b=1, value=1.0),
                "=",
                2,
            ),
        ]
        for weight_space, query, answer, support_lp_count in cases:
            with self.subTest(query=query, answer=answer):
                self.assertGreater(compute_polytope_intrinsic_volume(weight_space), 0.0)
                with self._record_support_lps() as calls:
                    probabilities = compute_exact_query_answer_probabilities(weight_space, query)
                self.assertEqual(len(calls), support_lp_count)
                self.assertEqual(probabilities[answer], 1.0)
                self.assertEqual(sum(probabilities.values()), 1.0)

    def test_near_zero_support_uses_lps_even_outside_answer_tolerance(self) -> None:
        weight_space = build_weight_space(
            3, [Query(ziel_index_a=0, ziel_index_b=1, value=1.0).answer(">")]
        )
        compute_polytope_intrinsic_volume(weight_space)
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0 + 5e-9)
        objective = [1.0, -query.value, 0.0]
        self.assertLess(weight_space.minimize(objective).optimal_value, -1e-10)
        with patch.object(
            polytope_volume, "_query_support_from_cached_vertices", return_value=None
        ):
            reference = compute_exact_query_answer_probabilities(weight_space, query)
        with self._record_support_lps() as calls:
            actual = compute_exact_query_answer_probabilities(weight_space, query)
        self.assertEqual(len(calls), 1)
        self.assertEqual(actual, reference)

    def test_inequality_degenerate_geometry_keeps_lp_fallback(self) -> None:
        weight_space = build_weight_space(3, [])
        for a, value in enumerate((0.6, 0.3, 0.1)):
            row = [0.0] * 3
            row[a] = 1.0
            weight_space.add_inequality(row, value)
            weight_space.add_inequality([-coefficient for coefficient in row], -value)
        self.assertEqual(compute_polytope_intrinsic_volume(weight_space), 0.0)

        with self._record_support_lps() as calls:
            probabilities = compute_exact_query_answer_probabilities(
                weight_space, Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(probabilities, {"<": 0.0, "=": 0.0, ">": 1.0})

    def test_failed_geometry_does_not_become_a_support_cache_hit(self) -> None:
        weight_space = build_weight_space(3, [])
        weight_space.add_inequality([-1.0, 0.0, 0.0], -0.6)
        failed = PolytopeVertices(
            status="error",
            vertices=np.empty((0, 3)),
            affine_dimension=2,
            message="geometry failed",
        )
        with patch.object(polytope_volume, "enumerate_polytope_vertices", return_value=failed):
            recovered = compute_polytope_intrinsic_volume(weight_space)
            self.assertGreater(recovered, 0.0)
            self.assertTrue(getattr(recovered, "_exact_lowdimensional_volume_correction", False))
            self.assertEqual(polytope_volume.polytope_vertex_cache_info().currsize, 0)
        with self._record_support_lps() as calls:
            probabilities = compute_exact_query_answer_probabilities(
                weight_space, Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(probabilities[">"], 1.0)

    def test_unsupported_geometry_failure_still_raises_without_support_cache_hit(self) -> None:
        weight_space = build_weight_space(5, [])
        failed = PolytopeVertices(status="error", vertices=np.empty((0, 5)),
                                 affine_dimension=4, message="geometry failed")
        with patch.object(polytope_volume, "enumerate_polytope_vertices", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "^geometry failed$"):
                compute_polytope_intrinsic_volume(weight_space)
        self.assertEqual(polytope_volume.polytope_vertex_cache_info().currsize, 0)

    def test_geometry_for_candidate_region_does_not_replace_whole_weight_space(self) -> None:
        weight_space = build_weight_space(3, [])
        candidate_region = build_weight_space(3, [])
        candidate_region.add_inequality([-1.0, 0.0, 0.0], -0.9)
        candidate_vertices = enumerate_polytope_vertices(candidate_region)
        self.assertGreater(
            polytope_volume.compute_polytope_intrinsic_volume_from_vertices(
                candidate_region, candidate_vertices
            ),
            0.0,
        )
        with self._record_support_lps() as calls:
            probabilities = compute_exact_query_answer_probabilities(
                weight_space, Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
            )
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(probabilities["<"], 0.5)
        self.assertAlmostEqual(probabilities[">"], 0.5)

    def test_constraint_changes_and_cache_reset_do_not_reuse_stale_vertices(self) -> None:
        weight_space = build_weight_space(3, [])
        compute_polytope_intrinsic_volume(weight_space)
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        equivalent = LinearConstraintSystem()
        equivalent.add_constraint_system(weight_space)
        with self._record_support_lps() as calls:
            compute_exact_query_answer_probabilities(equivalent, query)
        self.assertEqual(calls, [])

        weight_space.add_inequality([-1.0, 0.0, 0.0], -0.9)
        with self._record_support_lps() as calls:
            probabilities = compute_exact_query_answer_probabilities(weight_space, query)
        self.assertEqual(len(calls), 2)
        self.assertEqual(probabilities[">"], 1.0)
        compute_polytope_intrinsic_volume(weight_space)
        with self._record_support_lps() as calls:
            compute_exact_query_answer_probabilities(weight_space, query)
        self.assertEqual(calls, [])

        clear_polytope_volume_cache()
        with self._record_support_lps() as calls:
            compute_exact_query_answer_probabilities(weight_space, query)
        self.assertEqual(len(calls), 2)

    def test_support_vertex_cache_respects_geometry_tolerance(self) -> None:
        weight_space = build_weight_space(3, [])
        compute_polytope_intrinsic_volume(weight_space, tolerance=1e-10)
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        with self._record_support_lps() as calls:
            probabilities = compute_exact_query_answer_probabilities(weight_space, query, tolerance=1e-7)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(probabilities["<"], 0.5)

    def test_cached_geometry_is_immutable_and_detached_from_enumerator_array(self) -> None:
        weight_space = build_weight_space(3, [])
        vertices = enumerate_polytope_vertices(weight_space)
        with patch.object(polytope_volume, "enumerate_polytope_vertices", return_value=vertices):
            compute_polytope_intrinsic_volume(weight_space)
        cached = polytope_volume._cached_polytope_vertices(
            polytope_volume._constraint_system_key(weight_space), 1e-10
        )
        self.assertIsNotNone(cached)
        with self.assertRaises(ValueError):
            cached.vertices.setflags(write=True)
        vertices.vertices[:] = 0.0
        with self._record_support_lps() as calls:
            actual = compute_exact_query_answer_probabilities(
                weight_space, Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
            )
        self.assertEqual(calls, [])
        self.assertAlmostEqual(actual["<"], 0.5)

    def test_eviction_keeps_volume_cache_valid_and_support_falls_back(self) -> None:
        first = build_weight_space(3, [])
        second = build_weight_space(3, [])
        second.add_inequality([-1.0, 0.0, 0.0], -0.1)
        # Either limit must independently evict the first entry. Evicting
        # geometry must not discard its independently cached volume.
        for limits in ((1, 1_000_000), (100, 80)):
            with self.subTest(limits=limits), patch.object(
                polytope_volume, "_VERTEX_CACHE_MAX_SIZE", limits[0]
            ), patch.object(polytope_volume, "_VERTEX_CACHE_MAX_BYTES", limits[1]):
                clear_polytope_volume_cache()
                first_volume = compute_polytope_intrinsic_volume(first)
                compute_polytope_intrinsic_volume(second)
                info = polytope_volume.polytope_vertex_cache_info()
                self.assertLessEqual(info.currsize, info.maxsize)
                self.assertLessEqual(info.currbytes, info.maxbytes)
                self.assertGreaterEqual(info.currbytes, 0)
                self.assertEqual(compute_polytope_intrinsic_volume(first), first_volume)
                with self._record_support_lps() as calls:
                    actual = compute_exact_query_answer_probabilities(
                        first, Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
                    )
                self.assertEqual(len(calls), 2)
                self.assertAlmostEqual(actual["<"], 0.5)

    @staticmethod
    @contextmanager
    def _record_support_lps():
        original_minimize = LinearConstraintSystem.minimize
        calls = []

        def recorded_minimize(system, objective):
            calls.append(tuple(objective))
            return original_minimize(system, objective)

        with patch.object(LinearConstraintSystem, "minimize", recorded_minimize):
            yield calls


if __name__ == "__main__":
    unittest.main()
