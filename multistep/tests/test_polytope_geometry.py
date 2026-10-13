from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.spatial import HalfspaceIntersection, QhullError

from multistep.src import polytope_geometry
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.polytope_geometry import (
    _vertices_satisfy_system,
    enumerate_polytope_vertices,
)
from multistep.src.polytope_volume import compute_polytope_intrinsic_volume_from_vertices
from multistep.src.weight_space import build_weight_space


class PolytopeGeometryTest(unittest.TestCase):
    def test_normalized_center_recovers_recorded_depth_three_failure(self) -> None:
        ratio = 18378469.506142955
        chain_ratio = 0.001
        system = LinearConstraintSystem(
            inequalities_left_side=[
                [-1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, chain_ratio, -1.0, 0.0],
                [0.0, 0.0, 0.0, chain_ratio, -1.0],
                [0.0, -1.0, 0.0, 0.0, ratio],
            ],
            inequalities_right_side=[0.0] * 8,
            equalities_left_side=[[1.0] * 5],
            equalities_right_side=[1.0],
        )
        # This successful-but-invalid center was captured on Bench i7. Inject
        # it so the regression also exercises recovery on SciPy versions that
        # happen to solve the original unscaled LP correctly.
        invalid_result = OptimizeResult(
            success=True, status=0, message="recorded false interior point",
            x=np.asarray([
                0.8618072362300585, -0.1381956487848347,
                -0.138195648753611, -0.1381956487535791,
                3.0398013871656406e-08,
            ]),
        )
        original_solver = polytope_geometry.run_linprog_with_retries
        calls = []

        def replay_bad_center_then_solve(**kwargs):
            calls.append(kwargs)
            return invalid_result if len(calls) == 1 else original_solver(**kwargs)

        with patch.object(
            polytope_geometry, "run_linprog_with_retries",
            side_effect=replay_bad_center_then_solve,
        ):
            result = enumerate_polytope_vertices(system)

        self.assertEqual(len(calls), 2)
        original_matrix = calls[0]["A_ub"]
        norms = original_matrix[:, -1]
        np.testing.assert_array_equal(calls[1]["A_ub"][:, -1], np.ones(8))
        np.testing.assert_array_equal(
            calls[1]["A_ub"][:, :-1], original_matrix[:, :-1] / norms[:, None]
        )
        np.testing.assert_array_equal(calls[1]["b_ub"], calls[0]["b_ub"] / norms)
        for name in ("c", "bounds", "method", "options"):
            np.testing.assert_equal(calls[1][name], calls[0][name])
        self.assertEqual(result.status, "full_dimensional", result.message)
        self.assertEqual(result.affine_dimension, 4)
        self.assertEqual(len(result.vertices), 5)

        # Setting x3=w3-a*w2, x4=w4-a*w3, x1=w1-B*w4
        # gives a triangular transform with determinant 1. Its nonnegative
        # simplex has sum x0+x1+c2*x2+c3*x3+c4*x4=1. The hyperplane's
        # intrinsic-volume factor is sqrt(5), independent of the LP/Qhull.
        c2 = 1.0 + chain_ratio + (1.0 + ratio) * chain_ratio**2
        c3 = 1.0 + (1.0 + ratio) * chain_ratio
        c4 = 1.0 + ratio
        expected_volume = np.sqrt(5.0) / (24.0 * c2 * c3 * c4)
        volume = compute_polytope_intrinsic_volume_from_vertices(system, result)
        self.assertGreater(volume, 0.0)
        self.assertAlmostEqual(volume / expected_volume, 1.0, delta=1e-7)

    def test_valid_center_uses_one_original_lp_without_changing_center(self) -> None:
        original_solver = polytope_geometry.run_linprog_with_retries
        original_intersections = polytope_geometry._compute_halfspace_intersections_with_retries
        captured_results = []

        def record_result(**kwargs):
            result = original_solver(**kwargs)
            captured_results.append(result)
            return result

        with patch.object(
            polytope_geometry, "run_linprog_with_retries", side_effect=record_result,
        ) as mocked_solver, patch.object(
            polytope_geometry, "_compute_halfspace_intersections_with_retries",
            wraps=original_intersections,
        ) as mocked_intersections:
            result = enumerate_polytope_vertices(
                build_weight_space(goal_count=5, answered_queries=[])
            )

        self.assertEqual(result.status, "full_dimensional")
        self.assertEqual(mocked_solver.call_count, 1)
        np.testing.assert_array_equal(
            mocked_intersections.call_args.kwargs["center"], captured_results[0].x[:-1]
        )
        self.assertEqual(result.interior_radius, float(captured_results[0].x[-1]))

    def test_invalid_normalized_centers_remain_errors(self) -> None:
        bad_original = OptimizeResult(
            success=True, status=0, message="invalid original center",
            x=np.asarray([10.0, 10.0, 0.1]),
        )
        for invalid_x in (
            [10.0, 10.0, 0.1],  # positive radius with an infeasible center
            [10.0, 10.0, 0.0],  # must not falsely classify zero volume
            [np.nan, 0.0, 0.1],
            [np.inf, 0.0, 0.1],
            [0.0, 0.0, np.nan],
            [0.0, 0.0, np.inf],
            [0.0, 0.0, -1.0],
        ):
            invalid_retry = OptimizeResult(
                success=True, status=0, message="invalid normalized center",
                x=np.asarray(invalid_x),
            )
            with self.subTest(x=invalid_x), patch.object(
                polytope_geometry, "run_linprog_with_retries",
                side_effect=[bad_original, invalid_retry],
            ) as mocked_solver, patch.object(
                polytope_geometry, "HalfspaceIntersection",
            ) as mocked_intersection:
                result = enumerate_polytope_vertices(
                    build_weight_space(goal_count=3, answered_queries=[])
                )
            self.assertEqual(result.status, "error")
            self.assertIn("invalid center or radius", result.message)
            self.assertEqual(mocked_solver.call_count, 2)
            mocked_intersection.assert_not_called()

    def test_failed_normalized_retry_does_not_declare_region_empty(self) -> None:
        bad_original = OptimizeResult(
            success=True, status=0, message="invalid original center",
            x=np.asarray([10.0, 10.0, 0.1]),
        )
        failed_retry = OptimizeResult(
            success=False, status=2, message="numerical infeasibility", x=None,
        )
        with patch.object(
            polytope_geometry, "run_linprog_with_retries",
            side_effect=[bad_original, failed_retry],
        ):
            result = enumerate_polytope_vertices(
                build_weight_space(goal_count=3, answered_queries=[])
            )
        self.assertEqual(result.status, "error")
        self.assertIn("row-normalized center LP failed", result.message)

    def test_invalid_small_radius_or_boundary_center_is_retried(self) -> None:
        system = LinearConstraintSystem(
            inequalities_left_side=[[1.0], [-1.0]],
            inequalities_right_side=[1.0, 0.0],
        )
        original_solver = polytope_geometry.run_linprog_with_retries
        for invalid_x in ([2.0, 0.0], [0.0, 0.1]):
            calls = []

            def recover(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return OptimizeResult(
                        success=True, status=0, message="invalid center",
                        x=np.asarray(invalid_x),
                    )
                return original_solver(**kwargs)

            with self.subTest(x=invalid_x), patch.object(
                polytope_geometry, "run_linprog_with_retries", side_effect=recover,
            ):
                result = enumerate_polytope_vertices(system)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result.status, "full_dimensional")
            np.testing.assert_array_equal(result.vertices, [[0.0], [1.0]])

    def test_nonfinite_original_center_or_radius_is_retried(self) -> None:
        original_solver = polytope_geometry.run_linprog_with_retries
        for invalid_x in ([np.nan, 0.0, 0.1], [0.0, 0.0, np.nan]):
            calls = []

            def recover(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return OptimizeResult(
                        success=True, status=0, message="nonfinite center result",
                        x=np.asarray(invalid_x),
                    )
                return original_solver(**kwargs)

            with self.subTest(x=invalid_x), patch.object(
                polytope_geometry, "run_linprog_with_retries", side_effect=recover,
            ):
                result = enumerate_polytope_vertices(
                    build_weight_space(goal_count=3, answered_queries=[])
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(result.status, "full_dimensional")
            self.assertEqual(len(result.vertices), 3)

    def test_valid_zero_radius_preserves_lower_dimension_without_retry(self) -> None:
        system = LinearConstraintSystem(
            inequalities_left_side=[[1.0], [-1.0]],
            inequalities_right_side=[0.0, 0.0],
        )
        with patch.object(
            polytope_geometry, "run_linprog_with_retries",
            wraps=polytope_geometry.run_linprog_with_retries,
        ) as mocked_solver:
            result = enumerate_polytope_vertices(system)
        self.assertEqual(result.status, "lower_dimensional")
        self.assertEqual(result.interior_radius, 0.0)
        self.assertEqual(mocked_solver.call_count, 1)

    def test_retries_halfspace_intersection_after_qhull_precision_error(self) -> None:
        original = HalfspaceIntersection
        call_count = 0

        def fail_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise QhullError("synthetic precision error")
            return original(*args, **kwargs)

        with patch(
            "multistep.src.polytope_geometry.HalfspaceIntersection",
            side_effect=fail_once,
        ) as mocked_intersection:
            result = enumerate_polytope_vertices(
                build_weight_space(goal_count=3, answered_queries=[])
            )

        self.assertEqual(result.status, "full_dimensional")
        self.assertEqual(len(result.vertices), 3)
        self.assertEqual(mocked_intersection.call_count, 2)
        self.assertEqual(
            mocked_intersection.call_args_list[1].kwargs["qhull_options"],
            "Q12",
        )

    def test_retries_when_qhull_returns_vertices_outside_halfspaces(self) -> None:
        original = HalfspaceIntersection
        call_count = 0

        def return_invalid_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = original(*args, **kwargs)
            if call_count == 1:
                result.intersections[0, :] = 10.0
            return result

        with patch(
            "multistep.src.polytope_geometry.HalfspaceIntersection",
            side_effect=return_invalid_once,
        ) as mocked_intersection:
            result = enumerate_polytope_vertices(
                build_weight_space(goal_count=3, answered_queries=[])
            )

        self.assertEqual(result.status, "full_dimensional")
        self.assertEqual(len(result.vertices), 3)
        self.assertEqual(mocked_intersection.call_count, 2)
        self.assertEqual(
            mocked_intersection.call_args_list[1].kwargs["qhull_options"],
            "Q12",
        )

    def test_snaps_slightly_perturbed_qhull_vertex_to_active_facets(self) -> None:
        original = HalfspaceIntersection

        def return_slightly_invalid(*args, **kwargs):
            result = original(*args, **kwargs)
            halfspaces = np.asarray(args[0], dtype=float)
            point = result.intersections[0]
            residuals = halfspaces[:, :-1] @ point + halfspaces[:, -1]
            active_index = int(np.argmax(residuals))
            normal = halfspaces[active_index, :-1]
            result.intersections[0] = (
                point + 1e-7 * normal / float(normal @ normal)
            )
            return result

        with patch(
            "multistep.src.polytope_geometry.HalfspaceIntersection",
            side_effect=return_slightly_invalid,
        ) as mocked_intersection:
            result = enumerate_polytope_vertices(
                build_weight_space(goal_count=3, answered_queries=[])
            )

        self.assertEqual(result.status, "full_dimensional")
        self.assertEqual(len(result.vertices), 3)
        self.assertEqual(mocked_intersection.call_count, 1)

    def test_constraint_validation_is_relative_to_coefficient_scale(self) -> None:
        vertices = np.asarray([[1.0 + 5e-10]])
        self.assertTrue(
            _vertices_satisfy_system(
                vertices=vertices,
                inequality_matrix=np.asarray([[1e9]]),
                inequality_right_side=np.asarray([1e9]),
                equality_matrix=None,
                equality_right_side=None,
                tolerance=1e-10,
            )
        )


if __name__ == "__main__":
    unittest.main()
