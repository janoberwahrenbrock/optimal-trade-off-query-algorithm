from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import HalfspaceIntersection, QhullError

from multistep.src.polytope_geometry import (
    _vertices_satisfy_system,
    enumerate_polytope_vertices,
)
from multistep.src.weight_space import build_weight_space


class PolytopeGeometryTest(unittest.TestCase):
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
