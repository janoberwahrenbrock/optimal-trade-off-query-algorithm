from __future__ import annotations

import unittest
from unittest.mock import patch

from scipy.spatial import HalfspaceIntersection, QhullError

from multistep.src.polytope_geometry import enumerate_polytope_vertices
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


if __name__ == "__main__":
    unittest.main()
