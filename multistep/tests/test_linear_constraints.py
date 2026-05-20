from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multistep.src.linear_constraints import LinearConstraintSystem


UNKNOWN_PRIMAL_INFEASIBLE_MESSAGE = (
    "The HiGHS status code was not recognized. "
    "(HiGHS Status 15: model_status is Unknown; primal_status is Infeasible)"
)


class LinearConstraintSystemTests(unittest.TestCase):
    def test_minimize_and_maximize_over_simplex(self) -> None:
        system = LinearConstraintSystem()
        system.add_inequality([-1.0, 0.0], 0.0)
        system.add_inequality([0.0, -1.0], 0.0)
        system.add_equality([1.0, 1.0], 1.0)

        minimization_result = system.minimize([1.0, 0.0])
        maximization_result = system.maximize([1.0, 0.0])

        self.assertEqual(minimization_result.status, "optimal")
        self.assertAlmostEqual(minimization_result.optimal_value, 0.0)
        self.assertEqual(maximization_result.status, "optimal")
        self.assertAlmostEqual(maximization_result.optimal_value, 1.0)

    def test_minimize_maps_unknown_primal_infeasible_status_to_infeasible(self) -> None:
        system = self._build_one_variable_system()

        with patch(
            "multistep.src.linear_programming.linprog",
            return_value=SimpleNamespace(
                success=False,
                status=4,
                message=UNKNOWN_PRIMAL_INFEASIBLE_MESSAGE,
            ),
        ):
            result = system.minimize([1.0])

        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.objective_sense, "min")
        self.assertEqual(result.solver_status_code, 4)
        self.assertEqual(result.solver_message, UNKNOWN_PRIMAL_INFEASIBLE_MESSAGE)
        self.assertIsNone(result.optimal_value)

    def test_find_feasible_point_maps_unknown_primal_infeasible_status(self) -> None:
        system = self._build_one_variable_system()

        with patch(
            "multistep.src.linear_programming.linprog",
            return_value=SimpleNamespace(
                success=False,
                status=4,
                message=UNKNOWN_PRIMAL_INFEASIBLE_MESSAGE,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "system is infeasible"):
                system.find_feasible_point()

    def test_minimize_still_raises_for_unclassified_solver_status(self) -> None:
        system = self._build_one_variable_system()

        with patch(
            "multistep.src.linear_programming.linprog",
            return_value=SimpleNamespace(
                success=False,
                status=4,
                message="unexpected solver failure",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "linprog failed"):
                system.minimize([1.0])

    def test_maximize_retries_without_presolve_for_unclassified_solver_status(self) -> None:
        system = self._build_one_variable_system()

        with patch(
            "multistep.src.linear_programming.linprog",
            side_effect=[
                SimpleNamespace(
                    success=False,
                    status=4,
                    message="(HiGHS Status 0: Not Set)",
                ),
                SimpleNamespace(
                    success=False,
                    status=3,
                    message=(
                        "The problem is unbounded. "
                        "(HiGHS Status 10: model_status is Unbounded; "
                        "primal_status is Feasible)"
                    ),
                ),
            ],
        ) as mocked_linprog:
            result = system.maximize([1.0])

        self.assertEqual(result.status, "unbounded")
        self.assertEqual(result.objective_sense, "max")
        self.assertEqual(mocked_linprog.call_count, 2)
        self.assertNotIn("presolve", mocked_linprog.call_args_list[0].kwargs["options"])
        self.assertFalse(mocked_linprog.call_args_list[1].kwargs["options"]["presolve"])

    def test_maximize_retries_without_presolve_for_presolve_infeasible_status(self) -> None:
        system = self._build_one_variable_system()

        with patch(
            "multistep.src.linear_programming.linprog",
            side_effect=[
                SimpleNamespace(
                    success=False,
                    status=2,
                    message=(
                        "The problem is infeasible. "
                        "(HiGHS Status 8: model_status is Infeasible; "
                        "primal_status is None)"
                    ),
                ),
                SimpleNamespace(
                    success=False,
                    status=3,
                    message=(
                        "The problem is unbounded. "
                        "(HiGHS Status 10: model_status is Unbounded; "
                        "primal_status is Feasible)"
                    ),
                ),
            ],
        ) as mocked_linprog:
            result = system.maximize([1.0])

        self.assertEqual(result.status, "unbounded")
        self.assertEqual(result.objective_sense, "max")
        self.assertEqual(mocked_linprog.call_count, 2)
        self.assertFalse(mocked_linprog.call_args_list[1].kwargs["options"]["presolve"])

    def test_find_feasible_point_retries_without_presolve_and_returns_point(self) -> None:
        system = self._build_one_variable_system()

        with patch(
            "multistep.src.linear_programming.linprog",
            side_effect=[
                SimpleNamespace(
                    success=False,
                    status=4,
                    message="(HiGHS Status 0: Not Set)",
                ),
                SimpleNamespace(
                    success=True,
                    status=0,
                    message="Optimization terminated successfully.",
                    x=np.array([1.0], dtype=float),
                ),
            ],
        ) as mocked_linprog:
            point = system.find_feasible_point()

        self.assertEqual(point, [1.0])
        self.assertEqual(mocked_linprog.call_count, 2)
        self.assertFalse(mocked_linprog.call_args_list[1].kwargs["options"]["presolve"])

    def _build_one_variable_system(self) -> LinearConstraintSystem:
        system = LinearConstraintSystem()
        system.add_inequality([-1.0], 0.0)
        return system


if __name__ == "__main__":
    unittest.main()
