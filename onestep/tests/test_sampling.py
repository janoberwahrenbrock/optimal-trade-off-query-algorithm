from __future__ import annotations

import unittest

from onestep.src.build_ungleichungssysteme import build_W
from onestep.src.io_models import AnsweredQuery
from onestep.src.sampling import sample_points_from_ungleichungssystem


class SamplingRegressionTests(unittest.TestCase):
    def test_sampling_accepts_linprog_feasible_start_point_at_solver_tolerance(self) -> None:
        answered_queries = [
            AnsweredQuery(
                ziel_index_a=0,
                ziel_index_b=4,
                value=5.980121291776192,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=4,
                ziel_index_b=6,
                value=1.1016599577570163,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=1,
                ziel_index_b=6,
                value=10.904275609051547,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=5,
                ziel_index_b=6,
                value=1.3095453788006652,
                operator=">",
            ),
            AnsweredQuery(
                ziel_index_a=3,
                ziel_index_b=5,
                value=1.139659900048831,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=5,
                ziel_index_b=6,
                value=113.96424155807941,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=5,
                ziel_index_b=6,
                value=16.65563878152816,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=2,
                ziel_index_b=6,
                value=1.7769803532090718,
                operator=">",
            ),
            AnsweredQuery(
                ziel_index_a=2,
                ziel_index_b=5,
                value=172.05108394967047,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=2,
                ziel_index_b=6,
                value=225.30839235855754,
                operator="<",
            ),
            AnsweredQuery(
                ziel_index_a=3,
                ziel_index_b=6,
                value=7.236647336301882,
                operator="<",
            ),
        ]
        system = build_W(7, answered_queries)

        samples = sample_points_from_ungleichungssystem(
            system=system,
            num_samples=1,
            burn_in=0,
            thinning=1,
            seed=4106120257,
        )

        self.assertEqual(len(samples), 1)


if __name__ == "__main__":
    unittest.main()
