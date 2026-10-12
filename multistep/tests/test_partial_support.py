from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from multistep.src import linear_programming, polytope_volume, reusable_highs
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import Query
from multistep.src.models.linear_optimization_result import LinearOptimizationResult
from multistep.src.polytope_geometry import PolytopeVertices
from multistep.src.weight_space import build_weight_space


@contextmanager
def support_mode(legacy):
    """Replay the unchanged all-or-nothing support contract without fixture files."""
    if not legacy:
        yield
        return
    original = polytope_volume._query_support_from_cached_vertices
    def legacy_only(system, objective, tolerance, *, constraint_key=None, allow_partial=False):
        return original(system, objective, tolerance,
                        constraint_key=constraint_key, allow_partial=False)
    with patch.object(polytope_volume, '_query_support_from_cached_vertices', legacy_only):
        yield


def probability_hex(value):
    return tuple((operator, value[operator].hex()) for operator in ('<', '=', '>'))


def snapshot(value):
    if isinstance(value, np.ndarray):
        return ('array', value.shape, value.dtype.str, value.strides, value.tobytes())
    if isinstance(value, dict):
        return tuple((key, snapshot(part)) for key, part in value.items())
    if isinstance(value, float):
        return ('float', value.hex())
    if isinstance(value, np.generic):
        return ('scalar', value.dtype.str, value.tobytes())
    return value


@contextmanager
def record_minimize(constraint_class):
    original = constraint_class.minimize
    calls = []
    def recorded(system, objective):
        calls.append(tuple(objective))
        return original(system, objective)
    with patch.object(constraint_class, 'minimize', recorded):
        yield calls


class PartialSupportTests(unittest.TestCase):
    def setUp(self):
        polytope_volume.clear_polytope_volume_cache()
        reusable_highs.clear_reusable_highs_cache()

    def tearDown(self):
        polytope_volume.clear_polytope_volume_cache()
        reusable_highs.clear_reusable_highs_cache()

    def test_partial_directions_keep_exactly_original_required_lp_and_old_counters(self):
        cases = [
            (build_weight_space(5, []), 0.0, (1.0, -0.0, 0.0, 0.0, 0.0), (0, 1)),
            (build_weight_space(5, [Query(ziel_index_a=0, ziel_index_b=1, value=1.0).answer('>')]),
             1.0, (1.0, -1.0, 0.0, 0.0, 0.0), (0, 1)),
            (build_weight_space(5, [Query(ziel_index_a=0, ziel_index_b=1, value=1.0).answer('<')]),
             1.0, (-1.0, 1.0, -0.0, -0.0, -0.0), (1, 0)),
        ]
        for system, ratio, required_objective, direction in cases:
            with self.subTest(ratio=ratio, direction=direction):
                polytope_volume.clear_polytope_volume_cache()
                polytope_volume.compute_polytope_intrinsic_volume(system)
                with record_minimize(LinearConstraintSystem) as calls:
                    actual = polytope_volume.compute_exact_query_answer_probabilities(
                        system, Query(ziel_index_a=0, ziel_index_b=1, value=ratio))
                self.assertEqual(calls, [required_objective])
                self.assertEqual(actual['='], 0.0)
                info = polytope_volume.polytope_vertex_cache_info()
                self.assertEqual(info.support_hits, 0)
                self.assertEqual(info.support_fallbacks, 1)
                partial = polytope_volume.polytope_partial_support_cache_info()
                self.assertEqual(partial.partial_hits, 1)
                self.assertEqual((partial.lower_hits, partial.upper_hits), direction)
                self.assertEqual(partial.avoided_lps, 1)
                self.assertEqual(len(info), 8)
                self.assertEqual(info._fields[-2:], ('support_hits', 'support_fallbacks'))
        polytope_volume.clear_polytope_volume_cache()
        self.assertEqual(tuple(polytope_volume.polytope_partial_support_cache_info()), (0, 0, 0, 0))

    def test_legacy_helper_api_all_or_nothing_and_patch_override_remain(self):
        system = build_weight_space(3, [])
        polytope_volume.compute_polytope_intrinsic_volume(system)
        objective = [1.0, -0.0, 0.0]
        self.assertIsNone(polytope_volume._query_support_from_cached_vertices(system, objective, 1e-10))
        self.assertEqual(tuple(polytope_volume.polytope_partial_support_cache_info()), (0, 0, 0, 0))
        partial = polytope_volume._query_support_from_cached_vertices(
            system, objective, 1e-10, allow_partial=True)
        self.assertIsNone(partial[0]); self.assertGreater(partial[1], 0.0)
        with patch.object(polytope_volume, '_query_support_from_cached_vertices', return_value=None), \
             record_minimize(LinearConstraintSystem) as calls:
            result = polytope_volume.compute_exact_query_answer_probabilities(
                system, Query(ziel_index_a=0, ziel_index_b=1, value=0.0))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result, {'<': 0.0, '=': 0.0, '>': 1.0})

    def test_both_unsafe_both_safe_cold_degenerate_and_geometry_failure_paths(self):
        equality = build_weight_space(3, [Query(ziel_index_a=0, ziel_index_b=1, value=1.0).answer('=')])
        safe = build_weight_space(3, [])
        degenerate = build_weight_space(3, [])
        for i, value in enumerate((0.6, 0.3, 0.1)):
            row = [0.0] * 3; row[i] = 1.0
            degenerate.add_inequality(row, value)
            degenerate.add_inequality([-v for v in row], -value)
        for system, warm, expected_calls in ((equality, True, 2), (safe, True, 0),
                                             (safe, False, 2), (degenerate, True, 2)):
            with self.subTest(warm=warm, expected_calls=expected_calls):
                polytope_volume.clear_polytope_volume_cache()
                if warm: polytope_volume.compute_polytope_intrinsic_volume(system)
                with record_minimize(LinearConstraintSystem) as calls:
                    polytope_volume.compute_exact_query_answer_probabilities(
                        system, Query(ziel_index_a=0, ziel_index_b=1, value=1.0))
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(polytope_volume.polytope_partial_support_cache_info().partial_hits, 0)
        failed = PolytopeVertices(status='error', vertices=np.empty((0, 3)), affine_dimension=2,
                                 message='deliberate geometry failure')
        polytope_volume.clear_polytope_volume_cache()
        with patch.object(polytope_volume, 'enumerate_polytope_vertices', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, 'deliberate geometry failure'):
                polytope_volume.compute_polytope_intrinsic_volume(safe)
        with record_minimize(LinearConstraintSystem) as calls:
            polytope_volume.compute_exact_query_answer_probabilities(
                safe, Query(ziel_index_a=0, ziel_index_b=1, value=0.0))
        self.assertEqual(len(calls), 2)

    def test_same_exact_margin_boundary_signed_zero_and_nonfinite_conservatism(self):
        system = build_weight_space(3, [])
        objective = [1.0, 0.0, 0.0]
        margin = 1e-8
        for lower in (0.0, -0.0, margin, -margin, np.nextafter(margin, 0.0),
                      np.nextafter(-margin, 0.0), np.nextafter(margin, np.inf),
                      np.nextafter(-margin, -np.inf)):
            vertices = PolytopeVertices(status='full_dimensional',
                vertices=np.asarray([[lower, 0.0, 0.0], [1.0, 0.0, 0.0]]), affine_dimension=2)
            with patch.object(polytope_volume, '_cached_polytope_vertices', return_value=vertices):
                legacy = polytope_volume._query_support_from_cached_vertices(system, objective, 1e-10)
                partial = polytope_volume._query_support_from_cached_vertices(
                    system, objective, 1e-10, allow_partial=True)
            if abs(lower) <= margin:
                self.assertIsNone(legacy); self.assertIsNone(partial[0])
                self.assertEqual(partial[1].hex(), 1.0.hex())
            else:
                self.assertEqual(tuple(v.hex() for v in legacy), tuple(v.hex() for v in partial))
        for nonfinite in (np.inf, -np.inf, np.nan):
            vertices = PolytopeVertices(status='full_dimensional',
                vertices=np.asarray([[nonfinite, 0.0, 0.0], [1.0, 0.0, 0.0]]), affine_dimension=2)
            with patch.object(polytope_volume, '_cached_polytope_vertices', return_value=vertices):
                self.assertIsNone(polytope_volume._query_support_from_cached_vertices(
                    system, objective, 1e-10, allow_partial=True))
            polytope_volume._remember_polytope_vertices(
                polytope_volume._constraint_system_key(system), 1e-10, vertices)
        self.assertEqual(polytope_volume.polytope_vertex_cache_info().currsize, 0)

    def test_original_status_missing_value_and_min_before_max_messages(self):
        system = build_weight_space(3, [])
        query = Query(ziel_index_a=0, ziel_index_b=1, value=0.0)
        polytope_volume.compute_polytope_intrinsic_volume(system)
        for result, message in (
            (LinearOptimizationResult(status='infeasible', objective_sense='min'),
             'cannot determine exact query support'),
            (LinearOptimizationResult(status='optimal', objective_sense='min'),
             'query support optimization has no value'),
        ):
            with patch.object(LinearConstraintSystem, 'minimize', return_value=result) as call:
                with self.assertRaisesRegex(RuntimeError, message):
                    polytope_volume.compute_exact_query_answer_probabilities(system, query)
            self.assertEqual(call.call_count, 1)
        polytope_volume.clear_polytope_volume_cache()
        outcomes = [LinearOptimizationResult(status='infeasible', objective_sense='min'),
                    LinearOptimizationResult(status='optimal', objective_sense='min', optimal_value=-1.0)]
        with patch.object(LinearConstraintSystem, 'minimize', side_effect=outcomes) as call:
            with self.assertRaisesRegex(RuntimeError, 'cannot determine exact query support'):
                polytope_volume.compute_exact_query_answer_probabilities(system, query)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[0].args[0], [1.0, -0.0, 0.0])
        self.assertEqual(call.call_args_list[1].args[0], [-1.0, 0.0, -0.0])

    def test_cached_vertices_mutation_keys_eviction_tolerance_and_immutability(self):
        system = build_weight_space(3, [])
        vertices = polytope_volume.enumerate_polytope_vertices(system)
        with patch.object(polytope_volume, 'enumerate_polytope_vertices', return_value=vertices):
            polytope_volume.compute_polytope_intrinsic_volume(system)
        vertices.vertices[:] = 5.0
        cached = polytope_volume._cached_polytope_vertices(polytope_volume._constraint_system_key(system), 1e-10)
        with self.assertRaises(ValueError): cached.vertices.setflags(write=True)
        query = Query(ziel_index_a=0, ziel_index_b=1, value=0.0)
        with record_minimize(LinearConstraintSystem) as calls:
            result = polytope_volume.compute_exact_query_answer_probabilities(system, query)
        self.assertEqual(len(calls), 1); self.assertEqual(result['>'], 1.0)
        for changed in ('tolerance', 'mutation', 'eviction'):
            with self.subTest(changed=changed):
                tolerance = 1e-7 if changed == 'tolerance' else 1e-10
                if changed == 'mutation': system.add_inequality([-1.0, 0.0, 0.0], -0.2)
                if changed == 'eviction':
                    with patch.object(polytope_volume, '_VERTEX_CACHE_MAX_SIZE', 1):
                        other = build_weight_space(3, [])
                        other.add_inequality([-1.0, 0.0, 0.0], -0.3)
                        polytope_volume.compute_polytope_intrinsic_volume(other)
                with record_minimize(LinearConstraintSystem) as calls:
                    polytope_volume.compute_exact_query_answer_probabilities(system, query, tolerance=tolerance)
                self.assertEqual(len(calls), 2)

    def test_required_partial_lp_preserves_original_retry_chain_and_public_solver_override(self):
        for answer in (None, '<'):
            original = build_weight_space(3, [] if answer is None else [
                Query(ziel_index_a=0, ziel_index_b=1, value=1.0).answer(answer)])
            query_data = dict(ziel_index_a=0, ziel_index_b=1, value=0.0 if answer is None else 1.0)
            wanted_sign = 1.0 if answer is None else -1.0
            results = []
            for legacy in (False, True):
                polytope_volume.clear_polytope_volume_cache(); reusable_highs.clear_reusable_highs_cache()
                system = LinearConstraintSystem(**original.model_dump())
                polytope_volume.compute_polytope_intrinsic_volume(system)
                trace = []
                def overridden(**kwargs):
                    wanted = kwargs['c'][0] == wanted_sign
                    if wanted:
                        trace.append((tuple(float(x).hex() for x in kwargs['c']),
                                      kwargs['method'], dict(kwargs['options'])))
                    success = not wanted or len(trace) == 4
                    return OptimizeResult(success=success, status=0 if success else 4,
                        message='deliberate Not Set' if not success else 'optimal override',
                        x=np.zeros(3) if success else None,
                        fun=(0.0 if wanted else -1.0) if success else None)
                with support_mode(legacy), patch.object(linear_programming, 'linprog', overridden):
                    outcome = polytope_volume.compute_exact_query_answer_probabilities(system, Query(**query_data))
                results.append((probability_hex(outcome), trace))
            self.assertEqual(results[0], results[1])
            trace = results[0][1]
            self.assertEqual([call[1] for call in trace], ['highs', 'highs', 'highs-ds', 'highs-ipm'])
            self.assertNotIn('presolve', trace[0][2])
            self.assertTrue(all(call[2]['presolve'] is False for call in trace[1:]))

    def test_downstream_full_lp_results_keep_cold_numerics_after_skips_and_eviction(self):
        rng = np.random.default_rng(507132)
        models = []
        for index in range(48):
            matrix = np.vstack((np.eye(3), -np.eye(3), rng.normal(size=(3, 3))))
            models.append(dict(c=rng.normal(size=3), A_ub=matrix,
                b_ub=rng.uniform(0.2, 2.0, len(matrix)), bounds=[(None, None)] * 3,
                options=linear_programming.LINPROG_OPTIONS))
        sequences = []
        for legacy in (False, True):
            reusable_highs.clear_reusable_highs_cache(); polytope_volume.clear_polytope_volume_cache()
            system = build_weight_space(3, [])
            polytope_volume.compute_polytope_intrinsic_volume(system)
            results = []
            with support_mode(legacy):
                for index in range(96):
                    # Original all-or-nothing path solves min/max; partial path min only.
                    polytope_volume.compute_exact_query_answer_probabilities(system, Query(
                        ziel_index_a=0, ziel_index_b=1, value=0.0))
                    kwargs = models[index % len(models)]
                    results.append(snapshot(linear_programming.run_linprog_with_retries(**kwargs)))
                    # Repeated objectives then exercise native model hits too.
                    results.append(snapshot(linear_programming.run_linprog_with_retries(**kwargs)))
            sequences.append(results)
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(len(sequences[0]), 192)


if __name__ == '__main__':
    unittest.main()
