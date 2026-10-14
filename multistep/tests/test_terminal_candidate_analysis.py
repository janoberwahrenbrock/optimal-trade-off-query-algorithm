from __future__ import annotations

from dataclasses import replace, asdict
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import numpy as np

from multistep.optimized import query_efficiency as qe
from multistep.optimized import terminal_candidate_analysis as terminal
from multistep.optimized.value_function import (
    OptimizedMultistepConfig, compute_candidate_analysis_for_mode,
    ratio_interval_has_positive_width, ratio_interval_is_feasible,
)
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query
from multistep.src.polytope_geometry import PolytopeVertices
from multistep.src.polytope_volume import _ExactRecoveryVolume
from multistep.src.ratio_intervals import (
    _compute_ratio_intervals_from_vertices, compute_ratio_interval_for_candidate,
)
from multistep.src import ratio_intervals as ratios
from multistep.src.weight_space import build_weight_space


def hexify(value):
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {key: hexify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [hexify(item) for item in value]
    return value


def summary(analysis):
    return hexify((analysis.candidates, analysis.candidate_volumes, analysis.candidate_volume_shares))


class TerminalCandidateAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.config = OptimizedMultistepConfig(
            answer_probability_mode="exact_volume", use_ratio_terminal_counts=True,
            query_efficiency_stop_aware=True, volume_confidence_threshold=.99,
        )

    def compare_membership(self, vertices, geometry=1e-10, width=1e-12):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            intervals = _compute_ratio_intervals_from_vertices(vertices, vertices.shape[1], geometry)
            expected = (
                any(ratio_interval_is_feasible(i) for i in intervals.values()),
                any(ratio_interval_has_positive_width(i, width) for i in intervals.values()),
            )
            actual = terminal._vertices_membership(vertices, vertices.shape[1], geometry, width)
        self.assertEqual(actual, expected)

    def test_random_and_large_array_membership_matches_original(self):
        rng = np.random.default_rng(2026101421)
        for goals in (1, 2, 3, 5, 8):
            for rows in (1, 2, 5, 80, 81, 150):
                vertices = rng.normal(size=(rows, goals))
                vertices[rng.random(vertices.shape) < .2] = 0.
                for width in (0., 1e-12, .1, 1., 1e100):
                    self.compare_membership(vertices, width=width)

    def test_boundary_float_bits_overflow_and_strided_arrays(self):
        rng = np.random.default_rng(2026101422)
        bits = rng.integers(0, np.iinfo(np.uint64).max, size=(128, 5), dtype=np.uint64)
        bits &= np.uint64(0xFFEFFFFFFFFFFFFF)
        vertices = bits.view(np.float64)
        for tol in (1e-10, 1e-100, 1e-300):
            self.compare_membership(vertices, geometry=tol)
        values = np.asarray([-0., 0., np.nextafter(0., 1.), 1e-10,
                             np.nextafter(1e-10, 0.), np.nextafter(1e-10, np.inf), 1.])
        boundaries = np.column_stack([np.roll(values, index) for index in range(5)])
        for value in (boundaries, boundaries[::-1, ::-1], np.asfortranarray(boundaries)):
            self.compare_membership(value)

    def test_non_native_arrays_and_nonfinite_fallback_matches_original(self):
        base = np.asarray([[0., 1., .5], [.5, .2, 0.]])
        nonfinite = base.copy(); nonfinite[0, 1] = np.nan
        for array in (base.astype(np.float32), base.astype(int), base.astype('>f8'), nonfinite, np.empty((0, 3))):
            self.compare_membership(array)

    def test_no_interval_models_on_ordinary_terminal_fast_path(self):
        vertices = np.asarray([[.1, .2, .7], [.2, .1, .7]])
        with patch.object(terminal, "_compute_ratio_intervals_from_vertices", side_effect=AssertionError("unused models")):
            self.assertEqual(terminal._vertices_membership(vertices, 3, 1e-10, 1e-12), (True, True))

    def test_candidate_summary_preserves_volume_hex_and_order(self):
        rng = np.random.default_rng(2026101423)
        for goals in (2, 3, 5):
            alternatives = AlternativenMatrix(entries=rng.random((5, goals)).tolist())
            for answers in ([], [AnsweredQuery(ziel_index_a=0, ziel_index_b=1, value=.75, operator='<')],
                            [AnsweredQuery(ziel_index_a=0, ziel_index_b=1, value=1., operator='=')]):
                space = build_weight_space(goals, answers)
                for subset in (None, [4, 1, 3], []):
                    for volumes in (False, True):
                        args = (alternatives, space, subset, self.config)
                        expected = compute_candidate_analysis_for_mode(*args, include_candidate_volumes=volumes)
                        actual = terminal.compute_terminal_candidate_analysis_for_mode(*args, include_candidate_volumes=volumes)
                        self.assertEqual(summary(actual), summary(expected))

    def test_duplicate_ties_and_zero_width_feasible_fallback(self):
        alternatives = AlternativenMatrix(entries=[[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        for answers in ([], [AnsweredQuery(ziel_index_a=0, ziel_index_b=1, value=1., operator='=')]):
            space = build_weight_space(3, answers)
            expected = compute_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
            actual = terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
            self.assertEqual(summary(actual), summary(expected))

    def test_corrected_volume_keeps_all_original_ordered_lp_fallbacks(self):
        alternatives = AlternativenMatrix(entries=[[1., 0., 0.]])
        space = build_weight_space(3, [])
        with patch.object(ratios, "enumerate_polytope_vertices", return_value=PolytopeVertices('lower_dimensional', np.empty((0, 3)), 2)), \
             patch.object(ratios, "compute_polytope_intrinsic_volume_from_vertices", return_value=_ExactRecoveryVolume(.125)), \
             patch.object(ratios, "compute_ratio_interval_for_candidate", wraps=compute_ratio_interval_for_candidate) as fallback:
            result = terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, [0], self.config, True)
        self.assertEqual([(call.kwargs['goal_index_a'], call.kwargs['goal_index_b']) for call in fallback.call_args_list],
                         [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
        self.assertEqual(result.candidate_volume_shares, {0: 1.})

    def test_ambiguous_fallback_does_not_hide_later_original_error(self):
        alternatives = AlternativenMatrix(entries=[[1., 0., 0.]])
        space = build_weight_space(3, [])
        error = RuntimeError('original native LP error')
        real = compute_ratio_interval_for_candidate(alternatives, space, 0, 0, 1)
        with patch.object(ratios, "enumerate_polytope_vertices", return_value=PolytopeVertices('lower_dimensional', np.empty((0, 3)), 2)), \
             patch.object(ratios, "compute_ratio_interval_for_candidate", side_effect=[real, real, error]) as fallback:
            with self.assertRaises(RuntimeError) as raised:
                terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, [0], self.config)
        self.assertIs(raised.exception, error)
        self.assertEqual(fallback.call_count, 3)

    def test_original_geometry_alias_sentinel_identity_and_interval_overrides(self):
        alternatives = AlternativenMatrix(entries=[[1., 0.], [0., 1.]])
        space = build_weight_space(2, [])
        for hook in ('build_optimality_region', 'enumerate_polytope_vertices',
                     'compute_polytope_intrinsic_volume_from_vertices', '_compute_ratio_intervals_from_vertices'):
            sentinel = RuntimeError('original ratio hook override ' + hook)
            with self.subTest(hook=hook), patch.object(ratios, hook, side_effect=sentinel):
                for analyze in (compute_candidate_analysis_for_mode, terminal.compute_terminal_candidate_analysis_for_mode):
                    with self.assertRaises(RuntimeError) as raised:
                        analyze(alternatives, space, None, self.config, True)
                    self.assertIs(raised.exception, sentinel)
        with patch.object(terminal._values, 'ratio_interval_has_positive_width', return_value=False):
            expected = compute_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
            actual = terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
            self.assertEqual(asdict(actual), asdict(expected))
        interval_dict = {
            'lower': {'status': 'optimal', 'objective_sense': 'min', 'optimal_value': 0.},
            'upper': {'status': 'optimal', 'objective_sense': 'max', 'optimal_value': 1.},
        }
        with patch.object(ratios, 'enumerate_polytope_vertices', return_value=PolytopeVertices('lower_dimensional', np.empty((0, 2)), 1)), \
             patch.object(ratios, 'compute_ratio_interval_for_candidate', return_value=interval_dict):
            expected = compute_candidate_analysis_for_mode(alternatives, space, None, self.config)
            actual = terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, None, self.config)
            self.assertEqual(asdict(actual), asdict(expected))

    def test_original_geometry_alias_keyword_only_callback_is_preserved(self):
        alternatives = AlternativenMatrix(entries=[[1., 0.], [0., 1.]])
        space = build_weight_space(2, [])
        original = ratios.enumerate_polytope_vertices
        calls = []
        def keyword_only(*, system, tolerance):
            calls.append((system, tolerance))
            return original(system=system, tolerance=tolerance)
        with patch.object(ratios, 'enumerate_polytope_vertices', side_effect=keyword_only):
            expected = compute_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
            actual = terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
        self.assertEqual(summary(actual), summary(expected))
        self.assertEqual(len(calls), 4)

    def test_non_geometry_and_closed_lp_delegate_unchanged(self):
        alternatives = AlternativenMatrix(entries=[[1., 0.], [0., 1.]])
        space = build_weight_space(2, [])
        for config in (replace(self.config, ratio_interval_engine='lp'),
                       replace(self.config, query_efficiency_stop_aware=False, candidate_count_mode='closed_lp')):
            expected = compute_candidate_analysis_for_mode(alternatives, space, None, config, True)
            actual = terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, None, config, True)
            self.assertEqual(asdict(actual), asdict(expected))

    def test_known_thin_corrected_state_volume_hex_is_unchanged(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/wrong_confidence_three_p001.json').read_text())
        alternatives = AlternativenMatrix(entries=fixture['case']['alternatives'])
        answers = [AnsweredQuery(
            ziel_index_a=answer['goal_index_a'], ziel_index_b=answer['goal_index_b'],
            value=answer['value'], operator=answer['answer'],
        ) for answer in fixture['answers']]
        space = build_weight_space(3, answers)
        expected = compute_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
        actual = terminal.compute_terminal_candidate_analysis_for_mode(alternatives, space, None, self.config, True)
        self.assertEqual(summary(actual), summary(expected))
        self.assertIn(3, actual.candidates)
        self.assertGreater(actual.candidate_volumes[3], 0.)

    def test_h1_worker_matches_full_terminal_analysis_entire_proof(self):
        alternatives = AlternativenMatrix(entries=[[1., 0.], [48./49., 1.]])
        space = build_weight_space(2, [])
        payload = (alternatives, [], space, Query(ziel_index_a=0, ziel_index_b=1, value=197./3.), None, 1,
                   self.config, [0, 1], None, 'ratio')
        with patch.object(qe, "compute_terminal_candidate_analysis_for_mode", side_effect=compute_candidate_analysis_for_mode):
            old = qe._evaluate_efficiency_worker(payload)
        new = qe._evaluate_efficiency_worker(payload)
        self.assertEqual(hexify(asdict(old[0])), hexify(asdict(new[0])))
        self.assertEqual(hexify(old[1:]), hexify(new[1:]))

    def test_sixteen_saved_5d_h1_queries_keep_all_terminal_proof_bits(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/terminal_summary_five_p001.json').read_text())
        alternatives = AlternativenMatrix(entries=fixture['alternatives'])
        answers = [AnsweredQuery(
            ziel_index_a=answer['goal_index_a'], ziel_index_b=answer['goal_index_b'],
            value=answer['value'], operator=answer['answer'],
        ) for answer in fixture['initial_answered_queries']]
        space = build_weight_space(5, answers)
        for item in fixture['root_queries']:
            query = Query(ziel_index_a=item['ziel_index_a'], ziel_index_b=item['ziel_index_b'],
                          value=float.fromhex(item['value_hex']))
            payload = (alternatives, answers, space, query, None, 1, self.config,
                       fixture['candidate_subset'], None, 'ratio')
            with self.subTest(query=query):
                with patch.object(qe, "compute_terminal_candidate_analysis_for_mode", side_effect=compute_candidate_analysis_for_mode):
                    old = qe._evaluate_efficiency_worker(payload)
                new = qe._evaluate_efficiency_worker(payload)
                self.assertEqual(hexify(asdict(old[0])), hexify(asdict(new[0])))
                self.assertEqual(hexify(old[1:]), hexify(new[1:]))


if __name__ == '__main__':
    unittest.main()
