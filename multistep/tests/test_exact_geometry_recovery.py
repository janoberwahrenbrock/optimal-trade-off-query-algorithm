from fractions import Fraction
from dataclasses import replace
import copy
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from multistep.src import exact_geometry_recovery as recovery
from multistep.src import polytope_volume as volume
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import AlternativenMatrix, Query
from multistep.src.models import AnsweredQuery
from multistep.src.models.linear_optimization_result import LinearOptimizationResult
from multistep.src.ratio_intervals import GoalPairRatioIntervals, RatioInterval
from multistep.src.value_function import ValueFunctionResult
from multistep.optimized import value_function as vf
from multistep.optimized import query_efficiency as qe


def simplex(count):
    return [[[ -float(i == j) for j in range(count)] for i in range(count)],
            [0.] * count, [[1.] * count], [1.]]


def system(key):
    return LinearConstraintSystem(inequalities_left_side=key[0], inequalities_right_side=key[1],
                                  equalities_left_side=key[2], equalities_right_side=key[3])


class ExactGeometryRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent / 'fixtures/thin_simplex_recovery.json').read_text())

    def setUp(self):
        volume.clear_polytope_volume_cache()

    def test_captured_failure_matches_exact_independent_probabilities(self):
        result = recovery.recover_query_probabilities(self.fixture['parent_constraint_key'], self.fixture['query'])
        self.assertEqual(result['exact_probabilities'], self.fixture['exact_probabilities'])
        self.assertEqual(result['probabilities']['>'].hex(), float(Fraction(self.fixture['exact_probabilities']['>'])).hex())
        self.assertGreater(result['probabilities']['>'], 0.)

    def test_integrated_two_zero_children_recovers_tiny_positive_probability(self):
        result = volume.compute_exact_query_answer_probabilities(
            system(self.fixture['parent_constraint_key']), Query(**self.fixture['query']),
        )
        self.assertEqual(result['>'].hex(), float(Fraction(self.fixture['exact_probabilities']['>'])).hex())
        self.assertEqual(result['<'].hex(), float(Fraction(self.fixture['exact_probabilities']['<'])).hex())

    def test_zero_parent_failure_recovers_from_original_positive_geometry(self):
        with patch.object(volume, '_compute_polytope_intrinsic_volume_cached', return_value=0.):
            result = volume.compute_exact_query_answer_probabilities(system(simplex(3)), Query(ziel_index_a=0,ziel_index_b=1,value=1.))
        self.assertEqual(result, {'<': .5, '=': 0., '>': .5})

    def test_ordinary_success_keeps_float_hex_and_does_not_call_recovery(self):
        with patch.object(recovery, 'recover_probabilities_from_system', side_effect=AssertionError('ordinary path must not invoke recovery')):
            result = volume.compute_exact_query_answer_probabilities(system(simplex(3)), Query(ziel_index_a=0,ziel_index_b=1,value=1.))
            deterministic = volume.compute_exact_query_answer_probabilities(system(simplex(3)), Query(ziel_index_a=0,ziel_index_b=1,value=0.))
        # Native hull rounding is retained rather than replaced by rational .5.
        self.assertAlmostEqual(result['<'], .5)
        self.assertAlmostEqual(result['>'], .5)
        self.assertEqual({answer: value.hex() for answer,value in deterministic.items()}, {'<': (0.).hex(), '=': (0.).hex(), '>': (1.).hex()})

    def test_unsupported_high_dimension_keeps_original_failure_explicit(self):
        with patch.object(volume, '_compute_polytope_intrinsic_volume_cached', return_value=0.):
            with self.assertRaisesRegex(RuntimeError, '^weight space has no positive intrinsic volume$') as raised:
                volume.compute_exact_query_answer_probabilities(system(simplex(4)), Query(ziel_index_a=0,ziel_index_b=1,value=1.))
        self.assertIsInstance(raised.exception.__cause__, recovery.RecoveryNotApplicable)

    def test_interval_forced_equality_and_explicit_point(self):
        key = simplex(3);key[2].append([1.,-.5,0.]);key[3].append(0.)
        result = recovery.recover_query_probabilities(key, {'ziel_index_a':0,'ziel_index_b':1,'value':.5})
        self.assertEqual(result['probabilities'], {'<':0.,'=':1.,'>':0.})
        key = simplex(3);key[2] += [[1.,0.,0.],[0.,1.,0.]];key[3] += [.25,.5]
        result = recovery.recover_query_probabilities(key, {'ziel_index_a':0,'ziel_index_b':1,'value':.5})
        self.assertEqual(result['dimension'],0)
        self.assertEqual(result['probabilities'], {'<':0.,'=':1.,'>':0.})

    def test_zero_area_and_inconsistent_equalities_remain_errors(self):
        key = simplex(3);key[0].append([1.,0.,0.]);key[1].append(0.)
        with self.assertRaises(recovery.RecoveryGeometryError):
            recovery.recover_query_probabilities(key, {'ziel_index_a':0,'ziel_index_b':1,'value':1.})
        key = simplex(3);key[2].append([1.,1.,1.]);key[3].append(2.)
        with self.assertRaises(recovery.RecoveryGeometryError):
            recovery.recover_query_probabilities(key, {'ziel_index_a':0,'ziel_index_b':1,'value':1.})

    def test_captured_empty_filter_is_certified_strict_singleton(self):
        fields = recovery.recover_strict_singleton_from_systems(
            parent_system=system(self.fixture['parent_constraint_key']),
            child_system=system(self.fixture['child_constraint_key']),
            alternatives=AlternativenMatrix(entries=self.fixture['alternatives']),
            parent_candidate_subset=self.fixture['parent_candidate_subset'],
            filtered_child_subset=[], probability=self.fixture['probability'],
            query=Query(**self.fixture['query']), answer=self.fixture['answer'],
        )
        self.assertEqual(fields['candidates'], [3])
        self.assertGreater(fields['candidate_volumes'][3], 0.)
        self.assertEqual(fields['candidate_volume_shares'], {3: 1.})
        self.assertIsNone(fields['ratio_intervals'])

    def test_singleton_guards_do_not_force_ties_or_missing_parent_candidates(self):
        fixtures = [([[1.,0.,0.],[0.,1.,0.]],[0,1],recovery.RecoveryNotApplicable),
                    ([[1.,1.,1.],[1.,1.,1.]],[0,1],recovery.RecoveryNotApplicable),
                    ([[1.,1.,1.],[0.,1.,0.]],[0,1],recovery.RecoveryNotApplicable),
                    ([[0.,0.,0.],[1.,1.,1.]],[0],recovery.RecoveryGeometryError)]
        for alternatives, subset, error in fixtures:
            with self.subTest(alternatives=alternatives), self.assertRaises(error):
                recovery.recover_singleton_after_empty_prefilter(simplex(3), alternatives, subset, [], .5)
        for filtered, probability in [([0],.5),([],0.)]:
            with self.assertRaises(recovery.RecoveryNotApplicable):
                recovery.recover_singleton_after_empty_prefilter(simplex(3), [[1.,1.,1.],[0.,0.,0.]], [0,1], filtered, probability)

    def test_marker_mapping_copy_and_serialization_contract(self):
        result = volume.compute_exact_query_answer_probabilities(system(self.fixture['parent_constraint_key']), Query(**self.fixture['query']))
        self.assertTrue(getattr(result, '_exact_lowdimensional_recovery', False))
        self.assertIs(vf.compute_exact_query_answer_probabilities, volume.compute_exact_query_answer_probabilities)
        for duplicate in (result.copy(),copy.copy(result),copy.deepcopy(result),pickle.loads(pickle.dumps(result))):
            self.assertTrue(getattr(duplicate, '_exact_lowdimensional_recovery', False))
            self.assertEqual({k:v.hex() for k,v in duplicate.items()},{k:v.hex() for k,v in result.items()})
        self.assertEqual(set(result), {'<','=','>'})
        self.assertEqual(json.dumps(result), json.dumps(dict(result)))
        ordinary = volume.compute_exact_query_answer_probabilities(system(simplex(3)), Query(ziel_index_a=0,ziel_index_b=1,value=1.))
        self.assertIs(type(ordinary),dict)

    def _plain_candidate_arguments(self, depth):
        q = Query(**self.fixture['query'])
        bad_interval = RatioInterval(
            lower=LinearOptimizationResult(status='optimal',objective_sense='min',optimal_value=0.),
            upper=LinearOptimizationResult(status='optimal',objective_sense='max',optimal_value=q.value-1.),
        )
        intervals = {(q.ziel_index_a,q.ziel_index_b):GoalPairRatioIntervals(
            goal_index_a=q.ziel_index_a,goal_index_b=q.ziel_index_b,
            intervals_by_candidate={candidate:bad_interval for candidate in self.fixture['parent_candidate_subset']},
        )}
        return dict(alternatives=AlternativenMatrix(entries=self.fixture['alternatives']),
            answered_queries=[AnsweredQuery(**answer) for answer in self.fixture['parent_answered_queries']],
            weight_space=system(self.fixture['parent_constraint_key']),query=q,samples=None,
            remaining_depth=depth,config=vf.OptimizedMultistepConfig(answer_probability_mode='exact_volume',use_ratio_terminal_counts=True),
            candidate_subset=self.fixture['parent_candidate_subset'],ratio_intervals_by_goal_pair=intervals)

    def test_plain_fixed_recovered_empty_branch_is_absorbing_singleton(self):
        observations=[]
        def ordinary_child(**kwargs):
            self.assertNotEqual(kwargs['candidate_subset'],[])
            return ValueFunctionResult(kwargs['remaining_depth'],2.,None,2,(),True)
        with patch.object(vf,'compute_value_function_optimized',side_effect=ordinary_child):
            evaluation = vf.evaluate_query_candidate_optimized(**self._plain_candidate_arguments(2),
                _child_result_observer=lambda answer,probability,result:observations.append((answer,result)))
        branch=next(branch for branch in evaluation.branches if branch.answer=='>')
        self.assertGreater(branch.probability,0.)
        self.assertEqual(branch.child_candidate_count,1)
        self.assertEqual(branch.child_value,1.)
        child=next(result for answer,result in observations if answer=='>')
        self.assertIsNone(child.best_query)
        self.assertEqual(child.candidate_count,1)

    def test_plain_terminal_zero_after_native_fallback_is_not_kept_for_recovered_probability(self):
        with patch.object(vf,'compute_terminal_candidate_count_fallback',return_value=(0,True)):
            evaluation=vf.evaluate_query_candidate_optimized(**self._plain_candidate_arguments(1))
        branch=next(branch for branch in evaluation.branches if branch.answer=='>')
        self.assertEqual(branch.child_candidate_count,1)
        self.assertEqual(branch.child_value,1.)
        self.assertTrue(branch.is_child_feasible)

    def test_plain_unmarked_empty_prefilter_does_not_invoke_recovery(self):
        # Preserve the former fixed-planner calculation for ordinary mappings.
        probabilities = dict(volume.compute_exact_query_answer_probabilities(
            system(self.fixture['parent_constraint_key']), Query(**self.fixture['query']),
        ))
        def native_child(**kwargs):
            count = len(kwargs['candidate_subset'])
            return ValueFunctionResult(kwargs['remaining_depth'],float(count),None,count,(),True)
        with patch.object(vf,'compute_exact_query_answer_probabilities',return_value=probabilities), \
             patch.object(vf,'compute_value_function_optimized',side_effect=native_child), \
             patch.object(recovery,'recover_strict_singleton_from_systems',side_effect=AssertionError('unmarked branch must not invoke recovery')):
            evaluation=vf.evaluate_query_candidate_optimized(**self._plain_candidate_arguments(2))
        branch=next(branch for branch in evaluation.branches if branch.answer=='>')
        self.assertGreater(branch.probability,0.)
        self.assertEqual(branch.child_candidate_count,0)
        self.assertEqual(branch.child_value,0.)

    def test_marked_zero_branch_does_not_invoke_recovery_when_skip_zero_disabled(self):
        arguments=self._plain_candidate_arguments(2)
        arguments['config']=replace(arguments['config'],skip_zero_probability_branches=False)
        def native_child(**kwargs):
            count=len(kwargs['candidate_subset'])
            return ValueFunctionResult(kwargs['remaining_depth'],float(count),None,count,(),True)
        with patch.object(vf,'compute_value_function_optimized',side_effect=native_child), \
             patch.object(LinearConstraintSystem,'is_feasible',return_value=True), \
             patch.object(vf,'_recover_strict_singleton_child_analysis',wraps=vf._recover_strict_singleton_child_analysis) as certified:
            evaluation=vf.evaluate_query_candidate_optimized(**arguments)
        zero=next(branch for branch in evaluation.branches if branch.answer=='=')
        self.assertEqual(zero.probability,0.)
        self.assertEqual(zero.child_candidate_count,0)
        self.assertEqual(certified.call_count,1)
        self.assertEqual(certified.call_args.args[7],'>')
        self.assertGreater(certified.call_args.args[5],0.)

    def _horizon_three_payload(self, stop_aware):
        arguments=self._plain_candidate_arguments(3)
        config=vf.OptimizedMultistepConfig(
            answer_probability_mode='exact_volume',query_efficiency_stop_aware=stop_aware,
            volume_confidence_threshold=.99 if stop_aware else None,
        )
        return (arguments['alternatives'],arguments['answered_queries'],arguments['weight_space'],
                arguments['query'],None,3,config,arguments['candidate_subset'],
                arguments['ratio_intervals_by_goal_pair'],'ratio')

    @staticmethod
    def _unextended_frontier(**kwargs):
        analysis=kwargs['analysis']
        count=len(analysis.candidates)
        return (qe._ContinuationOption(
            None,float(count),float(count),0.,
            stop_reason='single_candidate' if count==1 else None,
            maximum_volume_share=1. if analysis.candidate_volume_shares is not None and count==1 else None,
        ),)

    def test_horizon_three_empty_branch_is_certified_in_both_modes(self):
        def native_nonempty_analysis(alternatives,space,subset,config,**kwargs):
            self.assertNotEqual(subset,[])
            return vf.CandidateAnalysis(candidates=list(subset),ratio_intervals=None)
        for stop_aware in (False,True):
            with self.subTest(stop_aware=stop_aware), \
                 patch.object(qe,'compute_candidate_analysis_for_mode',side_effect=native_nonempty_analysis), \
                 patch.object(qe,'_depth_two_child_frontier',side_effect=self._unextended_frontier):
                evaluation,cost,continuations=qe._evaluate_depth_three_query(self._horizon_three_payload(stop_aware))
            branch=next(branch for branch in evaluation.branches if branch.answer=='>')
            self.assertGreater(branch.probability,0.)
            self.assertEqual(branch.child_candidate_count,1)
            self.assertEqual(branch.child_value,1.)
            continuation=next(child for child in continuations if child.answer=='>')
            self.assertIsNone(continuation.evaluation)
            self.assertEqual(cost,1.)
            if stop_aware:
                self.assertEqual(continuation.maximum_volume_share,1.)

    def test_horizon_three_unresolved_empty_branch_keeps_original_failure(self):
        def native_nonempty_analysis(alternatives,space,subset,config,**kwargs):
            return vf.CandidateAnalysis(candidates=list(subset),ratio_intervals=None)
        with patch.object(qe,'compute_candidate_analysis_for_mode',side_effect=native_nonempty_analysis), \
             patch.object(qe,'_depth_two_child_frontier',side_effect=self._unextended_frontier), \
             patch.object(recovery,'recover_strict_singleton_from_systems',side_effect=recovery.RecoveryNotApplicable('unsupported affine dimension')):
            with self.assertRaisesRegex(RuntimeError,'^positive-probability branch has no valid candidates in horizon-three plan$') as raised:
                qe._evaluate_depth_three_query(self._horizon_three_payload(False))
        self.assertIsInstance(raised.exception.__cause__,recovery.RecoveryNotApplicable)


if __name__ == '__main__':
    unittest.main()
