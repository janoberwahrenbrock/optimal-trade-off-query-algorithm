from __future__ import annotations
from decimal import Decimal
from fractions import Fraction
import copy,json,pickle
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from multistep.src import exact_geometry_recovery as recovery
from multistep.src import polytope_volume as volume
from multistep.src import ratio_intervals as ratios
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import AlternativenMatrix,AnsweredQuery,Query
from multistep.src.optimality_region import build_optimality_region
from multistep.src.polytope_geometry import PolytopeVertices
from multistep.src.weight_space import build_weight_space
from multistep.optimized import value_function as vf


def simplex(n):return build_weight_space(n,[])
def key(system):return recovery._system_key(system)
def geometric(status='lower_dimensional',n=3,message=None):
    return PolytopeVertices(status,np.empty((0,n)),n-1,1e-12,message)


class VolumeAmbiguityRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=Path(__file__).parent/'fixtures'
        cls.fixture=json.loads((directory/'wrong_confidence_three_p001.json').read_text())
        cls.parent_partition=json.loads((directory/'three_p003_parent_partition.json').read_text())
        cls.alternatives=AlternativenMatrix(entries=cls.fixture['case']['alternatives'])
        cls.answers=[AnsweredQuery(ziel_index_a=q['goal_index_a'],ziel_index_b=q['goal_index_b'],value=q['value'],operator=q['answer']) for q in cls.fixture['answers']]
        cls.space=build_weight_space(3,cls.answers)
    def setUp(self):volume.clear_polytope_volume_cache()

    def test_all_final_regions_match_independent_exact_clipping(self):
        for candidate in range(5):
            with self.subTest(candidate=candidate):
                system=build_optimality_region(self.alternatives,self.space,candidate)
                result=recovery.recover_intrinsic_volume(key(system))
                expected=self.fixture['independent_exact_regions']['regions'][str(candidate)]
                self.assertEqual(Fraction(result['exact_projected_measure']),Fraction(expected['projected_area_exact']))
                self.assertEqual(result['intrinsic_volume_float'].hex(),float(Decimal(expected['intrinsic_area_decimal'])).hex())

    def test_native_lower_dimensional_true_winner_gets_positive_original_area(self):
        system=build_optimality_region(self.alternatives,self.space,3)
        value=volume.compute_polytope_intrinsic_volume_from_vertices(system,geometric())
        expected=self.fixture['independent_exact_regions']['regions']['3']
        self.assertEqual(value.hex(),float(Decimal(expected['intrinsic_area_decimal'])).hex())
        self.assertTrue(getattr(value,'_exact_lowdimensional_volume_correction',False))

    def test_final_state_must_not_trigger99percent_confidence(self):
        config=vf.OptimizedMultistepConfig(answer_probability_mode='exact_volume',candidate_count_mode='ratio_relevant',volume_confidence_threshold=.99)
        analysis=vf.compute_candidate_analysis_for_mode(self.alternatives,self.space,[1,3],config,include_candidate_volumes=True)
        self.assertEqual(analysis.candidates,[1,3])
        self.assertAlmostEqual(analysis.candidate_volume_shares[1],.7731145704562179,places=6)
        self.assertAlmostEqual(analysis.candidate_volume_shares[3],.22688542954378216,places=6)
        self.assertIsNone(vf.evaluate_volume_confidence_termination(analysis,.99))

    def test_supported_native_infeasible_and_error_are_checked_exactly(self):
        for status in ['infeasible','error']:
            with self.subTest(status=status):
                value=volume.compute_polytope_intrinsic_volume_from_vertices(simplex(3),geometric(status,message='row-normalized center LP failed'))
                self.assertGreater(value,0.)
                self.assertTrue(getattr(value,'_exact_lowdimensional_volume_correction',False))

    def test_genuine_zero_area_is_not_replaced_by_implicit_line_length(self):
        system=simplex(3);system.add_inequality([1.,0.,0.],0.)
        result=recovery.recover_intrinsic_volume(key(system));self.assertEqual(result['dimension'],2)
        self.assertEqual(result['exact_projected_measure'],'0/1');self.assertEqual(result['intrinsic_volume_float'],0.)
        value=volume.compute_polytope_intrinsic_volume_from_vertices(system,geometric())
        self.assertIs(type(value),float);self.assertEqual(value,0.)

    def test_explicit_interval_and_point_measure_contract(self):
        system=simplex(3);system.add_equality([1.,0.,0.],0.)
        system.add_inequality([0.,1.,0.],.5);system.add_inequality([0.,-1.,0.],-.25)
        value=recovery.recover_intrinsic_volume(key(system));self.assertEqual(value['dimension'],1)
        self.assertAlmostEqual(value['intrinsic_volume_float'],np.sqrt(2)*.25)
        system.add_equality([0.,1.,0.],.375)
        self.assertEqual(recovery.recover_intrinsic_volume(key(system))['intrinsic_volume_float'],0.)

    def test_higher_dimensional_zero_declines_before_fraction_conversion(self):
        with patch.object(recovery,'_fraction',side_effect=AssertionError('must cheaply decline')):
            value=volume.compute_polytope_intrinsic_volume_from_vertices(simplex(5),geometric(n=5))
        self.assertEqual(value,0.);self.assertIs(type(value),float)

    def test_unsupported_native_exception_keeps_original_identity(self):
        original=volume._tag_native_geometry_volume_error(RuntimeError('row-normalized center LP failed: original'))
        with patch.object(volume,'_compute_polytope_intrinsic_volume_cached',side_effect=original):
            with self.assertRaises(RuntimeError) as raised:
                volume.compute_exact_query_answer_probabilities(simplex(4),Query(ziel_index_a=0,ziel_index_b=1,value=1.))
        self.assertIs(raised.exception,original)
        self.assertIsInstance(raised.exception.__cause__,recovery.RecoveryNotApplicable)

    def test_parent_less_greater_volume_exceptions_repartition_common_parent(self):
        pk=self.parent_partition['geometry']['parent']['constraint_key']
        system=LinearConstraintSystem(inequalities_left_side=pk[0],inequalities_right_side=pk[1],equalities_left_side=pk[2],equalities_right_side=pk[3])
        q=self.parent_partition['query'];query=Query(ziel_index_a=q['goal_index_a'],ziel_index_b=q['goal_index_b'],value=q['value'])
        for phase in range(3):
            with self.subTest(phase=phase):
                original=volume._tag_native_geometry_volume_error(RuntimeError('row-normalized center LP failed: captured-class synthetic exception'))
                effects=[1.]*phase+[original]
                with patch.object(volume,'_compute_polytope_intrinsic_volume_cached',side_effect=effects):
                    probabilities=volume.compute_exact_query_answer_probabilities(system,query)
                for answer,value in self.parent_partition['probabilities_exact'].items():
                    self.assertEqual(probabilities[answer].hex(),float(Fraction(value)).hex())
                self.assertTrue(getattr(probabilities,'_exact_lowdimensional_recovery',False))

    def test_arbitrary_enumerator_callback_error_propagates_unchanged(self):
        original=RuntimeError('arbitrary callback failure')
        with patch.object(volume,'enumerate_polytope_vertices',side_effect=original):
            with self.assertRaises(RuntimeError) as raised:
                volume.compute_exact_query_answer_probabilities(simplex(3),Query(ziel_index_a=0,ziel_index_b=1,value=1.))
        self.assertIs(raised.exception,original)
        self.assertFalse(getattr(original,'_otqa_native_geometry_volume_error',False))

    def test_genuine_empty_inequalities_and_inconsistent_equalities_keep_zero(self):
        for equality in [False,True]:
            with self.subTest(equality=equality):
                system=simplex(3)
                if equality:system.add_equality([1.,1.,1.],2.)
                else:system.add_inequality([1.,1.,1.],-1.)
                value=volume.compute_polytope_intrinsic_volume_from_vertices(system,geometric('infeasible'))
                self.assertEqual(value,0.);self.assertIs(type(value),float)

    def test_positive_exact_volume_underflow_remains_explicit_error(self):
        system=simplex(3);system.add_inequality([1.,0.,0.],1e-200);system.add_inequality([0.,1.,0.],1e-200)
        with self.assertRaisesRegex(RuntimeError,'ambiguous zero intrinsic volume could not be certified'):
            volume.compute_polytope_intrinsic_volume_from_vertices(system,geometric())

    def test_positive_volume_correction_uses_exact_partition_not_mixed_native_values(self):
        with patch.object(volume,'_compute_polytope_intrinsic_volume_cached',side_effect=[1.,volume._ExactRecoveryVolume(.1),.8]):
            result=volume.compute_exact_query_answer_probabilities(simplex(3),Query(ziel_index_a=0,ziel_index_b=1,value=1.))
        self.assertEqual(result,{'<':.5,'=':0.,'>':.5})
        self.assertTrue(getattr(result,'_exact_lowdimensional_recovery',False))

    def test_confirmed_genuine_zero_leaves_ordinary_probability_path_identical(self):
        with patch.object(recovery,'recover_probabilities_from_system',side_effect=AssertionError('genuine zero must not force repartition')),patch.object(volume,'_compute_polytope_intrinsic_volume_cached',side_effect=[1.,0.,1.]):
            result=volume.compute_exact_query_answer_probabilities(simplex(3),Query(ziel_index_a=0,ziel_index_b=1,value=1.))
        self.assertIs(type(result),dict);self.assertEqual(result,{'<':0.,'=':0.,'>':1.})

    def test_positive_native_volume_does_not_call_recovery_or_change_bits(self):
        poly=PolytopeVertices('full_dimensional',np.eye(3),2,1.)
        expected=float.fromhex('0x1.23456789abcdep-8')
        with patch.object(recovery,'recover_intrinsic_volume_from_system',side_effect=AssertionError('ordinary positive volume path')),patch.object(volume,'ConvexHull',return_value=SimpleNamespace(volume=expected)):
            actual=volume.compute_polytope_intrinsic_volume_from_vertices(simplex(3),poly)
        self.assertEqual(actual.hex(),expected.hex());self.assertIs(type(actual),float)

    def test_recovered_volume_marker_survives_cache_copy_serialization(self):
        poly=PolytopeVertices('full_dimensional',np.eye(3),2,1.)
        with patch.object(volume,'enumerate_polytope_vertices',return_value=poly),patch.object(volume,'ConvexHull',return_value=SimpleNamespace(volume=0.)),patch.object(volume,'_remember_polytope_vertices',side_effect=AssertionError('unreliable native vertices must not be cached')):
            first=volume.compute_polytope_intrinsic_volume(simplex(3));second=volume.compute_polytope_intrinsic_volume(simplex(3))
        self.assertIs(first,second)
        for value in [copy.copy(first),copy.deepcopy(first),pickle.loads(pickle.dumps(first))]:
            self.assertEqual(value.hex(),first.hex());self.assertTrue(getattr(value,'_exact_lowdimensional_volume_correction',False))
        self.assertEqual(json.loads(json.dumps(first)),float(first))

    def test_positive_corrected_infeasible_region_retains_lp_ratio_fallback(self):
        alternatives=AlternativenMatrix(entries=[[1.,0.,0.],[0.,1.,0.]])
        with patch.object(ratios,'enumerate_polytope_vertices',return_value=geometric('infeasible')),patch.object(ratios,'_build_infeasible_ratio_interval',side_effect=AssertionError('positive exact region is not infeasible')):
            analysis=ratios.compute_all_ratio_interval_analysis(alternatives,simplex(3),[0],include_candidate_volumes=True)
        self.assertGreater(analysis.candidate_volumes[0],0.)
        self.assertTrue(all(pair.intervals_by_candidate[0].lower.status=='optimal' for pair in analysis.ratio_intervals))

if __name__=='__main__':unittest.main()
