from __future__ import annotations
from fractions import Fraction
import json
from math import comb
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.exact_lp_recovery import recover_exact_linear_minimum


def system(n, extra=(), equalities=()):
    result = LinearConstraintSystem()
    for i in range(n):
        result.add_inequality([float(-1 if j == i else 0) for j in range(n)], 0.)
    for row, right in extra: result.add_inequality(row, right)
    for row, right in equalities: result.add_equality(row, right)
    return result


def cert(result):
    assert result.certificate is not None, result.rejection_reason
    return result.certificate


def verify(point, rows, right):
    if point is None: return False
    return all(sum(Fraction.from_float(float(v))*x for v,x in zip(row,point)) <= Fraction.from_float(float(b)) for row,b in zip(rows,right))


class ExactLPRecoveryTests(unittest.TestCase):
    def test_captured_ratio_lp_exact_unbounded_certificate(self):
        payload=Path(__file__).resolve().parent/'fixtures'/'five_p098_ratio_lp.json'
        kwargs=json.loads(payload.read_text())['kwargs']
        original=LinearConstraintSystem(inequalities_left_side=kwargs['A_ub'],inequalities_right_side=kwargs['b_ub'],equalities_left_side=kwargs['A_eq'],equalities_right_side=kwargs['b_eq'])
        proof=cert(recover_exact_linear_minimum(system=original,objective=kwargs['c']))
        self.assertEqual(proof.status,'unbounded')
        self.assertEqual((proof.point_sets_tested,proof.recession_sets_tested),(715,286))
        self.assertEqual(proof.combinations_required,1001)
        self.assertTrue(verify(proof.point,kwargs['A_ub'],kwargs['b_ub']))
        self.assertTrue(verify(proof.recession_ray,kwargs['A_ub'],[0.]*13))
        self.assertEqual(proof.point[3],1); self.assertEqual(proof.recession_ray[3],0)
        self.assertEqual(sum(proof.recession_ray),1)
        self.assertLess(sum(Fraction.from_float(v)*x for v,x in zip(kwargs['c'],proof.recession_ray)),0)
        self.assertTrue(all(v>=0 for v in proof.recession_ray))
        self.assertFalse(verify(tuple(-v for v in proof.recession_ray),kwargs['A_ub'],[0.]*13))

    def test_bounded_simplex_minimum(self):
        proof=cert(recover_exact_linear_minimum(system=system(2,equalities=[([1.,1.],1.)]),objective=[1.,2.]))
        self.assertEqual((proof.status,proof.exact_objective,proof.point),('optimal',Fraction(1),(Fraction(1),Fraction(0))))
        self.assertEqual(proof.to_linear_result().optimal_value,1.)

    def test_infeasible_pointed_inequalities_complete_scan(self):
        proof=cert(recover_exact_linear_minimum(system=system(1,extra=[([1.],-1.)]),objective=[0.]))
        self.assertEqual(proof.status,'infeasible'); self.assertEqual(proof.point_sets_tested,comb(2,1)); self.assertEqual(proof.point_vertex_count,0)

    def test_degenerate_point_and_dependent_equalities(self):
        proof=cert(recover_exact_linear_minimum(system=system(2,equalities=[([1.,1.],0.),([2.,2.],0.)]),objective=[-1.,-1.]))
        self.assertEqual((proof.status,proof.exact_objective,proof.point),('optimal',Fraction(0),(Fraction(0),Fraction(0))))
        self.assertEqual(proof.point_vertex_count,1)

    def test_inconsistent_equalities_are_exactly_infeasible(self):
        proof=cert(recover_exact_linear_minimum(system=system(1,equalities=[([1.],1.),([1.],2.)]),objective=[0.]))
        self.assertEqual(proof.status,'infeasible'); self.assertTrue(proof.inconsistent_equalities); self.assertEqual(proof.affine_dimension,-1)

    def test_unbounded_polyhedron_bounded_objective(self):
        proof=cert(recover_exact_linear_minimum(system=system(1),objective=[1.]))
        self.assertEqual((proof.status,proof.exact_objective),('optimal',Fraction(0)))
        self.assertEqual(proof.recession_vertex_count,1)

    def test_nonpointed_or_missing_explicit_nonnegativity_rejected(self):
        original=LinearConstraintSystem(equalities_left_side=[[1.,0.]],equalities_right_side=[1.])
        attempt=recover_exact_linear_minimum(system=original,objective=[1.,0.])
        self.assertIsNone(attempt.certificate); self.assertIn('explicit homogeneous nonnegativity',attempt.rejection_reason)

    def test_high_dimension_rejected_before_enumeration(self):
        attempt=recover_exact_linear_minimum(system=system(6),objective=[0.]*6)
        self.assertIsNone(attempt.certificate); self.assertIn('dimension',attempt.rejection_reason)

    def test_over_limit_rejected_without_partial_enumeration(self):
        original=system(5,extra=[([1.,0.,0.,0.,0.],1.)]*40)
        with patch('multistep.src.exact_lp_recovery._vertices',side_effect=AssertionError('must not enumerate')):
            attempt=recover_exact_linear_minimum(system=original,objective=[1.,0.,0.,0.,0.])
        self.assertIsNone(attempt.certificate); self.assertIn('none enumerated',attempt.rejection_reason)

    def test_nonfinite_coefficient_rejected(self):
        attempt=recover_exact_linear_minimum(system=system(1),objective=[float('nan')])
        self.assertIsNone(attempt.certificate); self.assertIn('finite real',attempt.rejection_reason)

    def test_scaled_nonnegativity_keeps_binary_input_exact(self):
        original=LinearConstraintSystem(inequalities_left_side=[[-2.]],inequalities_right_side=[0.],equalities_left_side=[[3.]],equalities_right_side=[1.])
        proof=cert(recover_exact_linear_minimum(system=original,objective=[0.1]))
        self.assertEqual(proof.exact_objective,Fraction.from_float(0.1)/3)

    def test_finite_exact_optimum_overflow_and_underflow_rejected(self):
        for rhs, cost in [(1e308, 1e308), (1e-320, 1e-10)]:
            with self.subTest(rhs=rhs, cost=cost):
                original=system(1,equalities=[([1.],rhs)])
                attempt=recover_exact_linear_minimum(system=original,objective=[cost])
                self.assertIsNone(attempt.certificate)
                self.assertIn('representable',attempt.rejection_reason)
                native=SimpleNamespace(success=False,status=4,message='unclassified native failure')
                with patch('multistep.src.linear_constraints.run_linprog_with_retries',return_value=native),self.assertRaisesRegex(RuntimeError,'^linprog failed: unclassified native failure$') as raised:
                    original.minimize([cost])
                self.assertTrue(any('representable' in note for note in raised.exception.__notes__))

    def test_ordinary_native_results_skip_helper_and_match_output(self):
        for status in ['optimal','infeasible','unbounded']:
            with self.subTest(status=status):
                native=SimpleNamespace(success=status=='optimal',status={'optimal':0,'infeasible':2,'unbounded':3}[status],message='native unchanged',x=[1.] if status=='optimal' else None,fun=1. if status=='optimal' else None)
                with patch('multistep.src.linear_constraints.run_linprog_with_retries',return_value=native),patch('multistep.src.exact_lp_recovery.recover_exact_linear_minimum',side_effect=AssertionError('ordinary path called exact helper')):
                    result=system(1).minimize([1.])
                self.assertEqual(result.status,status); self.assertEqual(result.solver_message,'native unchanged'); self.assertEqual(result.solver_status_code,native.status)
                self.assertEqual(result.optimal_value,1. if status=='optimal' else None)

    def test_exhausted_unknown_failure_recovers_min_and_max_senses(self):
        native=SimpleNamespace(success=False,status=4,message='(HiGHS Status 0: Not Set)')
        original=system(2,equalities=[([1.,1.],1.)])
        with patch('multistep.src.linear_constraints.run_linprog_with_retries',return_value=native):
            lower=original.minimize([1.,2.]); upper=original.maximize([1.,2.])
        self.assertEqual((lower.status,lower.objective_sense,lower.optimal_value),('optimal','min',1.))
        self.assertEqual((upper.status,upper.objective_sense,upper.optimal_value),('optimal','max',2.))

    def test_unknown_unsupported_failure_keeps_original_error_and_note(self):
        native=SimpleNamespace(success=False,status=4,message='(HiGHS Status 0: Not Set)')
        with patch('multistep.src.linear_constraints.run_linprog_with_retries',return_value=native),self.assertRaisesRegex(RuntimeError,'^linprog failed: \\(HiGHS Status 0: Not Set\\)$') as raised:
            system(6).minimize([0.]*6)
        self.assertTrue(any('exact LP recovery unavailable' in note for note in raised.exception.__notes__))


if __name__=='__main__':unittest.main()
