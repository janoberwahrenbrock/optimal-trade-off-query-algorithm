from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_array

from multistep.src import reusable_highs as new
from multistep.src import linear_programming as public

# Reference cleaning is the installed SciPy function, not another fast path.
# Reference native solve changes only the LP-preparation entry to that cleaner;
# separate states preserve identical cold native-model reuse across objectives.
old = new


def bits(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.strides, value.tobytes())
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, dict):
        return {key: bits(item) for key, item in value.items()}
    return value


def model():
    return {'c': [1., -.25, .5], 'A_ub': np.asarray([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]),
            'b_ub': np.asarray([0., 0., 0., .8]), 'A_eq': np.asarray([[1., 1., 1.]]), 'b_eq': np.asarray([1.]),
            'bounds': [(None, None)] * 3, 'method': 'highs', 'options': public.LINPROG_OPTIONS}


class ValidatedLPCleaningTests(unittest.TestCase):
    def setUp(self):
        new.clear_reusable_highs_cache()
        self.reference_state=new._State()
        self.candidate_state=new._State()

    def compare_clean(self, kwargs):
        expected = old._clean_inputs(old._LPProblem(kwargs['c'], kwargs.get('A_ub'), kwargs.get('b_ub'),
                                  kwargs.get('A_eq'), kwargs.get('b_eq'), kwargs.get('bounds'), None, None))
        actual = new._clean_private_inputs(kwargs)
        self.assertEqual([bits(x) for x in expected], [bits(x) for x in actual])
        for previous, current in zip(expected[1:6], actual[1:6]):
            a, b = old._constraint_matrix_csc(expected[1], expected[3]), new._constraint_matrix_csc(actual[1], actual[3])
            self.assertEqual(tuple(bits(getattr(a, n)) for n in ('data', 'indices', 'indptr')),
                             tuple(bits(getattr(b, n)) for n in ('data', 'indices', 'indptr')))
        return actual

    def compare_solve(self, kwargs):
        with patch.object(new,'_clean_private_inputs',side_effect=lambda arguments:new._ORIGINAL_CLEAN_INPUTS(new._problem(arguments))):
            expected=new._solve(self.reference_state,kwargs,compact_result=True,_internal_continuous_check=True)
        actual=new._solve(self.candidate_state,kwargs,compact_result=True,_internal_continuous_check=True)
        self.assertEqual(bits(dict(expected)),bits(dict(actual)))

    def test_cleaned_arrays_signed_zero_layout_and_changing_objective(self):
        kwargs = model(); kwargs['A_ub'][0, 1] = -0.0
        for c in ([1., -.25, .5], [-1., 0., -0.0], np.asarray([0., 0., 0.]), (2., -1., 1.)):
            kwargs['c'] = c; self.compare_clean(kwargs); self.compare_solve(kwargs)
        self.assertEqual(set(vars(new._state())),set(vars(self.reference_state)))

    def test_degenerate_c_and_f_contiguous_layout_copies_preserve_strides(self):
        kwargs=model()
        kwargs['A_ub']=np.array([[1.,0.,0.]],order='F')
        kwargs['b_ub']=np.array([.8])
        self.assertTrue(kwargs['A_ub'].flags.c_contiguous)
        self.assertEqual(kwargs['A_ub'].strides,(8,8))
        self.compare_clean(kwargs);self.compare_solve(kwargs)
        kwargs=dict(c=np.array([1.]),A_eq=np.array([[1.]],order='F'),b_eq=np.array([1.]),
                    bounds=np.array([[0.,2.]],order='F'),method='highs',options=public.LINPROG_OPTIONS)
        self.compare_clean(kwargs);self.compare_solve(kwargs)
        kwargs=dict(c=np.array([1.,0.,0.]),A_ub=np.empty((0,3)),b_ub=np.empty(0),bounds=None,
                    method='highs',options=public.LINPROG_OPTIONS)
        self.compare_clean(kwargs);self.compare_solve(kwargs)

    def test_actual_native_center_lps_and_multiple_statuses_methods_options(self):
        rng = np.random.default_rng(819)
        for idx in range(12):
            a = np.vstack((np.eye(4), -np.eye(4), rng.normal(size=(3, 4))))
            kwargs = dict(c=np.r_[np.zeros(4), -1.], A_ub=np.column_stack((a, np.linalg.norm(a, axis=1))),
                          b_ub=rng.uniform(.2, 2., len(a)), bounds=[(None, None)] * 4 + [(0., None)],
                          method='highs', options=public.LINPROG_OPTIONS)
            for _ in range(2): self.compare_clean(kwargs); self.compare_solve(kwargs)
        for method in ('highs', 'highs-ds', 'highs-ipm'):
            for presolve in (True, False):
                for kwargs in (dict(c=np.array([1.]), A_ub=np.array([[1.], [-1.]]), b_ub=np.array([0., -1.]), bounds=[(None, None)]),
                               dict(c=np.array([-1.]), bounds=[(None, None)]),
                               dict(c=[1.], A_eq=np.array([[1.]]), b_eq=np.array([1e-9]), bounds=[(None, None)])):
                    kwargs.update(method=method, options={**public.LINPROG_OPTIONS, 'presolve': presolve})
                    for _ in range(2): self.compare_clean(kwargs); self.compare_solve(kwargs)

    def test_constraint_rhs_and_bounds_mutation_freshly_prepared(self):
        kwargs=model(); self.compare_clean(kwargs)
        for name, index in [('A_ub',(0,0)), ('b_ub',0), ('A_eq',(0,0)), ('b_eq',0)]:
            kwargs[name][index] += .001
            self.compare_clean(kwargs); self.compare_solve(kwargs)
        kwargs['bounds']=np.array([[0., .7],[0., .7],[0., .7]])
        self.compare_clean(kwargs); kwargs['bounds'][0,1]=.6
        self.compare_clean(kwargs); self.compare_solve(kwargs)

    def test_original_cleaner_exception_type_message_unchanged(self):
        for update in ({'c':[np.nan,0.,0.]}, {'c':np.array([np.inf,0.,0.])}, {'c':[]}, {'c':[[1.,1.],[1.,1.]]},
                       {'A_ub':np.array([[np.nan,0.,0.]])}, {'b_eq':np.array([np.inf])},
                       {'bounds':[(0.,None)] * 2}, {'bounds':[[0.,'bad']]*3}):
            kwargs={**model(),**update}
            errors=[]
            for fn in (lambda: old._clean_inputs(old._LPProblem(kwargs['c'],kwargs.get('A_ub'),kwargs.get('b_ub'),kwargs.get('A_eq'),kwargs.get('b_eq'),kwargs.get('bounds'),None,None)),
                       lambda:new._clean_private_inputs(kwargs)):
                try:fn()
                except Exception as exc:errors.append((type(exc),str(exc)))
                else:self.fail('expected original validation failure')
            self.assertEqual(errors[0],errors[1])
        kwargs=model();kwargs['bounds']=np.array([[np.nan,np.nan]]*3)
        self.compare_clean(kwargs)

    def test_unsupported_subclass_layout_and_dtype_keep_original_cleaner(self):
        class Subarray(np.ndarray):pass
        for change in ({'A_ub':model()['A_ub'].view(Subarray)}, {'A_eq':np.asfortranarray(np.ones((2,3))), 'b_eq':np.ones(2)},
                       {'A_ub':model()['A_ub'].astype(np.float32)}, {'c':np.array([1.,0.,0.]).view(Subarray)},
                       {'c':np.array([1.,0.,0.],dtype=np.float32)}, {'bounds':(0.,None)},
                       {'A_ub':model()['A_ub'].tolist()}):
            new.clear_reusable_highs_cache();kwargs={**model(),**change};self.compare_clean(kwargs)
            with patch.object(new,'_problem',wraps=new._problem) as fallback:
                new._clean_private_inputs(kwargs)
                self.assertEqual(fallback.call_count,1)
        kwargs=model();kwargs['A_ub']=csc_array(kwargs['A_ub'])
        self.assertIsNone(new.try_reusable_highs(kwargs,compact_result=True,_internal_continuous_check=True))

    def test_cleaner_override_and_default_nonprivate_mode_keep_original_cleaner(self):
        kwargs=model();self.compare_clean(kwargs)
        expected=new._ORIGINAL_CLEAN_INPUTS(new._problem(kwargs))
        with patch.object(new,'_clean_inputs',wraps=new._ORIGINAL_CLEAN_INPUTS) as cleaner:
            actual=new._clean_private_inputs(kwargs)
            self.assertEqual([bits(v) for v in expected],[bits(v) for v in actual])
            self.assertEqual(cleaner.call_count,1)
        kwargs=model();new.clear_reusable_highs_cache()
        result=new.try_reusable_highs(kwargs)
        self.assertTrue(result.success)
        self.assertEqual(set(vars(new._state())),set(vars(self.reference_state)))
        with patch.object(public,'linprog',return_value=object()) as solver:
            self.assertIs(public._run_linprog(_internal_compact_result=True,**kwargs),solver.return_value)

    def test_standard_finite_checks_fresh_and_original_cache_state_reset(self):
        kwargs=model()
        with patch.object(new,'_problem',wraps=new._problem) as fallback:
            new._clean_private_inputs(kwargs)
            self.assertEqual(fallback.call_count,0)
            kwargs['A_ub'][0,0]=np.nan
            with self.assertRaises(ValueError):new._clean_private_inputs(kwargs)
            self.assertEqual(fallback.call_count,1)
        self.compare_solve(model())
        new._state().pid=-1
        self.assertEqual(new.reusable_highs_cache_info().currsize,0)
        self.assertEqual(set(vars(new._state())),set(vars(self.reference_state)))
        with patch.object(new,'_solve',side_effect=RuntimeError('native injected failure')):
            self.assertIsNone(new.try_reusable_highs(model(),compact_result=True,_internal_continuous_check=True))
        self.assertEqual(new.reusable_highs_cache_info().currsize,0)

    def test_thread_state_isolation(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        barrier=threading.Barrier(2)
        def worker(value):
            new.clear_reusable_highs_cache();barrier.wait();kwargs=model();kwargs['b_eq'][0]=value
            for _ in range(2):new.try_reusable_highs(kwargs,compact_result=True,_internal_continuous_check=True)
            return new.reusable_highs_cache_info().calls,new.reusable_highs_cache_info().model_hits
        with ThreadPoolExecutor(max_workers=2) as pool:self.assertEqual(list(pool.map(worker,(1.,2.))),[(2,1),(2,1)])
        self.assertEqual(new.reusable_highs_cache_info().calls,0)


if __name__=='__main__':unittest.main()
