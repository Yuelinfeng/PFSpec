#!/usr/bin/env python3
"""CPU-only numerical-parity and microbenchmark for GaussianGPT HGB Ranker.

Reads the frozen sklearn HistGradientBoostingClassifier and the existing on-policy
Position corpus. Compiles a *numeric-split, binary-classification only* scorer
with Numba; rejects unsupported tree configurations instead of making guesses.
No Target checkpoint or GPU is needed. JIT compile time excluded from benchmark.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from joblib import load
from threadpoolctl import threadpool_limits

try:
    from numba import njit
except ImportError as e:
    raise SystemExit("Numba is not installed. Install compatible numba into gaussiangpt env.") from e

from eval_lean_ranker_cpu import build_examples


@njit(cache=True)
def _predict_raw_numeric_binary(X, offsets, fidx, thresh, missing_left,
                                left, right, is_leaf, value, baseline):
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        score = baseline
        for k in range(len(offsets)):
            offset = int(offsets[k])
            at = offset
            while is_leaf[at] == 0:
                a = X[i, fidx[at]]
                if np.isnan(a):
                    go_left = missing_left[at] != 0
                else:
                    go_left = a <= thresh[at]
                at = offset + (int(left[at]) if go_left else int(right[at]))
            score += value[at]
        out[i] = score
    return out


class FastHistRanker:
    """Drop-in predict_proba scorer; exact numeric-only 2-class HGB trees."""
    def __init__(self, clf):
        if type(clf).__name__ != 'HistGradientBoostingClassifier':
            raise TypeError(f'Expected HistGradientBoostingClassifier, got {type(clf)}')
        if len(clf.classes_) != 2 or int(clf.n_trees_per_iteration_) != 1:
            raise ValueError('Only binary HistGradientBoosting supported')
        if getattr(clf, 'is_categorical_', None) is not None and np.any(clf.is_categorical_):
            raise ValueError('Categorical HGB trees unsupported by fast numerical scorer')
        predictors = [p[0] for p in clf._predictors]
        if not predictors:
            raise ValueError('Empty model')
        for p in predictors:
            if bool(np.any(p.nodes['is_categorical'])):
                raise ValueError('Categorical split present; refusing to optimize')
        counts = [len(p.nodes) for p in predictors]
        self.offsets = np.ascontiguousarray(np.r_[0, np.cumsum(counts)[:-1]], dtype=np.int32)
        self._node_arrays = []
        fields = [
            ('feature_idx', np.int32), ('num_threshold', np.float64),
            ('missing_go_to_left', np.uint8), ('left', np.int32),
            ('right', np.int32), ('is_leaf', np.uint8), ('value', np.float64),
        ]
        for field, dtype in fields:
            self._node_arrays.append(np.ascontiguousarray(np.concatenate([
                p.nodes[field] for p in predictors
            ]).astype(dtype, copy=False)))
        self.baseline = float(np.asarray(clf._baseline_prediction).reshape(-1)[0])
        self.n_features = int(clf.n_features_in_)

    def raw(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f'Expected (n,{self.n_features}) features, got {x.shape}')
        return _predict_raw_numeric_binary(
            x, self.offsets, *self._node_arrays, self.baseline
        )

    def predict_proba(self, x):
        raw = self.raw(x)
        # sigmoid is monotonic, but keep complete predict_proba API for drop-in use.
        prob = np.empty_like(raw)
        pos = raw >= 0
        prob[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
        exp_r = np.exp(raw[~pos])
        prob[~pos] = exp_r / (1.0 + exp_r)
        return np.column_stack((1.0 - prob, prob))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranker', type=Path, default=Path(
        'profiling/target_eval/results/ranker_holdout_81001_81016/'
        'ranker_trained_on_61001_61016.joblib'))
    ap.add_argument('--corpus', type=Path, default=Path(
        'profiling/target_eval/results/ranker_holdout_81001_81016/target_16x1024.pt'))
    args = ap.parse_args()
    original = load(args.ranker)
    fast = FastHistRanker(original)
    examples, _, _ = build_examples(args.corpus)
    multi = [e['features'] for e in examples if len(e['ids']) > 1]
    if not multi:
        raise AssertionError('No multi-candidate contexts')
    print(f'Model trees={len(fast.offsets)} nodes={len(fast._node_arrays[0])} '
          f'features={fast.n_features}')
    print(f'Positions={len(examples)} multi-candidate contexts={len(multi)}')
    print('Warmup including Numba compilation (excluded)...', flush=True)
    fast.predict_proba(multi[0])
    with threadpool_limits(limits=1, user_api='openmp'):
        max_raw_err = 0.0
        max_prob_err = 0.0
        mismatch = 0
        for x in multi:
            ref_raw = original._raw_predict(x).reshape(-1)
            got_raw = fast.raw(x)
            ref_prob = original.predict_proba(x)
            got_prob = fast.predict_proba(x)
            max_raw_err = max(max_raw_err, float(np.max(np.abs(ref_raw - got_raw))))
            max_prob_err = max(max_prob_err, float(np.max(np.abs(ref_prob - got_prob))))
            mismatch += int(np.argmax(ref_prob[:, 1]) != np.argmax(got_prob[:, 1]))
        print('\n========== NUMERICAL PARITY ==========')
        print(f'Raw logit maxabs:  {max_raw_err:.3e}')
        print(f'Probability maxabs:{max_prob_err:.3e}')
        print(f'Argmax mismatch:   {mismatch} / {len(multi)}')
        if max_raw_err > 1e-9 or max_prob_err > 1e-9 or mismatch:
            raise AssertionError('Numerical or candidate-choice mismatch; do not deploy fast ranker')

        for model, name in ((original, 'sklearn-1thread'), (fast, 'numba-fast')):
            for x in multi[:100]:
                model.predict_proba(x)
            t0 = time.perf_counter()
            for x in multi:
                model.predict_proba(x)
            dt = time.perf_counter() - t0
            print(f'{name:16s}: {1000*dt/len(multi):.4f} ms/call; '
                  f'{1000*dt/len(examples):.4f} ms/Position; {dt:.3f}s total')
    print('PASS: Fast scorer matches frozen model for all tested candidate sets.')
    print('Private sklearn tree layout depends on version; repeat parity when changing sklearn.')


if __name__ == '__main__':
    main()
