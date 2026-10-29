"""One-coin reliability estimation and Bayesian candidate readout."""
from __future__ import annotations
import json, math, random, re
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fractions import Fraction
from pathlib import Path
from hashlib import sha256
import numpy as np
def norm(x, kind='generic'):
    """Default normalisation: answers are already extracted strings."""
    if x is None:
        return ''
    s = str(x).strip()
    return s

def build_candidates(pools, qids, norm_fn=norm, kind='generic', K_fixed=None):
    """Return (cand, votes): per-question candidate classes and per-model vote lists.

    pools[q][m] is a list of raw answers; votes[q][m] holds normalised non-empty answers.
    """
    cand, votes = ({}, {})
    for q in qids:
        keys: List[str] = []
        per: Dict[str, List[str]] = {}
        for m in pools[q]:
            ks = [norm_fn(a, kind) for a in pools[q][m] if a is not None]
            ks = [k for k in ks if k != '']
            per[m] = ks
            keys.extend(ks)
        c = sorted(set(keys))
        if K_fixed is not None and len(c) < K_fixed:
            pass
        cand[q] = c
        votes[q] = per
    return (cand, votes)

def _log_likelihood(votes_q, cand_q, a, models, K):
    """log p(votes | y) for every candidate y, up to additive constants."""
    out = []
    for y in cand_q:
        lp = 0.0
        for m in models:
            ks = votes_q.get(m, [])
            n = len(ks)
            if not n:
                continue
            n_y = sum((1 for k in ks if k == y))
            am = a[m]
            if n_y:
                lp += n_y * math.log(max(am, 1e-09))
            if n - n_y:
                lp += (n - n_y) * math.log(max((1.0 - am) / max(K - 1, 1), 1e-12))
        out.append(lp)
    return out

def posterior(votes_q, cand_q, a, models, K):
    ll = _log_likelihood(votes_q, cand_q, a, models, K)
    if not ll:
        return {}
    mx = max(ll)
    ex = [math.exp(x - mx) for x in ll]
    s = sum(ex) or 1.0
    return {y: e / s for y, e in zip(cand_q, ex)}

def estimate_reliability(pools, qids, models=None, *, norm_fn=norm, kind='generic', K_fixed=None, init='pooled', prior_beta=1.5, n_iter=200, tol=1e-09):
    """Unsupervised one-coin EM.  Returns dict model -> reliability in (0,1)."""
    cand, votes = build_candidates(pools, qids, norm_fn, kind, K_fixed)
    models = list(models) if models else list(pools[qids[0]].keys())

    def _K(q):
        return K_fixed if K_fixed is not None else max(len(cand[q]), 2)
    a = {}
    if init == 'pooled':
        for m in models:
            hit = tot = 0
            for q in qids:
                ks = votes[q].get(m, [])
                if not ks:
                    continue
                pooled = [k for mm in models for k in votes[q].get(mm, [])]
                pl = pooled_majority(pooled)
                hit += sum((1 for k in ks if k == pl))
                tot += len(ks)
            a[m] = (hit + prior_beta) / (tot + 2 * prior_beta) if tot else 0.5
    elif init == 'self':
        for m in models:
            hit = tot = 0
            for q in qids:
                ks = votes[q].get(m, [])
                if not ks:
                    continue
                pl = pooled_majority(ks)
                hit += sum((1 for k in ks if k == pl))
                tot += len(ks)
            a[m] = (hit + prior_beta) / (tot + 2 * prior_beta) if tot else 0.5
    elif init == 'uniform':
        a = {m: 0.6 for m in models}
    elif isinstance(init, dict):
        a = dict(init)
    else:
        raise ValueError(init)
    a = {m: min(max(v, 0.02), 0.98) for m, v in a.items()}
    for _ in range(n_iter):
        gamma = {}
        for q in qids:
            if not cand[q]:
                gamma[q] = {}
                continue
            gamma[q] = posterior(votes[q], cand[q], a, models, _K(q))
        newa = {}
        for m in models:
            num = prior_beta
            den = 2 * prior_beta
            for q in qids:
                ks = votes[q].get(m, [])
                if not ks or not gamma[q]:
                    continue
                num += sum((gamma[q].get(k, 0.0) for k in ks))
                den += len(ks)
            newa[m] = min(max(num / den, 0.02), 0.98)
        delta = max((abs(newa[m] - a[m]) for m in models))
        a = newa
        if delta < tol:
            break
    return a

def pooled_majority(answers: Sequence[str]):
    if not answers:
        return ''
    cnt = Counter(answers)
    top = cnt.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return sorted([a for a, c in top if c == top[0][1]])[0]
    return top[0][0]

def readout(pools, models=None, *, reliabilities=None, norm_fn=norm, kind='generic', K_fixed=None, qids=None):
    """Return {qid: predicted raw answer}.

    If reliabilities is None they are estimated by EM on the same pools (training-free).
    """
    qids = list(pools) if qids is None else list(qids)
    models = list(models) if models else list(pools[qids[0]].keys())
    cand, votes = build_candidates(pools, qids, norm_fn, kind, K_fixed)
    a = estimate_reliability(pools, qids, models, norm_fn=norm_fn, kind=kind, K_fixed=K_fixed) if reliabilities is None else reliabilities
    preds = {}
    for q in qids:
        if not cand[q]:
            preds[q] = None
            continue
        K = K_fixed if K_fixed is not None else max(len(cand[q]), 2)
        g = posterior(votes[q], cand[q], a, models, K)
        if not g:
            preds[q] = None
            continue
        y = max(g.items(), key=lambda kv: (kv[1], kv[0]))[0]
        rep = None
        for m in models:
            for raw in pools[q][m]:
                if raw is not None and norm_fn(raw, kind) == y:
                    rep = raw
                    break
            if rep is not None:
                break
        preds[q] = rep
    return (preds, a)

def posterior_scores(pools_q, a, models, *, norm_fn=norm, kind='generic', K_fixed=None):
    """Scores for a single question's pool (used by the anytime stopping rule)."""
    cand, votes = build_candidates({0: pools_q}, [0], norm_fn, kind, K_fixed)
    if not cand[0]:
        return ({}, {})
    K = K_fixed if K_fixed is not None else max(len(cand[0]), 2)
    g = posterior(votes[0], cand[0], a, models, K)
    return (g, {m: votes[0].get(m, []) for m in models})
