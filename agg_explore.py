"""Voting, confusion-matrix aggregation, and paired statistics."""
from __future__ import annotations
import json, math, random, re
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fractions import Fraction
from pathlib import Path
from hashlib import sha256
import numpy as np
from answer_normalization import answers_equal, normalize_answer
MODELS = ["qwen", "llama", "mistral"]
def gold_ok(pred, gold, kind):
    if pred is None:
        return False
    return answers_equal(pred, gold, kind)

def majority(answers):
    """Majority over (possibly clustered) answer strings; deterministic tie-break."""
    if not answers:
        return None
    cnt = Counter(answers)
    top = cnt.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        tied = sorted([a for a, c in top if c == top[0][1]])
        return tied[0]
    return top[0][0]

def cluster_key(ans, kind):
    """Normalised key used to merge semantically identical answers before voting."""
    if ans is None:
        return ''
    return normalize_answer(ans, kind)

def cluster_vote(answers, kind):
    """Vote over equivalence classes; return a representative raw answer."""
    if not answers:
        return None
    groups = defaultdict(list)
    for a in answers:
        if a is None:
            continue
        groups[cluster_key(a, kind)].append(a)
    best = None
    best_n = -1
    for key in sorted(groups):
        n = len(groups[key])
        if n > best_n:
            best, best_n = (groups[key][0], n)
    return best

def vote_share(answers):
    n = len(answers)
    if n == 0:
        return {}
    cnt = Counter(answers)
    return {a: c / n for a, c in cnt.items()}

def readout_majority(pool, kind):
    return majority([a for m in MODELS for a in pool[m]])

def readout_cluster(pool, kind):
    return cluster_vote([a for m in MODELS for a in pool[m]], kind)

def readout_weighted(pool, weights):
    score = defaultdict(float)
    for m in MODELS:
        w = weights.get(m, 0.0)
        for a in pool[m]:
            score[a] += w
    if not score:
        return None
    return max(score.items(), key=lambda kv: (kv[1], kv[0]))[0]

def one_coin_em(pool, kind, n_iter=100, alpha=0.01, em=1e-08):
    """One-coin DS-style EM on free-form answers with per-question candidate sets.

    P(vote = true | true) = a_i ; P(vote = other candidate | true) = (1-a_i)/(K_q-1)
    Estimates a_i (reliability) unsupervised; returns posterior argmax per question.
    """
    qids = sorted(pool)
    cand, votes = ({}, {})
    for q in qids:
        keys = []
        per = {}
        for m in MODELS:
            ks = [cluster_key(a, kind) for a in pool[q][m] if a is not None]
            ks = [k for k in ks if k != '']
            per[m] = ks
            keys += ks
        cand[q] = sorted(set(keys))
        votes[q] = per
    a = {m: 0.6 for m in MODELS}
    for it in range(n_iter):
        gamma = {}
        for q in qids:
            cs = cand[q]
            if not cs:
                gamma[q] = {}
                continue
            K = len(cs)
            logp = []
            for y in cs:
                lp = 0.0
                for m in MODELS:
                    ks = votes[q][m]
                    n_y = sum((1 for k in ks if k == y))
                    n_tot = len(ks)
                    if n_tot:
                        if n_y:
                            lp += n_y * math.log(max(a[m], 1e-06))
                        if n_tot - n_y:
                            p_off = (1.0 - a[m]) / max(K - 1, 1)
                            lp += (n_tot - n_y) * math.log(max(p_off, 1e-12))
                logp.append(lp)
            mxl = max(logp)
            ex = [math.exp(x - mxl) for x in logp]
            s = sum(ex) or 1.0
            gamma[q] = {y: e / s for y, e in zip(cs, ex)}
        newa = {}
        for m in MODELS:
            num = alpha
            den = 2 * alpha
            for q in qids:
                n_tot = len(votes[q][m])
                if not n_tot:
                    continue
                g = gamma[q]
                if not g:
                    continue
                hit = sum((g.get(k, 0.0) for k in votes[q][m]))
                num += hit
                den += n_tot
            newa[m] = min(max(num / den, 0.001), 1 - 0.001)
        delta = sum((abs(newa[m] - a[m]) for m in MODELS))
        a = newa
        if delta < em:
            break
    preds = {}
    for q in qids:
        g = gamma[q]
        if not g:
            preds[q] = None
            continue
        y = max(g.items(), key=lambda kv: (kv[1], kv[0]))[0]
        rep = None
        for m in MODELS:
            for raw in pool[q][m]:
                if raw is not None and cluster_key(raw, kind) == y:
                    rep = raw
                    break
            if rep is not None:
                break
        preds[q] = rep
    return (preds, a)

def ds_full(pool, kind, n_iter=100, alpha=0.01):
    """Full Dawid-Skene over the 10 MMLU options (multi-vote per model)."""
    qids = sorted(pool)
    letters = 'ABCDEFGHIJ'
    L = len(letters)
    votes = {}
    for q in qids:
        per = {}
        for m in MODELS:
            ks = []
            for a in pool[q][m]:
                k = cluster_key(a, kind)
                if len(k) == 1 and k in letters:
                    ks.append(letters.index(k))
            per[m] = ks
        votes[q] = per
    pseudo = {}
    for q in qids:
        allv = [letters[i] for m in MODELS for i in votes[q][m]]
        pseudo[q] = majority(allv)
    pi = {}
    for m in MODELS:
        mat = np.full((L, L), alpha / L)
        for q in qids:
            y = pseudo.get(q)
            if y not in letters:
                continue
            yi = letters.index(y)
            for a in votes[q][m]:
                mat[yi, a] += 1
        pi[m] = mat / mat.sum(axis=1, keepdims=True)
    preds = {}
    for it in range(n_iter):
        gamma = {}
        for q in qids:
            lp = np.zeros(L)
            for m in MODELS:
                for a in votes[q][m]:
                    lp += np.log(np.maximum(pi[m][:, a], 1e-12))
            mx = lp.max()
            e = np.exp(lp - mx)
            gamma[q] = e / e.sum()
        preds = {q: letters[int(np.argmax(gamma[q]))] for q in qids}
        for m in MODELS:
            mat = np.full((L, L), alpha / L)
            for q in qids:
                g = gamma[q]
                for a in votes[q][m]:
                    mat[:, a] += g
            pi[m] = mat / mat.sum(axis=1, keepdims=True)
    return (preds, {m: pi[m].diagonal().round(4).tolist() for m in MODELS})

def accuracy(preds, gold, kind, qids):
    return float(np.mean([gold_ok(preds[q], gold[q], kind) for q in qids]))

def paired_bootstrap(pred_a, pred_b, gold, kind, qids, B=4000, seed=0):
    """Paired comparison with a percentile bootstrap CI and an exact two-sided sign test.

    For paired 0/1 outcomes the sign-flip permutation test is exactly the two-sided sign
    (McNemar) test on the discordant pairs, so we compute it in closed form instead of
    Monte-Carlo: p = min(1, 2 * BinomCDF(min(n10, n01); n10+n01, 1/2)).
    """
    a = np.array([gold_ok(pred_a[q], gold[q], kind) for q in qids], dtype=float)
    b = np.array([gold_ok(pred_b[q], gold[q], kind) for q in qids], dtype=float)
    d = a - b
    n = len(d)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if np.all(np.isin(d, (-1.0, 0.0, 1.0))):
        n10 = int((d == 1).sum())
        n01 = int((d == -1).sum())
        m = n10 + n01
        if m == 0:
            p = 1.0
        else:
            k = min(n10, n01)
            tail = sum((math.comb(m, i) for i in range(0, k + 1))) / 2.0 ** m
            p = float(min(1.0, 2.0 * tail))
    else:
        perm = rng.choice([-1.0, 1.0], size=(B, n))
        perm_stats = (d * perm).mean(axis=1)
        p = float((np.sum(np.abs(perm_stats) >= abs(obs) - 1e-12) + 1) / (B + 1))
    return dict(delta=float(obs), ci=[float(lo), float(hi)], p=float(p))
