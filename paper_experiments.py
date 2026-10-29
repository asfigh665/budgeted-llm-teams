"""Fixed-pool allocation and posterior inference primitives."""
from __future__ import annotations
import json, math, random, re
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fractions import Fraction
from pathlib import Path
from hashlib import sha256
import numpy as np
import agg_explore as A
import bayes_readout as BR
MODELS=["qwen","llama","mistral"]
K_FIXED={"mmlu_pro":10,"math500":None,"gsm8k":None}
SEED_SPLIT=20261020
N_BOOT=4000
def clean(xs):
    return [x for x in xs if x not in (None, '')]

def majority(answers):
    return A.majority(clean(answers))

def qwen_pool(data, qid, n=None):
    xs = list(data[qid]['qwen'])
    return xs if n is None else xs[:n]

def model_pool(data, qid, m, n=None):
    xs = qwen_pool(data, qid) if m == 'qwen' else list(data[qid][m])
    return xs if n is None else xs[:n]

def acc(preds, gold, kind, qids):
    return float(np.mean([A.gold_ok(preds[q], gold[q], kind) for q in qids]))

def boot(pa, pb, gold, kind, qids, B=N_BOOT, seed=0):
    return A.paired_bootstrap(pa, pb, gold, kind, qids, B=B, seed=seed)

def questions_with(data, need):
    out = []
    for q in sorted(data):
        ok = True
        for m, k in need.items():
            if m == 'qwen':
                have = len(qwen_pool(data, q))
            else:
                have = len(data[q].get(m, []))
            if have < k:
                ok = False
                break
        if ok:
            out.append(q)
    return out

def split_halves(qids, seed=SEED_SPLIT):
    rng = random.Random(seed)
    order = list(qids)
    rng.shuffle(order)
    h = len(order) // 2
    return (sorted(order[:h]), sorted(order[h:]))

def balanced_split(B, models):
    k = B // len(models)
    rem = B - k * len(models)
    return {m: k + (1 if i < rem else 0) for i, m in enumerate(models)}

def team_pool(data, qids, split):
    return {q: {m: model_pool(data, q, m, k) for m, k in split.items()} for q in qids}

def em_readout(data, qids, split, kind, K_fixed, pool=None, rel=None):
    pool = pool or team_pool(data, qids, split)
    models = list(split)
    if rel is None:
        rel = BR.estimate_reliability(pool, qids, models, norm_fn=A.cluster_key, kind=kind, K_fixed=K_fixed)
    cand, votes = BR.build_candidates(pool, qids, A.cluster_key, kind, K_fixed)
    preds = {}
    for q in qids:
        if not cand[q]:
            preds[q] = None
            continue
        K = K_fixed if K_fixed is not None else max(len(cand[q]), 2)
        g = BR.posterior(votes[q], cand[q], rel, models, K)
        if not g:
            preds[q] = None
            continue
        y = max(g.items(), key=lambda kv: (kv[1], kv[0]))[0]
        preds[q] = None
        for m in models:
            for a in pool[q][m]:
                if a not in (None, '') and A.cluster_key(a, kind) == y:
                    preds[q] = a
                    break
            if preds[q] is not None:
                break
    return (preds, rel)

def _posterior_now(pools_q, models, rel, kind, K_fixed):
    cand, votes = BR.build_candidates({0: pools_q}, [0], A.cluster_key, kind, K_fixed)
    cands = cand[0]
    if not cands:
        return (None, None)
    K = K_fixed if K_fixed is not None else max(len(cands), 2)
    g = BR.posterior(votes[0], cands, rel, models, K)
    if not g:
        return (None, None)
    return (g, cands)

def anytime_run(data, qids, rel, gold, kind, *, theta, n_min, n_max, models, p_first, K_fixed, order='round_robin'):
    preds, costs = ({}, {})
    for q in qids:
        seq = []
        if order == 'round_robin':
            for j in range(p_first):
                if j < len(model_pool(data, q, models[0])):
                    seq.append((models[0], j))
            i = 0
            while len(seq) < n_max:
                added = False
                for m in models[1:]:
                    if i < len(model_pool(data, q, m)):
                        seq.append((m, i))
                        added = True
                if i < len(model_pool(data, q, models[0])):
                    seq.append((models[0], i))
                    added = True
                i += 1
                if not added:
                    break
        else:
            for m in models:
                for j in range(len(model_pool(data, q, m))):
                    seq.append((m, j))
            seq = seq[:n_max]
        pools_q = {m: [] for m in models}
        used = 0
        pred = None
        for m, j in seq[:n_max]:
            pools_q[m].append(model_pool(data, q, m)[j])
            used += 1
            if used < n_min:
                continue
            g, cands = _posterior_now(pools_q, models, rel, kind, K_fixed)
            if g is None:
                continue
            best_key = max(g.items(), key=lambda kv: kv[1])[0]
            if g[best_key] >= theta:
                pred = next((a for mm in models for a in pools_q[mm] if a not in (None, '') and A.cluster_key(a, kind) == best_key), None)
                break
        if pred is None:
            g, cands = _posterior_now(pools_q, models, rel, kind, K_fixed)
            if g is not None:
                key = max(g.items(), key=lambda kv: kv[1])[0]
                pred = next((a for mm in models for a in pools_q[mm] if a not in (None, '') and A.cluster_key(a, kind) == key), None)
            else:
                pred = majority([a for m in models for a in pools_q[m]])
        preds[q] = pred
        costs[q] = used
    return (preds, costs)
