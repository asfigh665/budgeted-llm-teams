"""Shared call stream and single-model reference policies."""
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
import paper_experiments as P
MODELS=["qwen","llama","mistral"]
def mean(xs):
    return float(np.mean(xs))

def paired(pa, pb, gold, kind, qids, B=4000):
    return A.paired_bootstrap(pa, pb, gold, kind, qids, B=B)

def build_stream(data, q, models, n_max):
    seq, seen = ([], set())
    for j in range(2):
        if j < len(P.model_pool(data, q, models[0])):
            seq.append((models[0], j))
            seen.add((models[0], j))
    i = 0
    while len(seq) < n_max:
        added = False
        for m in models[1:]:
            if i < len(P.model_pool(data, q, m)) and (m, i) not in seen:
                seq.append((m, i))
                seen.add((m, i))
                added = True
        if i < len(P.model_pool(data, q, models[0])) and (models[0], i) not in seen:
            seq.append((models[0], i))
            seen.add((models[0], i))
            added = True
        i += 1
        if not added:
            break
    return seq[:n_max]

def policy_eval(data, qids, rel, gold, kind, *, theta, n_min, n_max, models, K_fixed):
    preds, costs = ({}, {})
    for q in qids:
        seq = build_stream(data, q, models, n_max)
        pools_q = {m: [] for m in models}
        used, pred = (0, None)
        for m, j in seq:
            pools_q[m].append(P.model_pool(data, q, m)[j])
            used += 1
            if used < n_min:
                continue
            g, _ = P._posterior_now(pools_q, models, rel, kind, K_fixed)
            if g is None:
                continue
            key = max(g.items(), key=lambda kv: kv[1])[0]
            if g[key] >= theta:
                pred = next((a for mm in models for a in pools_q[mm] if a not in (None, '') and A.cluster_key(a, kind) == key), None)
                break
        if pred is None:
            g, _ = P._posterior_now(pools_q, models, rel, kind, K_fixed)
            if g is not None:
                key = max(g.items(), key=lambda kv: kv[1])[0]
                pred = next((a for mm in models for a in pools_q[mm] if a not in (None, '') and A.cluster_key(a, kind) == key), None)
        preds[q], costs[q] = (pred, used)
    return (preds, costs)

def single_preds(data, qids, model, k, kind):
    return {q: P.majority(P.model_pool(data, q, model, k)) for q in qids}

def best_single_arm(data, gold, kind, qids, B):
    """Return (model, preds) for the strongest single-member arm at B calls on these questions."""
    cands = {}
    for m in MODELS:
        p = single_preds(data, qids, m, B, kind)
        cands[m] = (P.acc(p, gold, kind, qids), p)
    best = max(cands, key=lambda m: cands[m][0])
    return (best, cands[best][1], {m: cands[m][0] for m in MODELS})
