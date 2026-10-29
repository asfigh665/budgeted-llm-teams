"""Replay all core experiments from frozen, extracted answer pools."""
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
import final_numbers as FN
HERE=Path(__file__).resolve().parent
MODELS=["qwen","llama","mistral"]
THETAS=[.5,.6,.7,.75,.8,.85,.9,.95,.99]
MINIMA=[2,3,4,5]
ITEMS=[]
def rate(values):
    return float(np.mean(values))

def outcomes(pred, gold, kind, qids):
    return [int(A.gold_ok(pred[q], gold[q], kind)) for q in qids]

def accuracy(pred, gold, kind, qids):
    return rate(outcomes(pred, gold, kind, qids))

def paired(pa, pb, gold, kind, qids):
    return A.paired_bootstrap(pa, pb, gold, kind, qids, B=4000, seed=0)

def holm(objects):
    order = sorted(range(len(objects)), key=lambda i: objects[i]['p'])
    prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(order) - rank) * objects[i]['p']))
        objects[i]['p_holm'] = prev
        objects[i]['holm_family_size'] = len(objects)

def weighted_vote(pool, models, weights, kind):
    pred = {}
    for q, rec in pool.items():
        scores = defaultdict(float)
        for m in models:
            for a in rec[m]:
                if a not in (None, ''):
                    key = A.cluster_key(a, kind)
                    if key:
                        scores[key] += weights[m]
        key = max(scores, key=lambda c: (scores[c], c)) if scores else None
        pred[q] = next((a for m in models for a in rec[m] if a not in (None, '') and A.cluster_key(a, kind) == key), None)
    return pred

def prefix_cache(data, qids, models, rel, kind, K):
    cache = {}
    for q in qids:
        stream = FN.build_stream(data, q, models, 18)
        assert len(stream) == 18 and len(set(stream)) == 18
        pool = {m: [] for m in models}
        pref = []
        for m, j in stream:
            pool[m].append(P.model_pool(data, q, m)[j])
            posterior, _ = P._posterior_now(pool, models, rel, kind, K)
            if posterior:
                key = max(posterior, key=posterior.get)
                pred = next((a for mm in models for a in pool[mm] if a not in (None, '') and A.cluster_key(a, kind) == key), None)
                confidence = posterior[key]
            else:
                pred, confidence = (None, 0.0)
            all_answers = [a for mm in models for a in pool[mm]]
            maj = P.majority(all_answers)
            valid = P.clean(all_answers)
            share = max(Counter(valid).values()) / len(valid) if valid else 0.0
            pref.append({'posterior_pred': pred, 'confidence': float(confidence), 'majority_pred': maj, 'majority_share': float(share)})
        cache[q] = pref
    return cache

def stop_from_cache(cache, theta, n_min, rule):
    preds, costs = ({}, {})
    for q, pref in cache.items():
        pos = len(pref) - 1
        score = 'confidence' if rule == 'posterior' else 'majority_share'
        for i, row in enumerate(pref):
            if i + 1 >= n_min and row[score] >= theta:
                pos = i
                break
        key = 'posterior_pred' if rule == 'posterior' else 'majority_pred'
        preds[q], costs[q] = (pref[pos][key], pos + 1)
    return (preds, costs)

def pareto(points):
    kept = []
    best = -1.0
    for p in sorted(points, key=lambda r: (r['cost'], -r['accuracy'])):
        if p['accuracy'] > best + 1e-12:
            kept.append(p)
            best = p['accuracy']
    return kept

def minimum_meeting(points, target):
    ok = [p for p in points if p['accuracy'] >= target - 1e-12]
    return min(ok, key=lambda p: (p['cost'], -p['accuracy'])) if ok else None

def record_items(bench, qids, gold, methods, costs=None):
    for q in qids:
        ITEMS.append({'benchmark': bench, 'qid': q, 'gold': gold[q], 'predictions': {k: p[q] for k, p in methods.items()}, 'costs': {k: c[q] for k, c in (costs or {}).items()}})
def calculate():
    ITEMS.clear()
    inputs = [json.loads(line) for line in (HERE/'data/current_inputs.jsonl').read_text('utf8').splitlines()]
    report = {'version':'portable-replay-1', 'conventions': {'holm_families':
        {'allocation_three_minus_two':12,'allocation_two_minus_single':6,
         'weighted_minus_onecoin':12,'onecoin_minus_majority':12,'multicoin_minus_onecoin':4}},
         'benchmarks':{}, 'correlation_configurations':[]}
    families={k:[] for k in report['conventions']['holm_families']}
    for bench in ['mmlu_pro','math500','gsm8k']:
        rows=[r for r in inputs if r['benchmark']==bench]
        data={r['qid']:r['answers'] for r in rows}
        gold={r['qid']:r['gold'] for r in rows}
        kind='mmlu_pro' if bench=='mmlu_pro' else 'generic'
        cal=[r['qid'] for r in rows if r['split']=='calibration']
        ev=[r['qid'] for r in rows if r['split']=='evaluation']
        cal9 = {m: accuracy({q: P.majority(data[q][m][:9]) for q in cal},gold,kind,cal)
                for m in MODELS}
        order = sorted(MODELS, key=lambda m: -cal9[m]); team = order[:2]
        K = P.K_FIXED[bench]
        rel = BR.estimate_reliability(P.team_pool(data,cal,{m:9 for m in team}), cal,team,
                                      norm_fn=A.cluster_key,kind=kind,K_fixed=K)
        r = {'n':len(cal)+len(ev),'n_cal':len(cal),'n_eval':len(ev),
            'calibration_qids':cal,'evaluation_qids':ev,
            'order':order,'cal9':cal9,'rel_cal':rel,
            'calibration_calls':27*len(cal), 'allocation':{},'fixed_pools':{},'stopping':{}}
        single9 = {m:{q:P.majority(data[q][m][:9]) for q in ev} for m in MODELS}
        single9_acc = {m:accuracy(p,gold,kind,ev) for m,p in single9.items()}
        best9 = max(MODELS,key=lambda m:single9_acc[m])
        methods = {'single9_'+m:p for m,p in single9.items()}
        method_costs = {}
        for B in [6,9,12,18]:
            ps = {}
            splits = {}
            for tag,mods in [('two',team),('three',order)]:
                splits[tag]=P.balanced_split(B,mods)
                pool=P.team_pool(data,ev,splits[tag])
                assert all(sum(len(v) for v in rec.values())==B for rec in pool.values())
                ps[tag]={q:P.majority([a for m in mods for a in pool[q][m]]) for q in ev}
            stats=paired(ps['three'],ps['two'],gold,kind,ev)
            families['allocation_three_minus_two'].append(stats)
            row={'budget':B,'splits':splits,'two':accuracy(ps['two'],gold,kind,ev),
                 'three':accuracy(ps['three'],gold,kind,ev),'three_minus_two':stats,
                 'single':None,'two_minus_single':None}
            if B<=9:
                bm,pb,arms=FN.best_single_arm(data,gold,kind,ev,B)
                st=paired(ps['two'],pb,gold,kind,ev)
                families['allocation_two_minus_single'].append(st)
                row.update(single=accuracy(pb,gold,kind,ev),single_model=bm,
                           single_arms=arms,two_minus_single=st)
                methods[f'single_B{B}']=pb
            r['allocation'][str(B)]=row
            methods.update({f'alloc_B{B}_{k}':v for k,v in ps.items()})
        for tag,spl in [('2x4',{m:4 for m in team}),('3x3',{m:3 for m in order}),
                        ('2x9',{m:9 for m in team}),('3x9',{m:9 for m in order})]:
            pool=P.team_pool(data,ev,spl);ms=list(spl)
            pm={q:P.majority([a for m in ms for a in pool[q][m]]) for q in ev}
            pe,ee=P.em_readout(data,ev,spl,kind,K)
            pw=weighted_vote(pool,ms,cal9,kind)
            sample=[any(A.gold_ok(a,gold[q],kind) for m in ms for a in pool[q][m]) for q in ev]
            member=[any(A.gold_ok(P.majority(pool[q][m]),gold[q],kind) for m in ms) for q in ev]
            ae=accuracy(pe,gold,kind,ev)
            cwst,emst=paired(pw,pe,gold,kind,ev),paired(pe,pm,gold,kind,ev)
            families['weighted_minus_onecoin'].append(cwst)
            families['onecoin_minus_majority'].append(emst)
            row={'budget':sum(spl.values()),'split':spl,'majority':accuracy(pm,gold,kind,ev),
                'onecoin':ae,'cal_weighted':accuracy(pw,gold,kind,ev),'reliabilities':ee,
                'coverage_ceiling':rate(sample),'member_selection_oracle':rate(member),
                'gap_coverage':rate(sample)-ae,'gap_member':rate(member)-ae,
                'weighted_minus_onecoin':cwst,'onecoin_minus_majority':emst,
                'multi_coin':None,'multicoin_minus_onecoin':None}
            if bench=='mmlu_pro':
                padded={q:{m:pool[q].get(m,[]) for m in MODELS} for q in ev}
                pd,conf=A.ds_full(padded,kind,n_iter=100,alpha=.01)
                st=paired(pd,pe,gold,kind,ev)
                row.update(multi_coin=accuracy(pd,gold,kind,ev),multicoin_minus_onecoin=st)
                families['multicoin_minus_onecoin'].append(st)
                methods[f'pool_{tag}_multicoin']=pd
            r['fixed_pools'][tag]=row
            methods.update({f'pool_{tag}_'+k:v for k,v in [('majority',pm),('onecoin',pe),('weighted',pw)]})
        cc=prefix_cache(data,cal,team,rel,kind,K); ce=prefix_cache(data,ev,team,rel,kind,K)
        stopping={'single9_reference':{'model':best9,'accuracy':single9_acc[best9],
                                      'arms':single9_acc,'selection':'post-hoc strongest eval arm'},
                  'team18_reference':r['allocation']['18']['two'],
                  'posterior_grid':[],'vote_threshold_grid':[],'fixed_stream_curve':[],
                  'calibration_selected_targets':{}}
        for rule, key in [('posterior','posterior_grid'),('vote','vote_threshold_grid')]:
            for t in THETAS:
                for nm in MINIMA:
                    pc,co=stop_from_cache(cc,t,nm,rule)
                    pe,cost=stop_from_cache(ce,t,nm,rule)
                    point={'theta':t,'n_min':nm,'cost':rate(list(cost.values())),
                           'accuracy':accuracy(pe,gold,kind,ev),
                           'calibration_cost':rate(list(co.values())),
                           'calibration_accuracy':accuracy(pc,gold,kind,cal)}
                    stopping[key].append(point)
                    methods[f'{rule}_t{t}_n{nm}']=pe
                    method_costs[f'{rule}_t{t}_n{nm}']=cost
        for B in range(1,19):
            pe={q:ce[q][B-1]['majority_pred'] for q in ev}
            stopping['fixed_stream_curve'].append({'cost':float(B),'accuracy':accuracy(pe,gold,kind,ev)})
        for target in [3,5,8]:
            chosen=min(stopping['posterior_grid'],key=lambda p:abs(p['calibration_cost']-target))
            pe,cost=stop_from_cache(ce,chosen['theta'],chosen['n_min'],'posterior')
            # Independent call to archived reference confirms prefix-cache equivalence.
            pp,cs=FN.policy_eval(data,ev,rel,gold,kind,theta=chosen['theta'],
                     n_min=chosen['n_min'],n_max=18,models=team,K_fixed=K)
            assert pp==pe and cs==cost, 'prefix-cache/reference mismatch'
            stopping['calibration_selected_targets'][str(target)]=dict(chosen)
            methods[f'operational_target{target}']=pe
            method_costs[f'operational_target{target}']=cost
        for tag,key in [('posterior','posterior_grid'),('vote_threshold','vote_threshold_grid'),
                        ('fixed_stream','fixed_stream_curve')]:
            points=stopping[key]
            stopping[tag+'_pareto']=pareto(points)
            stopping[tag+'_minimum_to_single9']=minimum_meeting(points,single9_acc[best9])
            stopping[tag+'_minimum_to_team18']=minimum_meeting(points,stopping['team18_reference'])
        r['stopping']=stopping
        record_items(bench,ev,gold,methods,method_costs)
        # Correlation is based on each member's nine-draw majority correctness on cal.
        corrvec={m:np.array([A.gold_ok(P.majority(data[q][m][:9]),gold[q],kind) for q in cal],float)
                 for m in MODELS}
        for mods in [team,[m for m in order if m!=team[1]],[m for m in order if m!=team[0]],order]:
            spl=P.balanced_split(9,mods);pool=P.team_pool(data,ev,spl)
            pred={q:P.majority([a for m in mods for a in pool[q][m]]) for q in ev}
            corr=rate([np.corrcoef(corrvec[m],corrvec[n])[0,1] for m,n in combinations(mods,2)])
            report['correlation_configurations'].append({'benchmark':bench,'team':mods,
                'correlation':corr,'minimum_calibration_accuracy':min(cal9[m] for m in mods),
                'mean_calibration_accuracy':rate([cal9[m] for m in mods]),
                'accuracy':accuracy(pred,gold,kind,ev),'delta_vs_single9':accuracy(pred,gold,kind,ev)-single9_acc[best9],
                'paired_vs_single9':paired(pred,single9[best9],gold,kind,ev)})
        report['benchmarks'][bench]=r
        print(bench,'n/cal/eval',r['n'],r['n_cal'],r['n_eval'],'order',order,
              'operational',stopping['calibration_selected_targets'],flush=True)
    for name,objs in families.items():
        assert len(objs)==report['conventions']['holm_families'][name]
        holm(objs)
    def ranks(vals):
        return np.array([sum(v<u for u in vals)+.5*sum(v==u for u in vals) for v in vals])
    rows=report['correlation_configurations'];dv=[r['delta_vs_single9'] for r in rows]
    report['correlation_spearman']={key:float(np.corrcoef(ranks([r[key] for r in rows]),ranks(dv))[0,1])
          for key in ['correlation','minimum_calibration_accuracy','mean_calibration_accuracy']}
    return report
