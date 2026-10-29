"""Independent arithmetic, item-pairing and scoring audit of the verified core.

Does not import the experiment/recompute implementation or call language models.
"""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
import math
import re
import numpy as np
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
R=json.loads((HERE/'data/current_results.json').read_text('utf8'))
INPUT=[json.loads(l) for l in (HERE/'data/current_inputs.jsonl').read_text('utf8').splitlines()]
ITEM=[json.loads(l) for l in (HERE/'data/current_items.jsonl').read_text('utf8').splitlines()]
N=0


def check(ok,message):
    global N
    N+=1
    assert ok,message


def close(x,y,message):
    check(abs(x-y)<1e-12,message+f': {x} != {y}')


def fold(s):
    s=str(s).strip().replace('\\\\','\\')
    s=re.sub(r'\\(?:left|right)\s*','',s).replace('$','')
    for old,new in [('\\dfrac','\\frac'),('\\tfrac','\\frac'),('\\leq','\\le'),
                    ('\\geq','\\ge'),('\u2212','-'),('\u03c0','\\pi')]:
        s=s.replace(old,new)
    s=re.sub(r'\s+','',s).rstrip('.')
    if s.startswith('{') and s.endswith('}') and s.count('{')==1:
        s=s[1:-1]
    return s


def number(s):
    try:
        m=re.fullmatch(r'([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}',s)
        if m:
            return (-1 if m[1]=='-' else 1)*Fraction(m[2])/Fraction(m[3])
        m=re.fullmatch(r'([+-]?)\(?([+-]?\d+)\)?/\(?([+-]?\d+)\)?',s)
        if m:
            return (-1 if m[1]=='-' else 1)*Fraction(m[2])/Fraction(m[3])
        if re.fullmatch(r'[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?',s):
            a,b=s.split('/');return Fraction(a)/Fraction(b)
        if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',s):
            return Fraction(s)
    except (ValueError,ZeroDivisionError):
        pass
    return None


def correct(pred,gold,bench):
    if pred is None or not str(pred).strip() or not str(gold).strip():
        return False
    if bench=='mmlu_pro':
        return str(pred).strip()==str(gold).strip()
    a,b=fold(pred),fold(gold)
    def eq(x,y):
        if not x or not y:return False
        if x==y:return True
        u,v=number(x),number(y)
        return u is not None and v is not None and u==v
    if eq(a,b):return True
    aa=re.fullmatch(r'(?:[A-Za-z]|\\?[A-Za-z]_\d+)=(.+)',a)
    bb=re.fullmatch(r'(?:[A-Za-z]|\\?[A-Za-z]_\d+)=(.+)',b)
    if aa or bb:return eq(aa[1] if aa else a,bb[1] if bb else b)
    return False


def majority(xs):
    cc=Counter(x for x in xs if x not in (None,''))
    return min(cc,key=lambda a:(-cc[a],a)) if cc else None


def compare(stats,pa,pb,rows,bench,label):
    diff=np.array([int(correct(x['predictions'][pa],x['gold'],bench))-
                   int(correct(x['predictions'][pb],x['gold'],bench)) for x in rows])
    close(stats['delta'],float(diff.mean()),label+' paired difference')
    pos,neg=int(sum(diff>0)),int(sum(diff<0))
    pv=binomtest(pos,pos+neg,.5).pvalue if pos+neg else 1.
    close(stats['p'],pv,label+' exact paired p')
    rng=np.random.default_rng(0)
    ci=np.percentile(diff[rng.integers(0,len(diff),size=(4000,len(diff)))].mean(axis=1),[2.5,97.5])
    for j in [0,1]:close(stats['ci'][j],float(ci[j]),label+' bootstrap endpoint')


def main():
    groups=defaultdict(list);inputs=defaultdict(dict)
    for row in INPUT:inputs[row['benchmark']][row['qid']]=row
    for row in ITEM:groups[row['benchmark']].append(row)
    check(len({(r['benchmark'],r['qid']) for r in INPUT})==len(INPUT),'duplicate input IDs')
    check(len({(r['benchmark'],r['qid']) for r in ITEM})==len(ITEM),'duplicate outcome IDs')
    family=defaultdict(list)
    for bench,r in R['benchmarks'].items():
        rows=groups[bench];source=inputs[bench]
        check(len(source)==r['n'],'retained source count')
        check(len(rows)==r['n_eval'],'evaluation count')
        check(set(r['calibration_qids']).isdisjoint(r['evaluation_qids']),'calibration leakage')
        check(set(x['qid'] for x in rows)==set(r['evaluation_qids']),'evaluation ID alignment')
        check(r['n']==r['n_cal']+r['n_eval'],'count partition')
        check(r['calibration_calls']==27*r['n_cal'],'full calibration budget')
        for inp in source.values():
            check(all(len(a)==9 for a in inp['answers'].values()),'nine draws per model')
        for B,a in r['allocation'].items():
            for tag in ['two','three']:
                close(a[tag],np.mean([correct(x['predictions'][f'alloc_B{B}_{tag}'],x['gold'],bench) for x in rows]),'allocation accuracy')
                check(sum(a['splits'][tag].values())==int(B),'allocation budget')
            st=a['three_minus_two'];family['allocation_three_minus_two'].append(st)
            compare(st,f'alloc_B{B}_three',f'alloc_B{B}_two',rows,bench,'allocation')
            if int(B)>9:check(a['single'] is None,'unsupported single-model budget')
            else:
                close(a['single'],np.mean([correct(x['predictions'][f'single_B{B}'],x['gold'],bench) for x in rows]),'single model accuracy')
                st=a['two_minus_single'];family['allocation_two_minus_single'].append(st)
                compare(st,f'alloc_B{B}_two',f'single_B{B}',rows,bench,'single comparison')
        for tag,p in r['fixed_pools'].items():
            for key in ['majority','onecoin','weighted','multicoin']:
                expected=p.get({'weighted':'cal_weighted','multicoin':'multi_coin'}.get(key,key))
                if expected is not None:
                    close(expected,np.mean([correct(x['predictions'][f'pool_{tag}_{key}'],x['gold'],bench) for x in rows]),'fixed-pool accuracy')
            cov=[];mem=[]
            for x in rows:
                inp=source[x['qid']];aa={m:inp['answers'][m][:k] for m,k in p['split'].items()}
                cov.append(any(correct(v,x['gold'],bench) for vals in aa.values() for v in vals))
                mem.append(any(correct(majority(vals),x['gold'],bench) for vals in aa.values()))
                for key in ['majority','onecoin','weighted']:
                    pred=x['predictions'][f'pool_{tag}_{key}']
                    check(pred is None or any(pred in vals for vals in aa.values()),'selector returns observed candidate')
            close(p['coverage_ceiling'],np.mean(cov),'coverage ceiling')
            close(p['member_selection_oracle'],np.mean(mem),'member oracle')
            close(p['gap_coverage'],p['coverage_ceiling']-p['onecoin'],'coverage gap')
            close(p['gap_member'],p['member_selection_oracle']-p['onecoin'],'member gap')
            check(p['member_selection_oracle']<=p['coverage_ceiling'],'nested decision class')
            for field,pa,pb in [('weighted_minus_onecoin','weighted','onecoin'),
                                 ('onecoin_minus_majority','onecoin','majority'),
                                 ('multicoin_minus_onecoin','multicoin','onecoin')]:
                if p[field] is not None:
                    family[field].append(p[field])
                    compare(p[field],f'pool_{tag}_{pa}',f'pool_{tag}_{pb}',rows,bench,'pool comparison')
        stop=r['stopping']
        for rule,key in [('posterior','posterior_grid'),('vote','vote_threshold_grid')]:
            for point in stop[key]:
                name=f"{rule}_t{point['theta']}_n{point['n_min']}"
                close(point['accuracy'],np.mean([correct(x['predictions'][name],x['gold'],bench) for x in rows]),'stopping accuracy')
                close(point['cost'],np.mean([x['costs'][name] for x in rows]),'stopping cost')
                check(all(point['n_min']<=x['costs'][name]<=18 for x in rows),'stop within call bounds')
        for target,point in stop['calibration_selected_targets'].items():
            closest=min(stop['posterior_grid'],key=lambda q:abs(q['calibration_cost']-float(target)))
            check(point==closest,'threshold chosen by calibration cost only')
        for m in ['qwen','llama','mistral']:
            close(stop['single9_reference']['arms'][m],np.mean([correct(x['predictions']['single9_'+m],x['gold'],bench) for x in rows]),'single9 reference')
    for label,objects in family.items():
        n=R['conventions']['holm_families'][label]
        check(len(objects)==n,'Holm family count')
        order=sorted(objects,key=lambda o:o['p']);prev=0.
        for j,obj in enumerate(order):
            prev=max(prev,min(1.,(n-j)*obj['p']))
            close(prev,obj['p_holm'],'Holm adjustment')
    print(f'PASS: {N} independent item, scoring, arithmetic, uncertainty checks.')


if __name__=='__main__':main()
