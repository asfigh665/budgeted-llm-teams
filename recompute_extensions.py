"""Cross-pair oracle diagnostics and candidate fragmentation."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
import argparse
import json
from pathlib import Path
import numpy as np
import recompute_team as C

HERE=Path(__file__).resolve().parent
MODELS=['qwen','llama','mistral']
BENCHES=['mmlu_pro','math500','gsm8k']
BIN_LABELS=['1','2-3','4-6','7-10','>10']
OUTCOMES=['majority_correct','covered_but_missed','no_correct']
ITEMS=[]


def mean(xs):return float(np.mean(xs))


def interval(xs):
    x=np.asarray(xs,float)
    rng=np.random.default_rng(0)
    boot=x[rng.integers(0,len(x),size=(4000,len(x)))].mean(axis=1)
    return [float(v) for v in np.percentile(boot,[2.5,97.5])]


def ratio_audit(coverage,member,majority):
    """Never discard denominator-zero resamples to present an unconditional CI."""
    c,m,a=map(lambda x:np.asarray(x,float),(coverage,member,majority))
    den=mean(m-a);num=mean(c-a)
    rng=np.random.default_rng(0);idx=rng.integers(0,len(a),size=(4000,len(a)))
    db=(m-a)[idx].mean(axis=1);nb=(c-a)[idx].mean(axis=1)
    valid=db>0
    return {'point':num/den if den>0 else None,
        'denominator':den,'bootstrap_nonpositive_denominator':int(sum(~valid)),
        'bootstrap_total':4000,'bootstrap_nonpositive_fraction':float(mean(~valid)),
        'conditional_positive_denominator_ci':
          [float(x) for x in np.percentile(nb[valid]/db[valid],[2.5,97.5])] if any(valid) else None,
        'unconditional_ci':None,
        'interpretation':'Exploratory ratio only; the CI is conditional on a positive denominator. '
           'Nested coverage/member output classes guarantee their ordering and do not '
           'establish that aggregation research outperforms budget allocation.'}


def bucket_label(n):
    return '0' if n==0 else '1' if n==1 else '2-3' if n<=3 else '4-6' if n<=6 else '7-10' if n<=10 else '>10'


def summarize(items):
    n=len(items)
    counts=Counter({k:0 for k in OUTCOMES})
    for row in items:counts[row['outcome']]+=1
    loss=[r for r in items if r['outcome']=='covered_but_missed']
    shares=[r['gold_vote_share'] for r in loss]
    bins=[]
    for label in BIN_LABELS:
        subset=[r for r in items if bucket_label(r['distinct_candidates'])==label]
        cc=Counter({k:0 for k in OUTCOMES})
        for r in subset:cc[r['outcome']]+=1
        bins.append({'label':label,'n':len(subset),**dict(cc),
            'rates':{k:cc[k]/len(subset) for k in OUTCOMES} if subset else None})
    zero=[r for r in items if r['distinct_candidates']==0]
    curve=[]
    for t in np.arange(0,11)/10:
        recovered=int(sum(r['gold_vote_share']>=t for r in loss))
        curve.append({'threshold':float(t),'extra_correct':recovered,
            'accuracy':(counts['majority_correct']+recovered)/n,
            'fraction_of_missed_mass':recovered/len(loss) if loss else None})
    return {'n':n,**dict(counts),'counts':dict(counts),
        'rates':{k:counts[k]/n for k in OUTCOMES},'bins':bins,
        'zero_distinct_n':len(zero),
        'zero_distinct_counts':dict(Counter(r['outcome'] for r in zero)),
        'loss_gold_share':{'n':len(loss),'mean':mean(shares) if shares else None,
            'median':float(np.median(shares)) if shares else None,
            'mean_distinct_candidates':mean([r['distinct_candidates'] for r in loss]) if loss else None,
            'below_0_2':sum(s<.2 for s in shares),'at_least_0_5':sum(s>=.5 for s in shares),
            'strictly_above_0_5':sum(s>.5 for s in shares)},
        'recovery_curve':curve}


def calculate():
    source=HERE/'data/current_inputs.jsonl';corepath=HERE/'data/current_results.json'
    inputs=[json.loads(l) for l in source.read_text('utf8').splitlines()]
    core=json.loads(corepath.read_text('utf8'))
    report={'version':'portable-diagnostics-1',
        'input_sha256':sha256(source.read_bytes()).hexdigest(),
        'core_sha256':sha256(corepath.read_bytes()).hexdigest(),
        'conventions':{'bootstrap_replicates':4000,'bootstrap_seed':0,
            'grader':'Frozen core grader; corrected stored inputs, no raw reparse.',
            'oracle_names':['coverage','member_selection'],
            'baseline':'Uniform majority on identical normalized stored answer strings.',
            'h_mem':'coverage minus member_selection; observed candidate-only mass, '
                    'not an allocation optimality gap.',
            'output_class_partition':['member_selection_correct','candidate_only','no_correct'],
            'distinct_candidates':'Distinct stored answer strings used by uniform majority; '
                                  'not a claim of semantic inequivalence.',
            'fragmentation':'Descriptive association; outcome rates conditional on bucket.',
            'recovery_curve':'Reference-answer-dependent hypothetical mass coverage. '
                'Not an executable policy or impossibility proof for weighted voting.',
            'holm_families':{'onecoin_vs_majority_k9':9,'cal_weighted_vs_majority_k9':9,
                             'cal_weighted_vs_onecoin_k9':9},
            'nested_ordering':'No significance test of coverage>=member_selection; '
                'ordering follows from nested output classes. Magnitudes receive bootstrap CIs.',
            'budget_grid':'k1..9, B=2k (nine even budgets), three pairs by three benchmarks.'},
        'benchmarks':{},'all_pairs27':[],'k9_readouts':[]}
    families={k:[] for k in report['conventions']['holm_families']}
    for bench in BENCHES:
        records={r['qid']:r for r in inputs if r['benchmark']==bench}
        ref=core['benchmarks'][bench];qids=ref['evaluation_qids']
        assert set(qids)=={q for q,r in records.items() if r['split']=='evaluation'}
        gold={q:records[q]['gold'] for q in records}
        data={q:records[q]['answers'] for q in records}
        kind='mmlu_pro' if bench=='mmlu_pro' else 'generic'
        br={'n_eval':len(qids),'pairs':{},'all_pairs':{},'budget_curves':{}}
        for pair in combinations(MODELS,2):
            tag='+'.join(pair);nodes={}
            for k in range(1,10):
                observed=[];pa={};cov=[];mem=[];maj=[]
                for q in qids:
                    pp={m:data[q][m][:k] for m in pair}
                    answers=[a for m in pair for a in pp[m] if a not in (None,'')]
                    pred=C.P.majority(answers)
                    a=bool(C.A.gold_ok(pred,gold[q],kind))
                    c=any(C.A.gold_ok(x,gold[q],kind) for x in answers)
                    m=any(C.A.gold_ok(C.P.majority(pp[x]),gold[q],kind) for x in pair)
                    assert not m or c;assert not a or c
                    correct_votes=sum(C.A.gold_ok(x,gold[q],kind) for x in answers)
                    outcome='majority_correct' if a else 'covered_but_missed' if c else 'no_correct'
                    row={'benchmark':bench,'pair':list(pair),'k':k,'budget':2*k,'qid':q,
                        'majority_prediction':pred,'majority_correct':int(a),'coverage':int(c),
                        'member_selection':int(m),'candidate_only':int(c and not m),
                        'candidate_only_already_selected':int(c and not m and a),
                        'candidate_only_still_missed':int(c and not m and not a),
                        'outcome':outcome,'valid_answers':len(answers),
                        'distinct_candidates':len(set(answers)),'gold_votes':correct_votes,
                        'gold_vote_share':correct_votes/len(answers) if answers else 0.}
                    observed.append(row);ITEMS.append(row);pa[q]=pred
                    cov.append(int(c));mem.append(int(m));maj.append(int(a))
                h=np.asarray(cov)-np.asarray(mem)
                node={'k':k,'budget':2*k,'n':len(qids),'pair':list(pair),
                    'majority':mean(maj),'coverage':mean(cov),'member_selection':mean(mem),
                    'h_mem':mean(h),'h_mem_ci':interval(h),
                    'majority_coverage_gap':mean(np.asarray(cov)-np.asarray(maj)),
                    'member_margin':mean(np.asarray(mem)-np.asarray(maj)),
                    'counts':{'majority':sum(maj),'coverage':sum(cov),'member':sum(mem),
                        'candidate_only':int(sum(h)),
                        'candidate_only_already_selected':sum(r['candidate_only_already_selected'] for r in observed),
                        'candidate_only_still_missed':sum(r['candidate_only_still_missed'] for r in observed)}}
                if k in [2,9]:node['ratio_audit']=ratio_audit(cov,mem,maj)
                if k==9:
                    pool={q:{m:data[q][m][:9] for m in pair} for q in qids}
                    pe,rel=C.P.em_readout(data,qids,{m:9 for m in pair},kind,C.P.K_FIXED[bench])
                    rel_init=C.BR.estimate_reliability(pool,qids,list(pair),norm_fn=C.A.cluster_key,
                        kind=kind,K_fixed=C.P.K_FIXED[bench],n_iter=0)
                    pw=C.weighted_vote(pool,list(pair),ref['cal9'],kind)
                    one=C.accuracy(pe,gold,kind,qids);cw=C.accuracy(pw,gold,kind,qids)
                    p1=C.paired(pe,pa,gold,kind,qids);p2=C.paired(pw,pa,gold,kind,qids)
                    p3=C.paired(pw,pe,gold,kind,qids)
                    families['onecoin_vs_majority_k9'].append(p1)
                    families['cal_weighted_vs_majority_k9'].append(p2)
                    families['cal_weighted_vs_onecoin_k9'].append(p3)
                    node.update(onecoin=one,cal_weighted=cw,reliabilities=rel,
                        em_configuration={'initialization':'pooled-majority agreement',
                            'initial_reliabilities':rel_init,'prior_beta':1.5,
                            'maximum_iterations':200,'tolerance':1e-9,'clip':[.02,.98],
                            'K_fixed':C.P.K_FIXED[bench],
                            'candidate_normalization':'Frozen core cluster_key',
                            'fitted_on':'unlabeled evaluation pools (transductive)'},
                        onecoin_vs_majority=p1,cal_weighted_vs_majority=p2,cal_weighted_vs_onecoin=p3)
                    for row in observed:
                        q=row['qid'];row.update(onecoin_prediction=pe[q],cal_weighted_prediction=pw[q])
                    node['readout_analysis']=summarize(observed)
                    report['k9_readouts'].append({'benchmark':bench,'pair':list(pair),**node})
                nodes['k'+str(k)]=node
                if k in [3,6,9]:report['all_pairs27'].append({'benchmark':bench,**node})
            br['pairs'][tag]=nodes
            br['all_pairs'][tag]={k:v for k,v in nodes.items() if k in ['k3','k6','k9']}
            br['budget_curves'][tag]=list(nodes.values())
        br['fragmentation_qwen_llama_2x9']=br['pairs']['qwen+llama']['k9']['readout_analysis']
        refpool=ref['fixed_pools']['2x9'];ql=br['pairs']['qwen+llama']['k9']
        assert abs(ql['majority']-refpool['majority'])<1e-12
        assert abs(ql['coverage']-refpool['coverage_ceiling'])<1e-12
        assert abs(ql['member_selection']-refpool['member_selection_oracle'])<1e-12
        report['benchmarks'][bench]=br
        print(bench,'complete:27budget configurations; fragmentation',br['fragmentation_qwen_llama_2x9']['counts'],flush=True)
    for key,objs in families.items():
        assert len(objs)==9
        C.holm(objs)
    return report

