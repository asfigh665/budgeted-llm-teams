"""Recompute the paper's analyses and compare against frozen numerical references."""
from pathlib import Path
from itertools import combinations
import argparse
import json
import math
import subprocess
import sys

import recompute_team as C

HERE = Path(__file__).resolve().parent
CHECKS = 0


def compare(actual, expected, path='result'):
    """Compare every reference value; allow only floating-point roundoff."""
    global CHECKS
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, f'{path}: missing {key}'
            compare(actual[key], value, f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f'{path}: length mismatch'
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f'{path}[{i}]')
    elif isinstance(expected, float):
        CHECKS += 1
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f'{path}: {actual} != {expected}'
    else:
        CHECKS += 1
        assert actual == expected, f'{path}: {actual} != {expected}'


def save(relative, result):
    path = HERE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def read(relative):
    return json.loads((HERE / relative).read_text('utf8'))


def budget18():
    """Replay even budgets 2--36 on completed evaluation-only answer pools."""
    inputs = [json.loads(s) for s in (HERE/'data/budget18_inputs.jsonl').read_text('utf8').splitlines()]
    result = {}
    families = {'B36_minus_B2':[], 'B36_minus_B18':[]}
    for bench in ['mmlu_pro','math500']:
        records = [r for r in inputs if r['benchmark']==bench]
        qids = [r['qid'] for r in records]
        assert len(qids)==len(set(qids))
        data = {r['qid']:r['answers'] for r in records}
        gold = {r['qid']:r['gold'] for r in records}
        assert all(len(v)==18 for r in records for v in r['answers'].values())
        kind = 'mmlu_pro' if bench=='mmlu_pro' else 'generic'
        result[bench] = {'evaluation_n':len(qids),'pairs':{}}
        for pair in combinations(C.MODELS,2):
            curve, preds = [], {}
            for k in range(1,19):
                pred = {q:C.P.majority([a for m in pair for a in data[q][m][:k]]) for q in qids}
                preds[k] = pred
                coverage = sum(any(C.A.gold_ok(a,gold[q],kind) for m in pair for a in data[q][m][:k]) for q in qids)/len(qids)
                member = sum(any(C.A.gold_ok(C.P.majority(data[q][m][:k]),gold[q],kind) for m in pair) for q in qids)/len(qids)
                curve.append({'k':k,'budget':2*k,'n':len(qids),
                    'accuracy':C.accuracy(pred,gold,kind,qids), 'coverage':coverage,
                    'member_selection':member,'h_mem':coverage-member})
            contrasts = {}
            for start,name in [(1,'B36_minus_B2'),(9,'B36_minus_B18')]:
                stat = C.paired(preds[18],preds[start],gold,kind,qids)
                contrasts[name]=stat
                families[name].append(stat)
            result[bench]['pairs']['+'.join(pair)] = {
                'models':list(pair),'curve':curve,'endpoint_contrasts':contrasts}
    for family in families.values():
        assert len(family)==6
        C.holm(family)
    return result


def run(script):
    subprocess.run([sys.executable, str(HERE/script)], cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-figures', action='store_true', help='Run numerical reproduction without plotting.')
    args = parser.parse_args()
    result = C.calculate()
    compare(result,read('expected/core.json'),'core')
    save('data/current_results.json',result)
    (HERE/'data/current_items.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in C.ITEMS),encoding='utf8')
    import recompute_extensions as E
    extended = E.calculate()
    compare(extended,read('expected/extensions.json'),'diagnostics')
    save('data/current_extensions.json',extended)
    additional = budget18()
    compare(additional,read('expected/budget18.json'),'budget18')
    save('data/budget18_results.json',additional)
    run(Path('verify_team_core.py'))
    run(Path('verify_extensions.py'))
    run(Path('render_team.py'))
    if not args.skip_figures:
        for name in ['build_pair_matrix.py','build_fragmentation.py','build_cost_curves.py']:
            run(Path('figures')/name)
    report = {'status':'PASS', 'reference_values_checked':CHECKS,
              'core_evaluation_sizes':{b:r['n_eval'] for b,r in result['benchmarks'].items()},
              'budget18_evaluation_sizes':{b:r['evaluation_n'] for b,r in additional.items()},
              'figures_generated':not args.skip_figures}
    save('audit/reproduction.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
