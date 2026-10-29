"""Independent pool/outcome counts for the extended diagnostics and their figure.

Uses the independent grader, without importing the extension experiment script.
"""
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from verify_team_core import correct, majority

HERE = Path(__file__).resolve().parent


def calculate():
    source = HERE / 'data/current_inputs.jsonl'
    rows = [json.loads(line) for line in source.read_text('utf-8').splitlines()]
    result = {'input_sha256': sha256(source.read_bytes()).hexdigest(), 'configurations': []}
    for bench in ('mmlu_pro', 'math500', 'gsm8k'):
        evaluation = [row for row in rows if row['benchmark'] == bench and row['split'] == 'evaluation']
        for pair in combinations(('qwen', 'llama', 'mistral'), 2):
            for k in range(1, 10):
                counts = Counter()
                buckets = {}
                vote_shares = []
                for row in evaluation:
                    member = [[x for x in row['answers'][model][:k] if x] for model in pair]
                    answers = [answer for values in member for answer in values]
                    cov = any(correct(x, row['gold'], bench) for x in answers)
                    mem = any(correct(majority(values), row['gold'], bench) for values in member)
                    maj = correct(majority(answers), row['gold'], bench)
                    assert not mem or cov
                    assert not maj or cov
                    outcome = 'selected' if maj else ('missed' if cov else 'absent')
                    counts['coverage'] += int(cov)
                    counts['member'] += int(mem)
                    counts['majority'] += int(maj)
                    counts['candidate_only'] += int(cov and not mem)
                    counts[outcome] += 1
                    distinct = len(set(answers))
                    group = '0' if distinct == 0 else ('1' if distinct == 1 else
                        '2-3' if distinct <= 3 else '4-6' if distinct <= 6 else
                        '7-10' if distinct <= 10 else '>10')
                    bucket = buckets.setdefault(group, Counter())
                    bucket['n'] += 1
                    bucket[outcome] += 1
                    if cov and not maj:
                        gold_votes = sum(correct(x, row['gold'], bench) for x in answers)
                        vote_shares.append(gold_votes / len(answers))
                assert sum(counts[tag] for tag in ('selected', 'missed', 'absent')) == len(evaluation)
                assert counts['coverage'] - counts['member'] == counts['candidate_only']
                result['configurations'].append({'benchmark': bench, 'pair': list(pair),
                    'k': k, 'n': len(evaluation), 'counts': dict(counts),
                    'buckets': {key: dict(value) for key, value in buckets.items()},
                    'loss_gold_vote_shares': vote_shares})
    return result


if __name__ == '__main__':
    reference = calculate()
    destination = HERE / 'audit/extensions_independent_reference.json'
    destination.write_text(json.dumps(reference, indent=2) + '\n', encoding='utf-8')
    print(f'Independent counts verified: {len(reference["configurations"])} configurations.')
