# Budget allocation, answer aggregation, and stopping in heterogeneous LLM teams

Offline replication code for *Where Should the Budget Go? Allocation,
Aggregation, and Stopping in Heterogeneous LLM Teams*.

**Authors:** Jiabao Wu, Wenhan Jiang, ZiMeng Li, and Lizheng Zu.

**Repository:** https://github.com/asfigh665/budgeted-llm-teams

## Quick start

Use Python 3.10 or later. A GPU, model weights, service credentials, and network
access are not required after installing the dependencies.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py
```

For numerical reproduction without figures:

```bash
python reproduce.py --skip-figures
```

The entry point recomputes results from the bundled answer pools, compares every
reference value, runs independent scoring and arithmetic checks, and writes
`audit/reproduction.json`. A failed comparison raises an exception and returns
a nonzero exit code. Reference results in `expected/` are never overwritten.
Individual render scripts should be run after `reproduce.py`.

## What is reproduced

1. Equal-call two-member and three-member allocation, plus single-model references.
2. Majority, one-coin Bayesian, calibration-weighted, and MMLU-Pro full
   Dawid--Skene readouts on the fixed pools.
3. Candidate coverage and member-majority oracles across three model pairs,
   three tasks, and nine sampling depths, including the 27 displayed settings.
4. Candidate fragmentation and missed-candidate counts.
5. Posterior, vote-threshold, and fixed-length stopping on the same ordered
   stream, with retrospective accuracy targets and calibration-cost selection.
6. The separate 18-response-per-member evaluation subsets and their 18-versus-36
   call endpoint comparisons. These contain 167 MMLU-Pro and 76 MATH-500 items,
   not the full evaluation sets.
7. Numerical LaTeX tables and the three data-derived plots: the pair matrix,
   fragmentation bars, and accuracy--call curves. The renderers reproduce values;
   editorial wording and pagination of the manuscript are not part of this package.

## Data and scope

`data/current_inputs.jsonl` contains extracted answer sequences and reference
answers for 444 MMLU-Pro, 393 MATH-500, and 300 GSM8K questions. Each row includes
`benchmark`, `qid`, `split`, `gold`, and `answers`. Answers are ordered lists of
nine samples for `qwen`, `llama`, and `mistral`; an empty string is an abstention
that consumes a call. The frozen calibration/evaluation sizes are 222/222,
196/197, and 150/150 respectively. Question IDs are benchmark identifiers.

`data/budget18_inputs.jsonl` contains the separately sampled, completed
evaluation-only 18-response sequences. It uses the same answer conventions.
It must not be concatenated with the nine-response pools.

This release starts **after response extraction and duplicate exclusion**.
It reproduces the statistical analysis from frozen extracted answers; it does
not rerun model sampling, reconstruct prompts, or independently reproduce raw
text extraction. Raw response prose, benchmark question text, model checkpoints,
and training data are not included. No new model inference is performed.

The model labels refer to Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and
Mistral-7B-Instruct-v0.3. Original sampling used temperature 0.7. Online cost
counts responses rather than tokens, latency, or calibration acquisition cost.

## Statistical and protocol conventions

- Rates in JSON are fractions; plots and tables use percentages. Accuracy
  differences are absolute differences, not relative improvements.
- Fixed-pool one-coin estimation uses unlabeled evaluation answer pools.
  Sequential inference estimates reliability on calibration pools and freezes it.
- Calibration weighting uses labeled nine-sample member-majority accuracy.
- Candidate-string normalization and numerical answer grading are distinct.
  Empty answers are not correct. Majority ties select the smallest key; fixed-pool
  Bayesian and weighted score ties select the largest; sequential posterior ties
  select the smallest.
- Paired bootstrap intervals use 4,000 resamples and seed 0. Exact paired tests
  use question-level correctness. Holm correction is applied within the declared
  comparison families. Intervals remain pointwise.
- Retrospective stopping-grid endpoints use evaluation accuracy. They are distinct
  from thresholds selected before evaluation using calibration call cost.
- The oracle measures apply to realized answer pools, not all possible allocations.

## Files and outputs

| Path | Purpose |
|---|---|
| `reproduce.py` | Full offline reproduction and reference comparison |
| `recompute_team.py` | Core allocation, aggregation, and stopping analysis |
| `recompute_extensions.py` | Cross-pair diagnostics and fragmentation |
| `bayes_readout.py` | One-coin estimation and posterior calculation |
| `agg_explore.py` | Voting, full confusion-matrix readout, paired statistics |
| `paper_experiments.py`, `final_numbers.py` | Allocation and stream primitives |
| `answer_normalization.py` | Candidate normalization and correctness checks |
| `verify_team_core.py`, `verify_extensions.py` | Independent arithmetic/count checks |
| `expected/` | Frozen numerical references |
| `data/` | Extracted input pools and generated analysis results |
| `tables/` | Generated numerical LaTeX tables |
| `figures/` | Plot scripts and generated PDF, SVG, and PNG plots |
| `audit/` | Reproduction and figure-count reports |

`SHA256SUMS.json` records release-file hashes. Runtime-generated results may
change their serialization across dependency versions; numerical comparisons
allow floating-point roundoff only. The release includes `VALIDATION.json`
with the environment and checks used to validate the distributed files.

## Sources and third-party notices

The answer labels and benchmark identifiers derive from MMLU-Pro, MATH-500,
and GSM8K. Their original sources and terms remain applicable:

- MMLU-Pro: https://github.com/TIGER-AI-Lab/MMLU-Pro
- MATH / MATH-500: https://github.com/hendrycks/math and
  https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- GSM8K: https://github.com/openai/grade-school-math

The one-coin and full confusion-matrix comparisons use the Dawid--Skene model
family (Dawid and Skene, 1979, *Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm*).

Comic Neue and Source Sans 3 fonts are bundled solely for the quantitative
figures. Their original copyright and license notices are preserved under
`figures/fonts/`. This package does not relicense third-party assets or datasets.

## License

The original source code is available under the [MIT License](LICENSE).
Bundled benchmark material, archived model responses, and third-party assets
are excluded from this license grant and remain subject to their original
terms. This repository does not relicense those materials.

Git preserves file bytes (including line endings) so that the SHA-256 release
checksums and provenance hashes remain verifiable after cloning.
