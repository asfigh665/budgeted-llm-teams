"""Plot descriptive within-bin outcomes, checking source counts independently.

Counts come from current_extensions.json. Before plotting, the script recomputes
the partition using current_inputs.jsonl and the independent core-audit grader,
then checks its totals against current_results.json. No monotonic or causal
relationship is assumed. Zero-size bins are omitted from the plotted panels.
"""
from pathlib import Path
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comic_plot_style import HEADING
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

OUT = Path(__file__).resolve().parent
PAPER = OUT.parent
AUDIT = PAPER / "audit"
AUDIT.mkdir(exist_ok=True)
SOURCE = PAPER / "data/current_extensions.json"
CORE = PAPER / "data/current_results.json"
INPUTS = PAPER / "data/current_inputs.jsonl"
source_raw = SOURCE.read_bytes()
core_raw = CORE.read_bytes()
inputs_raw = INPUTS.read_bytes()
source_hash = sha256(source_raw).hexdigest()
report = json.loads(source_raw)
core = json.loads(core_raw)

BENCHMARKS = [("mmlu_pro", "MMLU-Pro"), ("math500", "MATH-500"), ("gsm8k", "GSM8K")]
CATEGORIES = ["majority_correct", "covered_but_missed", "no_correct"]
LABEL_ORDER = ["1", "2-3", "4-6", "7-10", ">10"]
COLORS = ["#71A48D", "#D8A067", "#BDC7CE"]
NAMES = ["Majority correct", "Covered but missed", "No correct candidate"]


def canonical_label(label):
    return str(label).replace("\u2013", "-").replace("\u2212", "-").replace(" ", "")


def bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7-10" if n <= 10 else ">10"


# This helper is independent of the experiment reconstruction and extension code.
independent = runpy.run_path(str(PAPER / "verify_team_core.py"))
correct = independent["correct"]
majority = independent["majority"]
recounted = defaultdict(lambda: defaultdict(Counter))
for line in inputs_raw.decode("utf-8").splitlines():
    row = json.loads(line)
    if row["split"] != "evaluation":
        continue
    bench = row["benchmark"]
    answers = row["answers"]["qwen"][:9] + row["answers"]["llama"][:9]
    valid = [a for a in answers if a not in (None, "")]
    group = bucket(len(set(valid)))
    covered = any(correct(a, row["gold"], bench) for a in valid)
    selected = correct(majority(answers), row["gold"], bench)
    assert not selected or covered, "Correct majority must be present in its pool"
    category = "majority_correct" if selected else "covered_but_missed" if covered else "no_correct"
    recounted[bench][group].update({category: 1, "n": 1})

checked_rows = {}
numerical_source = {
    "purpose": "Descriptive answer-pool outcomes by distinct normalized-answer count",
    "pool": {"qwen": 9, "llama": 9},
    "within_bin_normalization": "100 * category_count / n",
    "zero_size_bins": "omitted from panels",
    "source_files": {
        "data/current_extensions.json": source_hash,
        "data/current_results.json": sha256(core_raw).hexdigest(),
        "data/current_inputs.jsonl": sha256(inputs_raw).hexdigest(),
        "verify_team_core.py": sha256((PAPER / "verify_team_core.py").read_bytes()).hexdigest(),
    },
    "checks": [],
    "benchmarks": {},
}

for bench, title in BENCHMARKS:
    block = report["benchmarks"][bench]["fragmentation_qwen_llama_2x9"]
    assert not block.get("zero_distinct_n", 0), "Nonempty zero-answer group requires a visible extra bin"
    assert not recounted[bench].get("0", {}).get("n", 0), "Recount found questions with zero candidate keys"
    by_label = {canonical_label(row["label"]): row for row in block["bins"]}
    assert len(by_label) == len(block["bins"]), "Duplicate bin labels"
    assert set(by_label).issubset(set(LABEL_ORDER)), "Unrecognized positive-size bin"
    rows = []
    totals = Counter()
    for label in LABEL_ORDER:
        row = by_label.get(label, {"label": label, "n": 0, **dict.fromkeys(CATEGORIES, 0)})
        assert all(isinstance(row[key], int) and not isinstance(row[key], bool) and row[key] >= 0
                   for key in ["n", *CATEGORIES]), "Counts must be nonnegative integers"
        n = row["n"]
        assert sum(row[key] for key in CATEGORIES) == n, "Outcome partition does not sum to bin n"
        for key in ["n", *CATEGORIES]:
            assert row[key] == recounted[bench][label].get(key, 0), f"Independent recount mismatch: {bench}/{label}/{key}"
        percentages = {key: (100.0 * row[key] / n if n else None) for key in CATEGORIES}
        if n:
            assert math.isclose(sum(percentages.values()), 100.0, abs_tol=1e-10)
        rows.append({"label": label, "n": n, **{key: row[key] for key in CATEGORIES}, "percentages": percentages})
        totals.update({key: row[key] for key in ["n", *CATEGORIES]})
    assert totals["n"] == core["benchmarks"][bench]["n_eval"], "Binned questions do not cover evaluation set"
    pool = core["benchmarks"][bench]["fixed_pools"]["2x9"]
    assert set(pool["split"]) == {"qwen", "llama"} and set(pool["split"].values()) == {9}
    assert math.isclose(totals["majority_correct"] / totals["n"], pool["majority"], abs_tol=1e-12)
    assert math.isclose((totals["majority_correct"] + totals["covered_but_missed"]) / totals["n"], pool["coverage_ceiling"], abs_tol=1e-12)
    checked_rows[bench] = [row for row in rows if row["n"]]
    numerical_source["benchmarks"][bench] = {"label": title, "totals": dict(totals), "bins": rows}
    numerical_source["checks"].append(f"{bench}: integer counts; independent per-item recount; within-bin sum 100%; evaluation n; core majority and coverage totals: PASS")

plt.rcParams.update({
    "font.family": "Source Sans 3", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.7,
    "text.color": "#263F50", "axes.labelcolor": "#263F50",
    "axes.edgecolor": "#788992", "xtick.color": "#263F50", "ytick.color": "#263F50",
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "svg.hashsalt": "cscwd-team-fragmentation-v1",
})
fig, axes = plt.subplots(1, 3, figsize=(17 / 2.54, 5.8 / 2.54), sharey=True)
fig.subplots_adjust(left=0.088, right=0.99, top=0.785, bottom=0.30, wspace=0.16)

for panel, (ax, (bench, title)) in enumerate(zip(axes, BENCHMARKS)):
    rows = checked_rows[bench]
    xs = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    for key, color in zip(CATEGORIES, COLORS):
        values = [row["percentages"][key] for row in rows]
        ax.bar(xs, values, bottom=bottom, width=0.70, color=color, edgecolor="white", linewidth=0.5, zorder=2)
        bottom = [x + y for x, y in zip(bottom, values)]
    for x, row in zip(xs, rows):
        ax.text(x, 104, f'n={row["n"]}', ha="center", va="bottom", fontsize=7.6, color="#344C5C")
    ax.set_xticks(xs, [row["label"].replace("-", "\u2013") for row in rows])
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title(f"({chr(97+panel)}) {title}", fontproperties=HEADING, pad=5)
    ax.grid(axis="y", color="#E5EAED", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2.7, width=0.6, pad=2)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Questions within bin (%)", labelpad=3)
fig.suptitle("Qwen + Llama: nine samples per member", fontsize=8.5, y=0.99)
fig.text(0.54, 0.175, "Distinct normalized answers in the pool", ha="center", fontsize=8)
fig.legend([Patch(facecolor=c) for c in COLORS], NAMES, ncol=3, frameon=False,
           loc="lower center", bbox_to_anchor=(0.54, 0.012), fontsize=7.6,
           handlelength=1.6, handletextpad=0.6, columnspacing=1.8)
provenance = ("Source: data/current_extensions.json; SHA-256 " + source_hash
              + ". Independent recount from current_inputs.jsonl, with core accuracy/coverage cross-checks. "
              + "Bars are within-bin outcome percentages; labels show question counts. Empty bins omitted. Descriptive association, not a causal or monotonic claim.")
fig.savefig(OUT / "fragmentation_comic.pdf", metadata={"Title": "Answer-pool outcomes by distinct candidate count", "Subject": provenance, "Author": ""})
fig.savefig(OUT / "fragmentation_comic.svg", metadata={"Title": "Answer-pool outcomes by distinct candidate count", "Description": provenance})
fig.savefig(OUT / "fragmentation_comic.png", dpi=240, metadata={"Description": provenance})
plt.close(fig)
for name in ["fragmentation_comic.pdf", "fragmentation_comic.svg", "fragmentation_comic.png"]:
    numerical_source.setdefault("output_sha256", {})["figures/" + name] = sha256((OUT / name).read_bytes()).hexdigest()
(AUDIT / "fragmentation_comic_numerical_source.json").write_text(json.dumps(numerical_source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("source_sha256=" + source_hash)
for bench, row in numerical_source["benchmarks"].items():
    print(bench, row["totals"])
print("PASS: independent counts, normalization, evaluation totals, and core accuracy/coverage.")
