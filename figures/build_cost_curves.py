"""Plot only independently reproduced stopping curves from current_results.json.

Posterior and vote-threshold points are observed non-dominated grid points;
the fixed-stream curve retains every observed budget. Segments are visual
guides, and the best-single reference is the observed nine-call evaluation arm.
No proxy baseline or empirical value is supplied outside the source JSON.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comic_plot_style import HEADING
from matplotlib.ticker import FormatStrFormatter

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "data" / "current_results.json"
raw = SOURCE.read_bytes()
source_sha256 = hashlib.sha256(raw).hexdigest()
report = json.loads(raw)

plt.rcParams.update({
    "font.family": "Source Sans 3",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.7,
    "axes.edgecolor": "#75848C",
    "axes.labelcolor": "#263F50",
    "xtick.color": "#263F50",
    "ytick.color": "#263F50",
    "text.color": "#263F50",
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "svg.hashsalt": "cscwd-team-stopping-v1",
})

fig, axes = plt.subplots(1, 3, figsize=(17 / 2.54, 5.2 / 2.54))
fig.subplots_adjust(left=0.082, right=0.99, top=0.855, bottom=0.345, wspace=0.34)
config = [
    ("mmlu_pro", "MMLU-Pro", (39, 71), [40, 50, 60, 70]),
    ("math500", "MATH-500", (49, 66), [50, 55, 60, 65]),
    ("gsm8k", "GSM8K", (80, 96), [80, 85, 90, 95]),
]
curve_config = [
    ("fixed_stream_curve", "Fixed stream (majority)", "#4D86AB", "o", 1.15, 2),
    ("vote_threshold_pareto", "Vote threshold (Pareto)", "#428D80", "s", 1.2, 3),
    ("posterior_pareto", "Posterior stopping (Pareto)", "#C48743", "^", 1.45, 4),
]
handles = {}

for panel, (ax, (bench, title, ylim, yticks)) in enumerate(zip(axes, config)):
    z = report["benchmarks"][bench]["stopping"]
    ref = 100 * z["single9_reference"]["accuracy"]
    h = ax.axhline(ref, color="#8D979D", ls=(0, (4, 2.6)), lw=0.95,
                   zorder=1, label="Best single model (9 calls)")
    handles["Best single model (9 calls)"] = h
    # Label the reference at its line instead of making readers estimate its value.
    # On GSM8K use the space below the line, away from the nearly coincident curve.
    ax.text(18.1, ref + (-0.45 if bench == "gsm8k" else 0.45), f"{ref:.1f}%",
            ha="right", va="top" if bench == "gsm8k" else "bottom",
            fontsize=7.5, color="#455661",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6}, zorder=5)
    for key, label, color, marker, lw, order in curve_config:
        rows = z[key]
        xy = sorted((float(v["cost"]), 100 * float(v["accuracy"])) for v in rows)
        assert all(1 <= x <= 18 and 0 <= y <= 100 for x, y in xy)
        h, = ax.plot([x for x, _ in xy], [y for _, y in xy], color=color,
                     marker=marker, ms=3.3 if marker != "^" else 4.0,
                     markeredgecolor="white", markeredgewidth=0.35,
                     lw=lw, zorder=order, label=label, clip_on=False)
        handles[label] = h
    ax.set(title=f"({chr(97+panel)}) {title}", xlim=(0.8, 18.35), ylim=ylim, xlabel="Mean calls per question")
    ax.title.set_fontproperties(HEADING)
    ax.set_xticks([1, 6, 12, 18])
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="y", color="#E5EAED", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.7, width=0.6, pad=2.2)

axes[0].set_ylabel("Accuracy (%)", labelpad=3.7)
legend_order = ["Posterior stopping (Pareto)", "Fixed stream (majority)",
                "Vote threshold (Pareto)", "Best single model (9 calls)"]
fig.legend([handles[k] for k in legend_order], legend_order, loc="lower center",
           bbox_to_anchor=(0.53, 0.012), ncol=2, frameon=False,
           columnspacing=2.2, handlelength=2.5, handletextpad=0.7,
           borderaxespad=0, labelspacing=0.65)

provenance = ("Source: data/current_results.json; SHA-256 " + source_sha256
              + ". Posterior and vote curves: observed Pareto grid points; fixed stream: all observed budgets. "
              + "Segments are visual guides. Accuracy displayed as percentage; single-model reference is post-hoc strongest evaluation arm.")
fig.savefig(OUT / "cost_curves_comic.pdf", metadata={"Title": "Observed cost-accuracy operating points",
                                              "Subject": provenance, "Author": ""})
fig.savefig(OUT / "cost_curves_comic.svg", metadata={"Title": "Observed cost-accuracy operating points", "Description": provenance})
fig.savefig(OUT / "cost_curves_comic.png", dpi=240, metadata={"Description": provenance})
plt.close(fig)
print("source_sha256=" + source_sha256)
for bench, _, _, _ in config:
    z = report["benchmarks"][bench]["stopping"]
    print(bench, {k: len(z[k]) for k, *_ in curve_config}, "single9=", z["single9_reference"]["accuracy"])
for suffix in ("pdf", "svg", "png"):
    print(OUT / ("cost_curves_comic." + suffix))
