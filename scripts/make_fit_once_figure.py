#!/usr/bin/env python3
"""Main figure for the fit-once answer-token index (three panels from real data):
(a) fidelity, predicted vs measured answer loss at m=16 and m=8;
(b) atlas-coefficient benefit on contested vs easy triples;
(c) design-density law, held-out fidelity vs redundancy.
Writes figs_png/fit_once_index.pdf.

Rendered at final physical size (6.5in wide; the .tex includes it at
\\linewidth on a 6.5in TMLR text block) with the shared 8pt boxed style.
"""
import json
import collections
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fig_style import apply_style, BLUE, SKY, VERMILLION

import matplotlib.pyplot as plt
import numpy as np

apply_style()

fig, ax = plt.subplots(
    1, 3, figsize=(6.5, 2.35), layout="constrained",
    gridspec_kw={"width_ratios": [0.92, 1.04, 1.04]},
)

# ---- (a) fidelity: predicted vs measured answer loss ----
def per_member(p):
    d = json.load(open(REPO / "results_public" / p / "answer_panel_analysis.json"))
    pm = d["per_member"]
    return (np.array([e["pred_L_answer"] for e in pm]),
            np.array([e["L_answer"] for e in pm]),
            np.array([e["order"] for e in pm]))

p16, m16, o16 = per_member("scale_m16")
p8, m8, o8 = per_member("answer_panel_m8")

# Identical square limits covering every point (predictions dip below zero,
# so log-log is not an option; keep linear axes).
lo = min(0.0, p16.min(), p8.min(), m16.min(), m8.min()) - 0.35
hi = max(m16.max(), p16.max(), m8.max(), p8.max()) * 1.03
lim = [lo, hi]

# m=16: distinguish held-out triples (triangles) from design-point pairs (circles)
pr = o16 == 2
tr = o16 == 3
# correlations (0.98 / 0.96 / 0.52) are stated in the caption
ax[0].scatter(m16[pr], p16[pr], s=9, marker="o", c=SKY, edgecolors="none",
              zorder=3, label="$m{=}16$ pairs")
ax[0].scatter(m16[tr], p16[tr], s=11, marker="^", c=BLUE, edgecolors="none",
              zorder=4, label="$m{=}16$ triples")
# m=8 drawn last, open grey circles, so it stays visible over the m=16 cloud
ax[0].scatter(m8, p8, s=11, marker="o", facecolors="none", edgecolors="0.35",
              linewidths=0.6, zorder=5, label="$m{=}8$")
# identity line, labelled in the legend
ax[0].plot(lim, lim, "k--", lw=0.8, alpha=0.6, zorder=2, label="$y=x$")
ax[0].set_xlim(lim)
ax[0].set_ylim(lim)
ax[0].set_aspect("equal", adjustable="box")
ax[0].set_xlabel("measured answer loss (nats)")
ax[0].set_ylabel("predicted answer loss (nats)")
ax[0].set_title("(a) off-design fidelity, single fit")
# lower-right triangle is empty (no point has measured>7 and predicted<5)
leg = ax[0].legend(frameon=True, loc="lower right", fontsize=6,
                   handletextpad=0.1, borderpad=0.25, labelspacing=0.2,
                   handlelength=1.2, borderaxespad=0.25)
leg.get_frame().set_alpha(1.0)

# ---- (b) coefficient benefit: worst-member accuracy, atlas vs equal ----
rows = json.load(open(REPO / "results_public" / "coefficient" / "coeff_rows.json"))["rows"]
byv = collections.defaultdict(dict)
for r in rows:
    byv[(r["subset"], r["task"])][r["variant"]] = r
sub = collections.defaultdict(lambda: {"opt": [], "eq": []})
for (s, t), v in byv.items():
    if "atlas_opt" in v and "equal" in v:
        sub[s]["opt"].append(v["atlas_opt"]["accuracy"])
        sub[s]["eq"].append(v["equal"]["accuracy"])
eq_worst = {s: min(x["eq"]) for s, x in sub.items()}
opt_worst = {s: min(x["opt"]) for s, x in sub.items()}
hard = [s for s in sub if eq_worst[s] < 0.6]
easy = [s for s in sub if eq_worst[s] >= 0.6]
for grp, x0, col, lab in [(easy, 0, "0.6", "easy"), (hard, 1, VERMILLION, "hard")]:
    xs = np.full(len(grp), x0) + np.random.default_rng(0).uniform(-0.06, 0.06, len(grp))
    d = np.array([opt_worst[s] - eq_worst[s] for s in grp])
    ax[1].scatter(xs, d, s=14, c=col, edgecolors="none", zorder=3,
                  label=f"{lab} ($n{{=}}{len(grp)}$)")
    ax[1].scatter([x0], [d.mean()], marker="_", s=500, c="k", lw=1.4, zorder=4)
ax[1].axhline(0, color="k", lw=0.7)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["easy", "hard"])
ax[1].set_xlim(-0.55, 1.55)
ax[1].set_ylim(-0.26, 0.235)
ax[1].set_ylabel("worst-member accuracy:\natlas coeff. $-$ equal weight")
ax[1].set_title("(b) coefficient value")
# framed legend centred between the two data columns, above all points
leg_b = ax[1].legend(frameon=True, loc="upper center", fontsize=6,
                     handletextpad=0.1, borderpad=0.3, labelspacing=0.25,
                     borderaxespad=0.3)
leg_b.get_frame().set_alpha(1.0)

# ---- (c) design-density law ----
dens = json.load(open(REPO / "results_public" / "density" / "density_analysis.json"))["levels"]
red = [l["redundancy"] for l in dens]
r = [l["r_pred_vs_measured"] for l in dens]
ax[2].plot(red, r, "o-", c=BLUE, lw=1.4, ms=4, zorder=3)
ax[2].set_xlabel("design redundancy ($\\times$ poised)")
ax[2].set_ylabel("off-design fidelity (Pearson $r$)")
ax[2].set_title("(c) fidelity vs design density")
ax[2].set_ylim(0.6, 0.8)

out = REPO / "figs_png" / "fit_once_index.pdf"
fig.savefig(out)
print("wrote", out)
