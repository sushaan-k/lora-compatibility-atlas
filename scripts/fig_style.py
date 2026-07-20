"""Shared matplotlib style for the four main-text figures.

All figures are rendered at their final physical size (TMLR text width is
6.5in; each figsize equals the width at which the .tex includes the PDF), so
the 8pt base font is the printed font size.  Style: boxed axes with a dotted
grid, serif text, Okabe-Ito colorblind-safe palette.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#888888"

# One color per merge operator, used consistently across every panel.
OPERATOR_COLORS = {
    "linear": BLUE,
    "ties": VERMILLION,
    "dare_ties": GREEN,
    "magnitude_prune": PURPLE,
}
OPERATOR_LABELS = {
    "linear": "Linear",
    "ties": "TIES",
    "dare_ties": "DARE-TIES",
    "magnitude_prune": "Mag. pruning",
}
OPERATOR_MARKERS = {
    "linear": "o",
    "ties": "s",
    "dare_ties": "^",
    "magnitude_prune": "D",
}
OPERATOR_LINESTYLES = {
    "linear": "-",
    "ties": "--",
    "dare_ties": "-.",
    "magnitude_prune": (0, (3, 1, 1, 1)),
}


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "0.6",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
