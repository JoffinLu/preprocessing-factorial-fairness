"""Figure 5 (revision): utility-disparity plane, one point per B x C cell.

Data source
    results/revised/stats/cell_means.csv
    metric DP (vertical, lower is better) and F1 (horizontal, higher is better),
    averaged over the two missingness mechanisms, the three imputers A1-A3 and
    the 20 seed blocks. Averaging over A is deliberate: the three imputers move
    a cell by a median of 0.005 in DP, which is below one marker width at the
    printed scale, so drawing them separately only produces overprinted blobs.

Design
    Five dataset rows x two classifier columns. Marker shape encodes B and
    marker fill encodes C: C1 open, C2 blue, C3 orange. Panels in a row share
    limits; both axes start at zero. A light grey path joins C1-C2-C3 within
    each B so the triplets read as one repair trajectory.

Run
    python make_fig5_tradeoff.py

Outputs
    manuscript/figures/revised/fig5_tradeoff.pdf
    manuscript/figures/revised/fig5_preview.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]  # repository root
STATS = ROOT / "results" / "revised" / "stats" / "cell_means.csv"
OUT_DIR = ROOT / "results" / "revised" / "figures"

DATASETS = ["adult", "german", "bank", "diabetes", "compas"]
LABELS = {"adult": "Adult", "german": "German", "bank": "Bank",
          "diabetes": "Diabetes", "compas": "COMPAS"}
MODELS = [("LR", "LR"), ("LGBM", "LightGBM")]

B_ORDER = ["B1", "B2", "B3"]
B_MARKER = {"B1": "o", "B2": "s", "B3": "^"}
B_LABEL = {"B1": "B1: none", "B2": "B2: weights", "B3": "B3: group SMOTE"}
C_ORDER = ["C1", "C2", "C3"]
# Translucent fills: where two cells nearly coincide, both edges remain visible
# instead of the later marker painting over the earlier one.
C_FACE = {
    "C1": (1.0, 1.0, 1.0, 0.0),
    "C2": (0.0, 0.447, 0.698, 0.40),
    "C3": (0.835, 0.369, 0.0, 0.40),
}
C_EDGE = {"C1": "#111111", "C2": "#00456e", "C3": "#7a3600"}


def load_cells() -> pd.DataFrame:
    d = pd.read_csv(STATS, encoding="utf-8-sig")
    d = d[d.metric.isin(["DP", "F1"])]
    cells = (d.groupby(["dataset", "model_name", "B", "C", "metric"])["mean"]
             .mean().reset_index())
    wide = cells.pivot_table(index=["dataset", "model_name", "B", "C"],
                             columns="metric", values="mean").reset_index()
    missing = wide[["DP", "F1"]].isna().any(axis=1).sum()
    if missing:
        raise SystemExit(f"{missing} incomplete cells")
    return wide


def axis_limits(cells: pd.DataFrame, ds: str) -> tuple[tuple[float, float], tuple[float, float]]:
    s = cells[cells.dataset == ds]
    x_top = max(0.20, np.ceil(s.F1.max() * 1.10 * 100) / 100)
    y_top = max(0.10, np.ceil(s.DP.max() * 1.15 * 100) / 100)
    return (0.0, x_top), (0.0, y_top)


def make_figure(cells: pd.DataFrame):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(5, 2, figsize=(4.75, 5.45), sharex=False, sharey=False)
    for r, ds in enumerate(DATASETS):
        (x0, x1), (y0, y1) = axis_limits(cells, ds)
        for c, (model, mtitle) in enumerate(MODELS):
            ax = axes[r, c]
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

            s = cells[(cells.dataset == ds) & (cells.model_name == model)]

            # repair trajectories C1 -> C2 -> C3 within each B, drawn first
            for b in B_ORDER:
                q = s[s.B == b].set_index("C").loc[C_ORDER]
                ax.plot(q.F1, q.DP, color="#b0b0b0", lw=0.7, alpha=0.75,
                        zorder=2, solid_capstyle="round")
                for j, cname in enumerate(C_ORDER):
                    row = q.loc[cname]
                    ax.plot(row.F1, row.DP, marker=B_MARKER[b], ms=4.2,
                            mfc=C_FACE[cname], mec=C_EDGE[cname], mew=0.85,
                            ls="none", zorder=4 + (2 - j) * 0.1, clip_on=False)

            ax.text(0.02, 0.97, f"({chr(97 + 2 * r + c)}) {LABELS[ds]} / {mtitle}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.6)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.spines["left"].set_color("#333333")
            ax.spines["bottom"].set_color("#333333")
            ax.tick_params(direction="out", length=3.2, width=0.6, pad=3,
                           labelsize=8, colors="#222222")
            if c == 1:
                ax.set_yticklabels([])

    for c in range(2):
        axes[-1, c].set_xlabel("F1 score (higher)", fontsize=9, labelpad=5)
    fig.text(0.035, 0.50, "Demographic parity gap (lower)", rotation=90,
             va="center", ha="center", fontsize=9)

    # Two legend rows under the panels: C encoding and B encoding.
    c_handles = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=C_FACE[c],
                        mec=C_EDGE[c], mew=0.85, label=f"{c}: {lab}")
                 for c, lab in zip(C_ORDER, ["none", "reweigh", "quantile"])]
    b_handles = [Line2D([], [], ls="none", marker=B_MARKER[b], ms=4.2,
                        mfc=(1, 1, 1, 0), mec="#111111", mew=0.85, label=B_LABEL[b])
                 for b in B_ORDER]
    fig.legend(handles=c_handles, loc="lower center",
               bbox_to_anchor=(0.52, 0.078), ncol=3, frameon=False,
               fontsize=8, handletextpad=0.4, columnspacing=1.4)
    fig.legend(handles=b_handles, loc="lower center",
               bbox_to_anchor=(0.52, 0.030), ncol=3, frameon=False,
               fontsize=8, handletextpad=0.4, columnspacing=1.4)

    fig.subplots_adjust(left=0.145, right=0.975, top=0.965, bottom=0.235,
                        wspace=0.10, hspace=0.30)
    return fig


def main() -> None:
    cells = load_cells()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(cells)
    fig.savefig(OUT_DIR / "fig5_tradeoff.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "preview_fig5.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("written:", OUT_DIR / "fig5_tradeoff.pdf")


if __name__ == "__main__":
    main()
