"""Figure 4 (revision): model dependence as gradient-filled ridgelines.

Data source
    results/revised/stats/block_contrasts.csv (20 paired seed blocks per cell)
    contrast_id = BC_B2vB1_C2vC1 and BC_B3vB1_C3vC1, metric = EO,
    mechanism_scope = "marginal" (already averaged over A and M).

Design
    Two rows (contrast) x two columns (classifier). Each panel stacks the five
    datasets as ridgelines. The fill is a horizontal gradient evaluated on the
    shared x-scale and mapped through a diverging colormap centred at zero:
    blue marks a reduction of the EO gap, red an increase. Ridge outlines,
    per-block rug marks and 95% t intervals are drawn on top.

Run
    python make_fig4_ridgeline.py

Outputs
    manuscript/figures/revised/fig4_model_dependence.pdf   (vector text/outlines)
    manuscript/figures/revised/fig4_preview.png             (QA preview)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy import stats
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]  # repository root
STATS = ROOT / "results" / "revised" / "stats" / "block_contrasts.csv"
OUT_DIR = ROOT / "results" / "revised" / "figures"

DATASETS = ["adult", "german", "bank", "diabetes", "compas"]
LABELS = {"adult": "Adult", "german": "German", "bank": "Bank",
          "diabetes": "Diabetes", "compas": "COMPAS"}
CONTRASTS = [("BC_B2vB1_C2vC1", r"B2$\times$C2", "o"),
             ("BC_B3vB1_C3vC1", r"B3$\times$C3", "s")]
MODELS = [("LR", "LR"), ("LGBM", "LightGBM")]

# Diverging map: blue (gap reduced) - neutral - red (gap increased)
CMAP = LinearSegmentedColormap.from_list(
    "fair_diverging",
    ["#2c7bb6", "#a6d5e8", "#f7f7f7", "#fdc98f", "#d7191c"],
    N=256,
)

X_LIM = (-0.19, 0.25)
NORM = TwoSlopeNorm(vmin=X_LIM[0], vcenter=0.0, vmax=X_LIM[1])


def load_blocks() -> pd.DataFrame:
    df = pd.read_csv(STATS, encoding="utf-8-sig")
    d = df[(df.metric == "EO") & (df.mechanism_scope == "marginal")
           & (df.contrast_id.isin([c[0] for c in CONTRASTS])
              | df.contrast_id.str.startswith("BC_B2vB1_C2vC1")
              | df.contrast_id.str.startswith("BC_B3vB1_C3vC1"))]
    d = d[d.contrast_id.isin([c[0] for c in CONTRASTS])]
    return d


def ridge_profile(values: np.ndarray, grid: np.ndarray,
                   rel_bw: float = 0.35, min_bw: float = 0.008):
    """Gaussian KDE on the block values, scaled to a maximum of 1.

    A relative bandwidth alone produces needle-like ridges for cells with a
    small standard deviation (Bank, COMPAS), so the bandwidth is floored at an
    absolute value on the effect scale.
    """
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1)
    bw = max(rel_bw * sd, min_bw)
    if sd < 1e-9:
        v = v + np.linspace(-1e-4, 1e-4, len(v))
    z = (grid[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * z ** 2).sum(axis=1)
    dens -= dens.min()
    peak = dens.max()
    return dens / peak if peak > 0 else dens


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    m = v.mean()
    se = v.std(ddof=1) / np.sqrt(len(v))
    t = stats.t.ppf(0.975, len(v) - 1)
    return m, m - t * se, m + t * se


def make_figure(df: pd.DataFrame):
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

    fig, axes = plt.subplots(2, 2, figsize=(4.75, 5.15), sharex=True)
    spacing = 1.0
    for r, (cid, ctitle, marker) in enumerate(CONTRASTS):
        for c, (model, mtitle) in enumerate(MODELS):
            ax = axes[r, c]
            ax.set_xlim(*X_LIM)
            ax.set_ylim(-0.45, len(DATASETS) * spacing + 0.55)

            # zero reference
            ax.axvline(0.0, color="#555555", lw=0.7, ls=(0, (3, 2), ), zorder=1)

            grid = np.linspace(X_LIM[0], X_LIM[1], 600)

            # draw from the bottom dataset upward so upper ridges sit behind
            for i, ds in enumerate(DATASETS):
                row = len(DATASETS) - 1 - i  # Adult at the top, COMPAS at the bottom
                vals = df[(df.dataset == ds) & (df.model_name == model)
                          & (df.contrast_id == cid)].sort_values("seed").value.to_numpy()
                base = row * spacing
                heights = ridge_profile(vals, grid) * 0.85
                y_top = base + heights

                verts = np.vstack([np.column_stack([grid, y_top]),
                                   [grid[-1], base], [grid[0], base]])
                clip = PathPatch(MplPath(verts), transform=ax.transData,
                                 facecolor="none", edgecolor="none")
                ax.add_patch(clip)

                # Horizontal gradient evaluated on x only, shared norm across panels
                ramp = np.tile(grid, (2, 1))
                img = ax.imshow(ramp, origin="lower", aspect="auto", cmap=CMAP,
                                norm=NORM, interpolation="bilinear",
                                extent=(grid[0], grid[-1], base,
                                        base + max(heights.max(), 1e-3)),
                                zorder=2)
                img.set_clip_path(clip)

                # ridge outline (only where density is non-negligible, so the tail
                # does not paint a full-width line along the baseline)
                mask = heights > 0.004
                ax.plot(grid[mask], y_top[mask], color="#1a1a1a", lw=0.55,
                        alpha=0.95, solid_capstyle="round", zorder=4,
                        antialiased=True)
                sup = (vals.min() - 0.03, vals.max() + 0.03)
                sup = (max(sup[0], X_LIM[0]), min(sup[1], X_LIM[1]))
                ax.plot(list(sup), [base, base], color="#1a1a1a", lw=0.35,
                        alpha=0.5, zorder=4)

                # per-block rug
                ax.vlines(vals, base + 0.015, base + 0.085,
                          color="#333333", lw=0.45, alpha=0.75, zorder=5)

                # mean and 95% t interval
                m, lo, hi = ci95(vals)
                ax.hlines(base + 0.175, lo, hi, color="#111111", lw=0.85, zorder=6)
                ax.plot([lo, hi], [base + 0.175] * 2, color="#111111", lw=0.55,
                        zorder=5)
                ax.plot(m, base + 0.175, marker=marker, ms=4.6, mfc="white",
                        mec="#111111", mew=0.85, zorder=7)

            ax.set_yticks([len(DATASETS) - 1 - i for i in range(len(DATASETS))])
            ax.set_yticklabels([LABELS[d] for d in DATASETS] if c == 0 else [])
            ax.tick_params(axis="y", length=0, pad=6, labelsize=9)

            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.7)
            ax.spines["bottom"].set_color("#333333")
            ax.tick_params(axis="x", direction="out", length=3.6, width=0.7,
                           pad=4, labelsize=8.5)

            ax.text(0.02, 0.97, ctitle, transform=ax.transAxes, ha="left",
                    va="top", fontsize=9.5)
            if r == 0:
                ax.set_title(mtitle, fontsize=10, pad=6)

    for ax in axes[1, :]:
        ax.set_xlabel("EO difference-in-differences", fontsize=9.5, labelpad=6)

    # Shared gradient legend at the bottom: one horizontal colour ramp
    cax = fig.add_axes([0.28, 0.055, 0.44, 0.018])
    sm = mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-0.15, -0.05, 0.0, 0.05, 0.15, 0.25])
    cb.ax.tick_params(labelsize=6.6, length=2.2, width=0.55, pad=1.5)
    cb.outline.set_linewidth(0.55)
    # label above the ramp so it cannot be clipped by the figure edge
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.set_label("blue: EO gap reduced   |   red: EO gap increased",
                 fontsize=6.8, labelpad=1)

    fig.subplots_adjust(left=0.155, right=0.985, top=0.945, bottom=0.235,
                        wspace=0.10, hspace=0.18)
    return fig


def main() -> None:
    df = load_blocks()
    missing = [(d, m, c) for c, *_ in CONTRASTS for m, _ in MODELS for d in DATASETS
               if df[(df.dataset == d) & (df.model_name == m)
                     & (df.contrast_id == c)].shape[0] == 0]
    if missing:
        raise SystemExit(f"missing cells: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(df)
    fig.savefig(OUT_DIR / "fig4_model_dependence.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "fig4_preview.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("written:", OUT_DIR / "fig4_model_dependence.pdf")


if __name__ == "__main__":
    main()
