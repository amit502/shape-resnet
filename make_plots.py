#!/usr/bin/env python3
"""
Generate 6 publication-quality PDF figures:
  1. CIFAR-10-C   — mean accuracy per severity (averaged across corruptions)
  2. CIFAR-100-C  — mean accuracy per severity
  3. ImageNet-100-C — mean accuracy per severity
  4. CIFAR-10-C   — per-corruption subplots (accuracy vs severity)
  5. CIFAR-100-C  — per-corruption subplots
  6. ImageNet-100-C — per-corruption subplots

Usage:
    python make_plots.py
    python make_plots.py --corr_csv csvs/per_corruption.csv --out_dir figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Model config ─────────────────────────────────────────────────────────────
# (model_key, display_name, color, linestyle, marker)
MODEL_STYLE = {
    "baseline_res18":  ("ResNet-18",        "#4878CF", "-",  "o"),
    "shape_res18":     ("Shape-ResNet-18",   "#4878CF", "--", "s"),
    "baseline_res50":  ("ResNet-50",         "#D65F5F", "-",  "o"),
    "shape_res50":     ("Shape-ResNet-50",   "#D65F5F", "--", "s"),
    "baseline_res101": ("ResNet-101",        "#6ACC65", "-",  "o"),
    "shape_res101":    ("Shape-ResNet-101",  "#6ACC65", "--", "s"),
}

CORRUPTION_DISPLAY = {
    "gaussian_noise":    "Gaussian Noise",
    "shot_noise":        "Shot Noise",
    "impulse_noise":     "Impulse Noise",
    "defocus_blur":      "Defocus Blur",
    "glass_blur":        "Glass Blur",
    "motion_blur":       "Motion Blur",
    "zoom_blur":         "Zoom Blur",
    "snow":              "Snow",
    "frost":             "Frost",
    "fog":               "Fog",
    "brightness":        "Brightness",
    "contrast":          "Contrast",
    "elastic_transform": "Elastic",
    "pixelate":          "Pixelate",
    "jpeg_compression":  "JPEG",
}
CORRUPTION_ORDER = list(CORRUPTION_DISPLAY.keys())

DATASET_DISPLAY = {
    "cifar10":     "CIFAR-10-C",
    "cifar100":    "CIFAR-100-C",
    "imagenet100": "ImageNet-100-C",
}

SEVERITIES = [1, 2, 3, 4, 5]
SEV_COLS   = ["s1", "s2", "s3", "s4", "s5"]

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Melt severity columns; average across seeds."""
    df = df[df["model"].isin(MODEL_STYLE)].copy()
    long = df.melt(
        id_vars=["model", "dataset", "seed", "corruption"],
        value_vars=SEV_COLS,
        var_name="sev_col",
        value_name="acc",
    )
    long["severity"] = long["sev_col"].str[1].astype(int)
    # Average over seeds
    agg = (
        long.groupby(["model", "dataset", "corruption", "severity"])["acc"]
        .mean()
        .reset_index()
    )
    return agg


def plot_severity_mean(agg: pd.DataFrame, dataset: str, out_path: Path,
                       caption: str):
    """Line plot: mean accuracy (averaged across all corruptions) vs severity."""
    ds = agg[agg["dataset"] == dataset]
    models = [m for m in MODEL_STYLE if m in ds["model"].unique()]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for m in models:
        name, color, ls, mk = MODEL_STYLE[m]
        sub = (
            ds[ds["model"] == m]
            .groupby("severity")["acc"]
            .mean()
            .reset_index()
        )
        ax.plot(sub["severity"], sub["acc"],
                color=color, linestyle=ls, marker=mk,
                markersize=5, linewidth=1.6, label=name)

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title(f"{DATASET_DISPLAY[dataset]} — Mean Accuracy vs. Severity")
    ax.set_xticks(SEVERITIES)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_per_corruption(agg: pd.DataFrame, dataset: str, out_path: Path,
                        caption: str):
    """Grid of subplots: one per corruption type, accuracy vs severity."""
    ds = agg[agg["dataset"] == dataset]
    models = [m for m in MODEL_STYLE if m in ds["model"].unique()]
    corruptions = [c for c in CORRUPTION_ORDER if c in ds["corruption"].unique()]

    ncols = 3
    nrows = int(np.ceil(len(corruptions) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.8, nrows * 2.0),
                             sharey=False)
    axes = axes.flatten()

    for i, corr in enumerate(corruptions):
        ax = axes[i]
        sub_corr = ds[ds["corruption"] == corr]
        for m in models:
            name, color, ls, mk = MODEL_STYLE[m]
            sub = sub_corr[sub_corr["model"] == m].sort_values("severity")
            ax.plot(sub["severity"], sub["acc"],
                    color=color, linestyle=ls, marker=mk,
                    markersize=3.5, linewidth=1.4, label=name)
        ax.set_title(CORRUPTION_DISPLAY[corr], fontsize=8)
        ax.set_xticks(SEVERITIES)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        if i % ncols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=7)
        if i // ncols == nrows - 1 or i == len(corruptions) - 1:
            ax.set_xlabel("Severity", fontsize=7)

    # Hide unused subplots
    for j in range(len(corruptions), len(axes)):
        axes[j].set_visible(False)

    # Shared legend below the grid
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               ncol=len(models),
               framealpha=0.9,
               fontsize=8,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Per-Corruption Accuracy vs. Severity",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_csv", default="csvs/per_corruption.csv")
    ap.add_argument("--out_dir",  default="figures")
    args = ap.parse_args()

    df  = pd.read_csv(args.corr_csv)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    agg = prepare(df)

    datasets = [
        ("cifar10",     "CIFAR-10-C severity-averaged accuracy. Shape-biased models "
                        "consistently outperform baseline ResNets as corruption severity increases."),
        ("cifar100",    "CIFAR-100-C severity-averaged accuracy. Shape-biased models "
                        "maintain higher accuracy under increasing corruption severity."),
        ("imagenet100", "ImageNet-100-C severity-averaged accuracy. Shape-biased models "
                        "maintain higher accuracy under increasing corruption severity."),
    ]

    for ds, caption in datasets:
        if agg[agg["dataset"] == ds].empty:
            print(f"Skipping {ds}: no data.")
            continue
        plot_severity_mean(agg, ds, out / f"severity_mean_{ds}.pdf", caption)
        plot_per_corruption(agg, ds, out / f"per_corruption_{ds}.pdf", caption)


if __name__ == "__main__":
    main()
