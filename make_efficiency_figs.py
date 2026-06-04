#!/usr/bin/env python3
"""
make_efficiency_figs.py — Profiling tables and figures from profiler CSVs.

Finds the latest profile_<dataset>_*.csv in each provided directory, then
generates:
  - LaTeX efficiency table  (tables/efficiency_table.tex + .csv)
  - PDF bar-chart figures   (figures/efficiency_cifar.pdf,
                             figures/efficiency_imagenet100.pdf)

Usage:
    python make_efficiency_figs.py \\
        --cifar_dir     /pvc/results/profile \\
        --imagenet_dir  /pvc/results/profile

    # Both datasets in the same folder:
    python make_efficiency_figs.py --cifar_dir csvs/ --imagenet_dir csvs/
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--cifar_dir",    required=True,
                help="Folder containing profile_cifar10_*.csv")
ap.add_argument("--imagenet_dir", default=None,
                help="Folder containing profile_imagenet100_*.csv (optional)")
ap.add_argument("--out_dir",      default="figures")
ap.add_argument("--table_dir",    default="tables")
args = ap.parse_args()

Path(args.out_dir).mkdir(exist_ok=True)
Path(args.table_dir).mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def latest_csv(folder: str, pattern: str) -> pd.DataFrame:
    """Return the most recently modified CSV matching pattern in folder."""
    matches = sorted(glob.glob(os.path.join(folder, pattern)),
                     key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in '{folder}'")
    path = matches[-1]
    print(f"  Loading {path}")
    return pd.read_csv(path)


# ── Load data ─────────────────────────────────────────────────────────────────
df_cifar = latest_csv(args.cifar_dir, "profile_cifar10_*.csv")
df_cifar["display_dataset"] = "CIFAR"
frames      = [df_cifar]
df_imagenet = None

if args.imagenet_dir:
    try:
        df_imagenet = latest_csv(args.imagenet_dir, "profile_imagenet100_*.csv")
        df_imagenet["display_dataset"] = "ImageNet-100"
        frames.append(df_imagenet)
    except FileNotFoundError:
        print("  [INFO] No imagenet100 profile CSV found — skipping.")

df_all = pd.concat(frames, ignore_index=True)


# ── Display names ─────────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "baseline_res18":  "ResNet-18",
    "shape_res18":     "Shape-ResNet-18",
    "baseline_res50":  "ResNet-50",
    "shape_res50":     "Shape-ResNet-50",
    "baseline_res101": "ResNet-101",
    "shape_res101":    "Shape-ResNet-101",
}

# Backbone pairs: (baseline_key, shape_key, backbone_label, color)
BACKBONE_PAIRS = [
    ("baseline_res18",  "shape_res18",  "ResNet-18",  "#4878CF"),
    ("baseline_res50",  "shape_res50",  "ResNet-50",  "#D65F5F"),
    ("baseline_res101", "shape_res101", "ResNet-101", "#6ACC65"),
]

# Metrics shown in figures
METRICS = [
    ("params_total_M",   "Parameters (M)",    False),   # (col, label, higher_is_better)
    ("gflops",           "GFLOPs",            False),
    ("latency_mean_ms",  "Latency (ms)",      False),
    ("throughput_img_s", "Throughput (images/s)", True),
    ("peak_mem_mb",      "Peak Memory (MB)",  False),
]

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Bar-chart figure per dataset ───────────────────────────────────────────────
def make_figure(df: pd.DataFrame, dataset_label: str, out_path: str):
    """
    One row of subplots, one per metric.
    For each subplot: grouped bars — one group per backbone pair,
    two bars per group (baseline, shape).
    """
    # Backbone pairs present in this dataset
    present_pairs = [
        (b, s, lbl, col) for b, s, lbl, col in BACKBONE_PAIRS
        if b in df["model"].values and s in df["model"].values
    ]
    if not present_pairs:
        print(f"  [WARN] No complete backbone pairs for {dataset_label}, skipping.")
        return

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 2.6, 3.2))

    bar_w  = 0.32
    x_base = np.arange(len(present_pairs))

    for ax, (col, ylabel, higher_is_better) in zip(axes, METRICS):
        for i, (base_key, shape_key, lbl, color) in enumerate(present_pairs):
            b_row = df[df["model"] == base_key]
            s_row = df[df["model"] == shape_key]
            if b_row.empty or s_row.empty:
                continue

            b_val = float(b_row[col].iloc[0])
            s_val = float(s_row[col].iloc[0])

            ax.bar(x_base[i] - bar_w / 2, b_val,
                   width=bar_w, color=color, alpha=0.85,
                   label="Baseline" if i == 0 else "")
            ax.bar(x_base[i] + bar_w / 2, s_val,
                   width=bar_w, color=color, alpha=0.85,
                   hatch="///", edgecolor="white",
                   label="Shape" if i == 0 else "")

            # delta annotation above the shape bar
            delta_pct = (s_val - b_val) / b_val * 100
            sign      = "+" if delta_pct > 0 else ""
            bar_top   = max(b_val, s_val)
            ax.text(x_base[i] + bar_w / 2, bar_top * 1.03,
                    f"{sign}{delta_pct:.0f}%",
                    ha="center", va="bottom", fontsize=6.5, color="#333333")

        ax.set_ylabel(ylabel)
        ax.set_xticks(x_base)
        ax.set_xticklabels([lbl for _, _, lbl, _ in present_pairs],
                           rotation=15, ha="right")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    # Shared legend
    solid_patch   = mpatches.Patch(facecolor="#888888", alpha=0.85, label="Baseline")
    hatch_patch   = mpatches.Patch(facecolor="#888888", alpha=0.85,
                                   hatch="///", edgecolor="white", label="Shape")
    fig.legend(handles=[solid_patch, hatch_patch],
               loc="lower center", ncol=2, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f"{dataset_label} — efficiency comparison (baseline vs shape-biased)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


make_figure(df_cifar, "CIFAR", f"{args.out_dir}/efficiency_cifar.pdf")
if df_imagenet is not None:
    make_figure(df_imagenet, "ImageNet-100", f"{args.out_dir}/efficiency_imagenet100.pdf")


# ── LaTeX + CSV table ─────────────────────────────────────────────────────────
def fmt(val: float, bold: bool = False) -> str:
    s = f"{val:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def make_latex_table(df: pd.DataFrame):
    """Returns (latex_str, csv_rows)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Efficiency comparison of baseline and shape-biased models "
        r"on {CIFAR}-10 (32$\times$32 input, batch size 1). "
        r"Bold indicates the more efficient value within each backbone pair.}",
        r"\begin{center}",
        r"\begin{tabular}{|l|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Model}"
        r"  & \rotatebox{90}{\textbf{Params (M)}\hspace{4pt}}"
        r"  & \rotatebox{90}{\textbf{GFLOPs}\hspace{4pt}}"
        r"  & \rotatebox{90}{\textbf{Latency (ms)}\hspace{4pt}}"
        r"  & \rotatebox{90}{\textbf{Throughput (images/s)}\hspace{4pt}}"
        r"  & \rotatebox{90}{\textbf{Peak Memory (MB)}\hspace{4pt}} \\",
        r"\hline",
    ]
    csv_rows = []

    ds_df = df[df["display_dataset"] == "CIFAR"]
    pairs = [
        (b, s, lbl) for b, s, lbl, _ in BACKBONE_PAIRS
        if b in ds_df["model"].values and s in ds_df["model"].values
    ]

    for pair_idx, (base_key, shape_key, _) in enumerate(pairs):
        for model_key in (base_key, shape_key):
            row = ds_df[ds_df["model"] == model_key]
            if row.empty:
                continue
            r = row.iloc[0]

            partner_key = shape_key if model_key == base_key else base_key
            partner_row = ds_df[ds_df["model"] == partner_key]

            def is_bold_lower(col):
                if partner_row.empty:
                    return False
                return float(r[col]) < float(partner_row.iloc[0][col])

            def is_bold_higher(col):
                if partner_row.empty:
                    return False
                return float(r[col]) > float(partner_row.iloc[0][col])

            disp        = MODEL_DISPLAY.get(model_key, model_key)
            params_cell = fmt(r["params_total_M"],   is_bold_lower("params_total_M"))
            gflop_cell  = fmt(r["gflops"],           is_bold_lower("gflops"))
            lat_cell    = fmt(r["latency_mean_ms"],  is_bold_lower("latency_mean_ms"))
            tput_cell   = fmt(r["throughput_img_s"], is_bold_higher("throughput_img_s"))
            mem_cell    = fmt(r["peak_mem_mb"],      is_bold_lower("peak_mem_mb"))

            lines.append(
                f"{disp} & {params_cell} & {gflop_cell} "
                f"& {lat_cell} & {tput_cell} & {mem_cell} \\\\"
            )
            csv_rows.append({
                "Model":            disp,
                "Params (M)":       f"{r['params_total_M']:.2f}",
                "GFLOPs":           f"{r['gflops']:.2f}",
                "Latency (ms)":     f"{r['latency_mean_ms']:.2f}",
                "Throughput (images/s)": f"{r['throughput_img_s']:.1f}",
                "Peak Memory (MB)": f"{r['peak_mem_mb']:.2f}",
            })

        if pair_idx < len(pairs) - 1:
            lines.append(r"\hline")

    lines += [r"\hline", r"\end{tabular}", r"\label{tab:efficiency}",
              r"\end{center}", r"\end{table}"]
    return "\n".join(lines), csv_rows


tex, csv_rows = make_latex_table(df_all)

tex_path = Path(args.table_dir) / "efficiency_table.tex"
csv_path = Path(args.table_dir) / "efficiency_table.csv"

tex_path.write_text(tex)
print(f"  Wrote {tex_path}")

with open(csv_path, "w", newline="") as f:
    fields = ["Model", "Params (M)", "GFLOPs",
              "Latency (ms)", "Throughput (images/s)", "Peak Memory (MB)"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(csv_rows)
print(f"  Wrote {csv_path}")

print()
print(tex)
