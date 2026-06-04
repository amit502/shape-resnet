#!/usr/bin/env python3
"""
Generate per-corruption LaTeX table and CSV from per_corruption.csv.

One table per (dataset) — e.g. CIFAR-10-C, CIFAR-100-C, ImageNet-100-C.
Columns are the models; rows are the 15 corruption types.
Bold = best within each backbone group for that corruption row.

Usage:
    python make_corruption_table.py
    python make_corruption_table.py --corr_csv csvs/per_corruption.csv --out_dir tables/
"""

import argparse
import csv
import pandas as pd
from pathlib import Path

# ── Corruption display names (preserves original order) ─────────
CORRUPTION_DISPLAY = {
    "gaussian_noise":   "Gaussian noise",
    "shot_noise":       "Shot noise",
    "impulse_noise":    "Impulse noise",
    "defocus_blur":     "Defocus blur",
    "glass_blur":       "Glass blur",
    "motion_blur":      "Motion blur",
    "zoom_blur":        "Zoom blur",
    "snow":             "Snow",
    "frost":            "Frost",
    "fog":              "Fog",
    "brightness":       "Brightness",
    "contrast":         "Contrast",
    "elastic_transform": "Elastic",
    "pixelate":         "Pixelate",
    "jpeg_compression": "JPEG",
}
CORRUPTION_ORDER = list(CORRUPTION_DISPLAY.keys())

# ── Column groups — edit to add/remove models ────────────────────
# Each entry: (model_key, display_name, backbone_group)
# Columns with no data for a given dataset are dropped automatically.
COLUMN_GROUPS = [
    [
        ("baseline_res18",  "ResNet-18",        "res18"),
        ("shape_res18",     "Shape-ResNet-18",  "res18"),
    ],
    [
        ("baseline_res50",  "ResNet-50",        "res50"),
        ("shape_res50",     "Shape-ResNet-50",  "res50"),
    ],
    [
        ("baseline_res101", "ResNet-101",       "res101"),
        ("shape_res101",    "Shape-ResNet-101", "res101"),
    ],
]

DATASET_DISPLAY = {
    "cifar10":     "CIFAR-10",
    "cifar100":    "CIFAR-100",
    "imagenet100": "ImageNet-100",
    "imagenet":    "ImageNet",
}

DATASET_ORDER = ["cifar10", "cifar100", "imagenet100", "imagenet"]


def fmt(mean, sd, bold=False):
    cell = f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f}$\\pm${sd:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def fmt_plain(mean, sd):
    return f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f}±{sd:.2f}"


def make_table(dataset: str, stats: pd.DataFrame) -> tuple:
    """Returns (latex_str, csv_rows) for one dataset."""

    # Only keep models that have at least one data point for this dataset
    models_with_data = set(stats["model"].unique())
    active_groups = [
        [col for col in grp if col[0] in models_with_data]
        for grp in COLUMN_GROUPS
    ]
    active_groups = [grp for grp in active_groups if grp]  # drop empty groups

    all_cols  = [col for grp in active_groups for col in grp]
    col_names = [c[1] for c in all_cols]

    col_fmt = "|l|" + "c|" * len(all_cols)

    # IEEE-style two-row header: \multicolumn group names + \cline + sub-header row
    clines             = []
    group_header_cells = []
    sub_header_cells   = []
    pos = 2
    for grp in active_groups:
        n   = len(grp)
        end = pos + n - 1
        group_header_cells.append(f"\\multicolumn{{{n}}}{{c|}}{{{grp[0][1]}}}")
        for key, _, _ in grp:
            if key.startswith("baseline_"):
                sub_header_cells.append(r"\textbf{\textit{Baseline}}")
            else:
                sub_header_cells.append(r"\textbf{\textit{Shape}}")
        clines.append(f"\\cline{{{pos}-{end}}}")
        pos = end + 1

    group_header_row = " & ".join([r"\textbf{Corruption}"] + group_header_cells)
    sub_header_row   = " & ".join([""] + sub_header_cells)

    dataset_disp = DATASET_DISPLAY.get(dataset, dataset)
    label        = f"tab:{dataset}_corruptions"
    caption      = (
        f"{dataset_disp}-C severity-averaged corruption accuracies (\\%). "
        f"Bold indicates best accuracy within each backbone group per corruption type."
    )

    lines = [
        r"\begin{table*}[htbp]",
        f"\\caption{{{caption}}}",
        r"\begin{center}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\hline",
        f"{group_header_row} \\\\",
        " ".join(clines),
        f"{sub_header_row} \\\\",
        r"\hline",
    ]

    csv_rows = []

    for corr_key in CORRUPTION_ORDER:
        subset = stats[stats["corruption"] == corr_key]
        if subset.empty:
            continue

        # model_key → (mean, sd)
        cell_data = {}
        for _, r in subset.iterrows():
            cell_data[r["model"]] = (r["mean_mean"], r["mean_sd"])

        # bold = best (highest) within each backbone group
        bold_set = set()
        for grp in active_groups:
            grp_keys  = [c[0] for c in grp]
            available = {k: cell_data[k][0] for k in grp_keys if k in cell_data}
            if available:
                bold_set.add(max(available, key=available.get))

        cells     = []
        csv_cells = {}
        for key, name, _ in all_cols:
            if key in cell_data:
                m, s = cell_data[key]
                cells.append(fmt(m, s, key in bold_set))
                csv_cells[name] = fmt_plain(m, s)
            else:
                cells.append("--")
                csv_cells[name] = "--"

        corr_disp = CORRUPTION_DISPLAY.get(corr_key, corr_key)
        lines.append(f"\\textit{{{corr_disp}}} & {' & '.join(cells)} \\\\")
        csv_rows.append({"Corruption": corr_disp, **csv_cells})

    lines += [r"\hline", r"\end{tabular}", f"\\label{{{label}}}", r"\end{center}", r"\end{table*}"]
    return "\n".join(lines), csv_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_csv", default="csvs/per_corruption.csv")
    ap.add_argument("--out_dir",  default="tables")
    args = ap.parse_args()

    df  = pd.read_csv(args.corr_csv)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    # mean ± SD of per-corruption accuracy across seeds; NaN SD = single seed
    stats = (
        df.groupby(["model", "dataset", "corruption"])["mean"]
        .agg(mean_mean="mean", mean_sd="std")
        .round(2)
        .reset_index()
    )

    for dataset in DATASET_ORDER:
        ds_stats = stats[stats["dataset"] == dataset]
        if ds_stats.empty:
            continue

        tex, csv_rows = make_table(dataset, ds_stats)

        tex_path = out / f"corruption_table_{dataset}.tex"
        csv_path = out / f"corruption_table_{dataset}.csv"

        tex_path.write_text(tex)

        col_names = [c[1] for grp in COLUMN_GROUPS for c in grp]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Corruption"] + col_names)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Wrote {tex_path}")
        print(f"Wrote {csv_path}\n")
        print(tex)
        print()


if __name__ == "__main__":
    main()
