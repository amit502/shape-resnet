#!/usr/bin/env python3
"""
Aggregate per-seed summary.csv into mean ± SD across seeds.
Output CSV is one row per (model, dataset), ready for LaTeX table generation.

Usage:
    python make_stats_csv.py
    python make_stats_csv.py --summary_csv csvs/summary.csv --out csvs/summary_stats.csv
"""

import argparse
import pandas as pd
from pathlib import Path

# ── Display names ────────────────────────────────────────────────
MODEL_DISPLAY = {
    "baseline_res18":           "ResNet-18",
    "baseline_res34":           "ResNet-34",
    "baseline_res50":           "ResNet-50",
    "baseline_res101":          "ResNet-101",
    "baseline_convnext_tiny":   "ConvNeXt-Tiny",
    "baseline_convnext_base":   "ConvNeXt-Base",
    "baseline_efficientnet_b0": "EfficientNet-B0",
    "baseline_efficientnet_b4": "EfficientNet-B4",
    "shape_custom":             "Shape-Custom",
    "shape_res18":              "Shape-ResNet18",
    "shape_res34":              "Shape-ResNet34",
    "shape_res50":              "Shape-ResNet50",
    "shape_res101":             "Shape-ResNet101",
    "shape_convnext_tiny":      "Shape-ConvNeXt-Tiny",
    "shape_convnext_base":      "Shape-ConvNeXt-Base",
    "shape_effnet_b0":          "Shape-EfficientNet-B0",
    "shape_effnet_b4":          "Shape-EfficientNet-B4",
}

DATASET_DISPLAY = {
    "cifar10":    "CIFAR-10",
    "cifar100":   "CIFAR-100",
    "imagenet100": "ImageNet-100",
    "imagenet":   "ImageNet",
}

# ── Grouping (drives \hline structure in LaTeX) ──────────────────
# Each group gets a horizontal rule after it in the table.
BACKBONE_GROUP = {
    "baseline_res18":           "res18",
    "shape_custom":             "res18",
    "shape_res18":              "res18",
    "baseline_res34":           "res34",
    "shape_res34":              "res34",
    "baseline_res50":           "res50",
    "shape_res50":              "res50",
    "baseline_res101":          "res101",
    "shape_res101":             "res101",
    "baseline_convnext_tiny":   "convnext_tiny",
    "shape_convnext_tiny":      "convnext_tiny",
    "baseline_convnext_base":   "convnext_base",
    "shape_convnext_base":      "convnext_base",
    "baseline_efficientnet_b0": "effnet_b0",
    "shape_effnet_b0":          "effnet_b0",
    "baseline_efficientnet_b4": "effnet_b4",
    "shape_effnet_b4":          "effnet_b4",
}

GROUP_ORDER = {
    "res18": 0, "res34": 1, "res50": 2, "res101": 3,
    "convnext_tiny": 4, "convnext_base": 5,
    "effnet_b0": 6, "effnet_b4": 7,
}

# Row order within a backbone group
ROW_ORDER = {
    "baseline_res18": 0,  "shape_custom": 1,  "shape_res18": 2,
    "baseline_res34": 0,  "shape_res34": 1,
    "baseline_res50": 0,  "shape_res50": 1,
    "baseline_res101": 0, "shape_res101": 1,
    "baseline_convnext_tiny": 0,   "shape_convnext_tiny": 1,
    "baseline_convnext_base": 0,   "shape_convnext_base": 1,
    "baseline_efficientnet_b0": 0, "shape_effnet_b0": 1,
    "baseline_efficientnet_b4": 0, "shape_effnet_b4": 1,
}

DATASET_ORDER = {
    "cifar10": 0, "cifar100": 1, "imagenet100": 2, "imagenet": 3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="csvs/summary.csv")
    ap.add_argument("--out",         default="csvs/summary_stats.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    df = df[df["model"] != "shape_custom"]

    # Compute mean and SD per (model, dataset) across seeds
    grp = df.groupby(["model", "dataset"])[["clean_acc", "mCA", "mCE"]]
    mean = grp.mean().round(2)
    sd   = grp.std(ddof=1).round(2)   # NaN when only 1 seed — table scripts detect this

    stats = mean.rename(columns={
        "clean_acc": "clean_acc_mean",
        "mCA":       "mCA_mean",
        "mCE":       "mCE_mean",
    }).join(sd.rename(columns={
        "clean_acc": "clean_acc_sd",
        "mCA":       "mCA_sd",
        "mCE":       "mCE_sd",
    })).reset_index()

    # Attach display names and sort keys
    stats["model_display"]   = stats["model"].map(MODEL_DISPLAY).fillna(stats["model"])
    stats["dataset_display"] = stats["dataset"].map(DATASET_DISPLAY).fillna(stats["dataset"])
    stats["backbone_group"]  = stats["model"].map(BACKBONE_GROUP).fillna("other")
    stats["group_order"]     = stats["backbone_group"].map(GROUP_ORDER).fillna(99)
    stats["row_order"]       = stats["model"].map(ROW_ORDER).fillna(99)
    stats["dataset_order"]   = stats["dataset"].map(DATASET_ORDER).fillna(99)

    stats = stats.sort_values(["dataset_order", "group_order", "row_order"])

    out_cols = [
        "model_display", "dataset_display", "backbone_group",
        "clean_acc_mean", "clean_acc_sd",
        "mCA_mean",       "mCA_sd",
        "mCE_mean",       "mCE_sd",
    ]
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    stats[out_cols].to_csv(out, index=False)

    print(f"Seeds found : {sorted(df['seed'].unique())}")
    print(f"Models      : {sorted(df['model'].unique())}")
    print(f"Datasets    : {sorted(df['dataset'].unique())}")
    print(f"\nWrote {out}\n")
    print(stats[out_cols].to_string(index=False))


if __name__ == "__main__":
    main()
