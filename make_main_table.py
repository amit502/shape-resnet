#!/usr/bin/env python3
"""
Generate LaTeX for the main clean/corruption summary table from summary_stats.csv.

Usage:
    python make_main_table.py
    python make_main_table.py --stats_csv csvs/summary_stats.csv --out tables/main_table.tex
"""

import argparse
import csv
import pandas as pd
from pathlib import Path


def fmt(mean, sd, bold=False):
    cell = f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f}$\\pm${sd:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def fmt_plain(mean, sd):
    return f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f}±{sd:.2f}"


def make_table(df: pd.DataFrame) -> tuple:
    """Returns (latex_str, table_rows) where table_rows is a list of dicts for CSV."""
    lines = []
    rows  = []

    lines += [
        r"\begin{table*}[t]",
        r"\caption{Clean accuracy and corruption robustness comparison. "
        r"Bold indicates best performance within each backbone group. "
        r"Results show mean\,$\pm$\,std across seeds where multiple runs are available.}",
        r"\label{tab:main}",
        r"\vspace{6pt}",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Dataset} & \textbf{Clean Acc.} "
        r"& \textbf{mCA $\uparrow$} & \textbf{mCE $\downarrow$} \\",
        r"\midrule",
    ]

    groups = list(df.groupby(["dataset_display", "backbone_group"], sort=False))
    for i, ((dataset_disp, group), gdf) in enumerate(groups):
        best_clean = gdf["clean_acc_mean"].idxmax()
        best_mca   = gdf["mCA_mean"].idxmax()
        best_mce   = gdf["mCE_mean"].idxmin()

        for idx, row in gdf.iterrows():
            clean_cell = fmt(row["clean_acc_mean"], row["clean_acc_sd"], idx == best_clean)
            mca_cell   = fmt(row["mCA_mean"],       row["mCA_sd"],       idx == best_mca)
            mce_cell   = fmt(row["mCE_mean"],       row["mCE_sd"],       idx == best_mce)

            lines.append(
                f"{row['model_display']} & {row['dataset_display']} & "
                f"{clean_cell} & {mca_cell} & {mce_cell} \\\\"
            )
            rows.append({
                "Model":       row["model_display"],
                "Dataset":     row["dataset_display"],
                "Clean Acc.":  fmt_plain(row["clean_acc_mean"], row["clean_acc_sd"]),
                "mCA":         fmt_plain(row["mCA_mean"],       row["mCA_sd"]),
                "mCE":         fmt_plain(row["mCE_mean"],       row["mCE_sd"]),
            })

        if i < len(groups) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    return "\n".join(lines), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_csv", default="csvs/summary_stats.csv")
    ap.add_argument("--out",       default="tables/main_table.tex")
    ap.add_argument("--csv_out",   default="tables/main_table.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.stats_csv)

    tex, rows = make_table(df)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(tex)

    csv_out = Path(args.csv_out)
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Dataset", "Clean Acc.", "mCA", "mCE"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out}")
    print(f"Wrote {csv_out}\n")
    print(tex)


if __name__ == "__main__":
    main()
