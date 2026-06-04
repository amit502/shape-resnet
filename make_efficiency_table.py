#!/usr/bin/env python3
"""
make_efficiency_table.py — Parse profiler CSV and generate efficiency table.

Reads the latest profile_cifar10_*.csv from --profile_dir (or an explicit
--profile_csv path), and outputs:
  - tables/efficiency_table.tex  (IEEE-format LaTeX)
  - tables/efficiency_table.csv  (human-readable)

Usage:
    python make_efficiency_table.py --profile_dir /pvc/results/profile
    python make_efficiency_table.py --profile_csv /pvc/results/profile/profile_cifar10_20260602_111214.csv
"""

import argparse
import csv
import glob
import os
from pathlib import Path

# ── Model config ───────────────────────────────────────────────────────────────
# Backbone pairs: (baseline_key, shape_key, display_baseline, display_shape)
BACKBONE_PAIRS = [
    ("baseline_res18", "shape_res18",
     "ResNet-18",       "Shape-ResNet-18"),
    ("baseline_res50", "shape_res50",
     "ResNet-50",       "Shape-ResNet-50"),
]

# Metrics: (csv_column, display_label, higher_is_better)
METRICS = [
    ("params_total_M",   "Params (M)",         False),
    ("gflops",           "GFLOPs",             False),
    ("latency_mean_ms",  "Latency (ms)",        False),
    ("throughput_img_s", "Throughput (images/s)", True),
    ("peak_mem_mb",      "Peak Memory (MB)",    False),
]


# ── Load CSV ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict:
    """Returns {model_name: {col: value}} from the profiler CSV."""
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["model"]] = {k: float(v) for k, v in row.items()
                                  if k not in ("model", "dataset")}
    return data


def find_latest_csv(profile_dir: str) -> str:
    matches = sorted(
        glob.glob(os.path.join(profile_dir, "profile_cifar10_*.csv")),
        key=os.path.getmtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No profile_cifar10_*.csv found in '{profile_dir}'")
    print(f"  Using: {matches[-1]}")
    return matches[-1]


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(val: float, bold: bool = False) -> str:
    s = f"{val:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_plain(val: float) -> str:
    return f"{val:.2f}"


def is_better(a: float, b: float, higher_is_better: bool) -> bool:
    return a > b if higher_is_better else a < b


# ── LaTeX table ────────────────────────────────────────────────────────────────

def make_latex(data: dict) -> str:
    n_metrics  = len(METRICS)
    col_spec   = "|l|" + "c|" * n_metrics

    # Rotated column headers
    metric_headers = " & ".join(
        f"\\rotatebox{{90}}{{\\textbf{{{col[1]}}}}}"
        for col in METRICS
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Efficiency comparison of baseline and shape-biased models "
        r"on {CIFAR}-10 (32$\times$32 input, batch size 1). "
        r"Bold indicates the more efficient value within each backbone pair.}",
        r"\begin{center}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        f"\\textbf{{Model}} & {metric_headers} \\\\",
        r"\hline",
    ]

    csv_rows = []
    n_pairs  = len(BACKBONE_PAIRS)

    for p_idx, (base_key, shape_key, base_disp, shape_disp) in \
            enumerate(BACKBONE_PAIRS):

        if base_key not in data or shape_key not in data:
            print(f"  [WARN] Missing data for pair "
                  f"{base_key}/{shape_key} — skipping.")
            continue

        b = data[base_key]
        s = data[shape_key]

        for model_key, disp in ((base_key, base_disp), (shape_key, shape_disp)):
            row_data = data[model_key]
            partner  = data[shape_key if model_key == base_key else base_key]

            cells     = []
            csv_cells = {"Model": disp}

            for col, label, higher in METRICS:
                val     = row_data[col]
                par_val = partner[col]
                bold    = is_better(val, par_val, higher)
                cells.append(fmt(val, bold))
                csv_cells[label] = fmt_plain(val)

            lines.append(f"{disp} & {' & '.join(cells)} \\\\")
            csv_rows.append(csv_cells)

        if p_idx < n_pairs - 1:
            lines.append(r"\hline")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\label{tab:efficiency}",
        r"\end{center}",
        r"\end{table}",
    ]

    return "\n".join(lines), csv_rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--profile_dir", default=None,
                     help="Directory to search for profile_cifar10_*.csv "
                          "(latest file is used)")
    src.add_argument("--profile_csv", default=None,
                     help="Explicit path to profiler CSV file")

    ap.add_argument("--out_dir", default="tables",
                    help="Output directory for .tex and .csv")
    args = ap.parse_args()

    csv_path = (args.profile_csv if args.profile_csv
                else find_latest_csv(args.profile_dir))

    print(f"\nLoading profiler data from: {csv_path}")
    data = load_csv(csv_path)
    print(f"Models found: {list(data.keys())}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    tex, csv_rows = make_latex(data)

    # LaTeX
    tex_path = out_dir / "efficiency_table.tex"
    tex_path.write_text(tex)
    print(f"\nWrote {tex_path}")
    print(f"\n{tex}\n")

    # CSV
    fieldnames = ["Model"] + [m[1] for m in METRICS]
    csv_out    = out_dir / "efficiency_table.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
