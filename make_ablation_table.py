#!/usr/bin/env python3
"""
make_ablation_table.py — Parse ablation sweep results and generate LaTeX + CSV.

Reads JSON result files from --results_dir/seed-{N}/ subdirectories,
computes mean ± std across seeds, and outputs:
  - csvs/ablation_stats.csv     raw stats per model
  - tables/ablation_table.tex   IEEE-format LaTeX table
  - tables/ablation_table.csv   human-readable CSV table

Usage:
    python make_ablation_table.py --results_dir /pvc/results/ablation-sweep
    python make_ablation_table.py --results_dir results/ablation-sweep \
                                  --out_dir tables --csv_dir csvs
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

# ── Model definitions ──────────────────────────────────────────────────────────
# (key, display_name, group)
ABLATION_MODELS = [
    ("baseline_res18",
        "ResNet-18 (baseline)",                    "reference"),
    ("shape_res18",
        "Shape-ResNet-18 (full model)",             "reference"),
    ("shape_res18_early_gate",
        "Early gate ($r_1, r_2$) + late fusion",    "gating"),
    ("shape_res18_late_gate",
        "Late gate ($r_3$) + late fusion",          "gating"),
    ("shape_res18_gate_only",
        "Gate only, no fusion",                     "gating"),
    ("shape_res18_early_gate_nofuse",
        "Early gate, no fusion",                    "gating"),
    ("shape_res18_early_fuse",
        "Early fusion ($s_2 \\to r_2$)",            "fusion"),
    ("shape_res18_early_fuse_early_gate",
        "Early fusion + early gate",                "fusion"),
    ("shape_res18_early_fuse_late_gate",
        "Early fusion + late gate",                 "fusion"),
]

DATASET = "cifar10"


# ── Parsing ────────────────────────────────────────────────────────────────────

def load_records(results_dir: Path) -> list:
    """Walk seed-{N}/ subdirs, parse all *_results.json files."""
    records = []
    seen    = set()

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        m = re.search(r"(\d+)", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))

        for jf in sorted(seed_dir.glob("*_results.json")):
            try:
                data = json.loads(jf.read_text())
                key  = (data["model"], data.get("dataset", DATASET), seed)
                if key in seen:
                    continue
                seen.add(key)
                records.append({
                    "model":     data["model"],
                    "dataset":   data.get("dataset", DATASET),
                    "seed":      seed,
                    "clean_acc": data.get("clean_acc"),
                    "mca":       data.get("mCA"),
                    "mce":       data.get("mCE"),
                })
                print(f"  {jf.name:<55} "
                      f"clean={data.get('clean_acc')}  "
                      f"mCA={data.get('mCA')}")
            except Exception as e:
                print(f"  [WARN] {jf}: {e}")

    return records


# ── Statistics ─────────────────────────────────────────────────────────────────

def mean_sd(values: list) -> tuple:
    """Returns (mean, sd). sd is None when fewer than 2 values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = round(sum(vals) / len(vals), 2)
    sd = round(statistics.stdev(vals), 2) if len(vals) > 1 else None
    return m, sd


def compute_stats(records: list) -> dict:
    """Returns {model_key: {clean_acc, mca, mce, n_seeds}}."""
    grouped = {}
    for r in records:
        if r["dataset"] != DATASET:
            continue
        grouped.setdefault(r["model"], []).append(r)

    return {
        model: {
            "clean_acc": mean_sd([r["clean_acc"] for r in recs]),
            "mca":       mean_sd([r["mca"]       for r in recs]),
            "mce":       mean_sd([r["mce"]       for r in recs]),
            "n_seeds":   len(recs),
        }
        for model, recs in grouped.items()
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_latex(mean, sd, bold=False):
    if mean is None:
        return "--"
    cell = f"{mean:.2f}" if sd is None else f"{mean:.2f}$\\pm${sd:.2f}"
    return f"\\textbf{{{cell}}}" if bold else cell


def fmt_plain(mean, sd):
    if mean is None:
        return "--"
    return f"{mean:.2f}" if sd is None else f"{mean:.2f}±{sd:.2f}"


def best_in_group(group_keys, stats, metric, higher_is_better=True):
    candidates = {k: stats[k][metric][0] for k in group_keys
                  if k in stats and stats[k][metric][0] is not None}
    if not candidates:
        return None
    return max(candidates, key=candidates.get) if higher_is_better \
           else min(candidates, key=candidates.get)


# ── LaTeX table ────────────────────────────────────────────────────────────────

def make_latex(stats: dict) -> str:
    lines = [
        r"\begin{table*}[htbp]",
        r"\caption{Ablation study on {CIFAR}-10-{C} using a {ResNet}-18 backbone "
        r"trained on clean data only. Bold indicates best performance within each "
        r"variant group. Results show mean\,$\pm$\,std across three seeds.}",
        r"\begin{center}",
        r"\begin{tabular}{|l|c|c|c|}",
        r"\hline",
        r"\textbf{Variant} & \textbf{Clean Acc.} "
        r"& \textbf{mCA $\uparrow$} & \textbf{mCE $\downarrow$} \\",
        r"\hline",
    ]

    groups   = ["reference", "gating", "fusion"]
    n_groups = len(groups)

    for g_idx, group in enumerate(groups):
        group_keys = [k for k, _, grp in ABLATION_MODELS if grp == group]

        best_clean = best_in_group(group_keys, stats, "clean_acc", higher_is_better=True)
        best_mca   = best_in_group(group_keys, stats, "mca",       higher_is_better=True)
        best_mce   = best_in_group(group_keys, stats, "mce",       higher_is_better=False)

        for key, disp, grp in ABLATION_MODELS:
            if grp != group:
                continue
            if key not in stats:
                lines.append(f"{disp} & -- & -- & -- \\\\")
                continue
            s = stats[key]
            clean_cell = fmt_latex(*s["clean_acc"], bold=(key == best_clean))
            mca_cell   = fmt_latex(*s["mca"],       bold=(key == best_mca))
            mce_cell   = fmt_latex(*s["mce"],       bold=(key == best_mce))
            lines.append(
                f"{disp} & {clean_cell} & {mca_cell} & {mce_cell} \\\\"
            )

        if g_idx < n_groups - 1:
            lines.append(r"\hline")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\label{tab:ablation}",
        r"\end{center}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--results_dir", required=True,
                    help="Folder containing seed-{N}/ subdirectories")
    ap.add_argument("--out_dir",  default="tables",
                    help="Output directory for .tex and table .csv")
    ap.add_argument("--csv_dir",  default="csvs",
                    help="Output directory for stats .csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    csv_dir     = Path(args.csv_dir)
    out_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    print(f"\nLoading ablation results from: {results_dir}")
    records = load_records(results_dir)
    if not records:
        print("No records found. Check --results_dir.")
        return

    stats = compute_stats(records)
    print(f"\nModels found  : {sorted(stats.keys())}")
    print(f"Seeds per model: { {k: v['n_seeds'] for k, v in stats.items()} }\n")

    # ── Stats CSV ──────────────────────────────────────────────────────────────
    stats_path = csv_dir / "ablation_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "group", "n_seeds",
                           "clean_acc_mean", "clean_acc_sd",
                           "mca_mean", "mca_sd",
                           "mce_mean", "mce_sd"])
        writer.writeheader()
        for key, disp, group in ABLATION_MODELS:
            if key not in stats:
                continue
            s = stats[key]
            writer.writerow({
                "model":          key,
                "group":          group,
                "n_seeds":        s["n_seeds"],
                "clean_acc_mean": s["clean_acc"][0],
                "clean_acc_sd":   s["clean_acc"][1],
                "mca_mean":       s["mca"][0],
                "mca_sd":         s["mca"][1],
                "mce_mean":       s["mce"][0],
                "mce_sd":         s["mce"][1],
            })
    print(f"Wrote {stats_path}")

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    tex      = make_latex(stats)
    tex_path = out_dir / "ablation_table.tex"
    tex_path.write_text(tex)
    print(f"Wrote {tex_path}")
    print(f"\n{tex}\n")

    # ── Table CSV ──────────────────────────────────────────────────────────────
    table_path = out_dir / "ablation_table.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Variant", "Group", "Clean Acc.", "mCA", "mCE", "Seeds"])
        writer.writeheader()
        for key, disp, group in ABLATION_MODELS:
            if key not in stats:
                continue
            s = stats[key]
            writer.writerow({
                "Variant":    disp,
                "Group":      group,
                "Clean Acc.": fmt_plain(*s["clean_acc"]),
                "mCA":        fmt_plain(*s["mca"]),
                "mCE":        fmt_plain(*s["mce"]),
                "Seeds":      s["n_seeds"],
            })
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
