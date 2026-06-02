#!/usr/bin/env python3
"""
Parse ShapeBiasNet evaluation results into CSVs.

Handles two file formats found under results_dir/seed-XX/:
  JSON (sweep output): {model}_{dataset}_results.json   ← preferred
  TXT  (legacy HPC):   {dataset}.txt                    ← fallback

JSON entries take priority — a (model, dataset, seed) triple already
parsed from JSON is never re-parsed from a .txt file, so mixing both
formats in the same seed directory is safe.

Usage:
    python parse_results.py                         # defaults
    python parse_results.py --results_dir /pvc/results --out_dir csvs

Outputs:
    csvs/summary.csv         model, dataset, seed, clean_acc, mCA, mCE
    csvs/per_corruption.csv  model, dataset, seed, corruption, s1–s5, mean
"""

import csv
import json
import re
import argparse
from pathlib import Path

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


# ── JSON parser (new sweep format) ───────────────────────────────────────────

def parse_json(path: Path, seed: int) -> dict:
    data = json.loads(path.read_text())
    return {
        "model":          data["model"],
        "dataset":        data["dataset"],
        "seed":           seed,
        "clean_acc":      data.get("clean_acc"),
        "mca":            data.get("mCA"),
        "mce":            data.get("mCE"),
        "per_corruption": data.get("per_corruption", {}),
    }


# ── TXT parser (legacy HPC format) ───────────────────────────────────────────

def parse_txt(path: Path, seed: int) -> list:
    lines = path.read_text().splitlines()

    model_starts = []
    seen_lines   = set()
    for i, line in enumerate(lines):
        if re.search(r"Model\s*:\s*\S", line):
            for j in range(i - 1, max(i - 10, -1), -1):
                if re.match(r"\s*={10,}\s*$", lines[j]) and j not in seen_lines:
                    model_starts.append(j)
                    seen_lines.add(j)
                    break
    model_starts.sort()

    records = []
    for k, start in enumerate(model_starts):
        end        = model_starts[k + 1] if k + 1 < len(model_starts) else len(lines)
        block      = lines[start:end]
        block_text = "\n".join(block)

        def _find(pattern):
            m = re.search(pattern, block_text)
            return m.group(1) if m else None

        model   = _find(r"Model\s*:\s*(\S+)")
        dataset = _find(r"Dataset\s*:\s*(\S+)")

        clean_acc = float(_find(r"Clean accuracy\s*:\s*([\d.]+)%") or "nan") or None
        mca       = float(_find(r"\bmCA\s*:\s*([\d.]+)%")           or "nan") or None
        mce       = float(_find(r"\bmCE\s*:\s*([\d.]+)%")           or "nan") or None

        per_corr = {}
        for line in block:
            m = re.match(
                r"\s*([a-z_]+)\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)",
                line,
            )
            if m:
                per_corr[m.group(1)] = {
                    "s1": float(m.group(2)), "s2": float(m.group(3)),
                    "s3": float(m.group(4)), "s4": float(m.group(5)),
                    "s5": float(m.group(6)), "mean": float(m.group(7)),
                }

        if model and dataset:
            records.append({
                "model": model, "dataset": dataset, "seed": seed,
                "clean_acc": clean_acc, "mca": mca, "mce": mce,
                "per_corruption": per_corr,
            })

    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir",     default="csvs")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    all_records = []
    seen        = set()   # (model, dataset, seed) already parsed from JSON

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        m = re.search(r"(\d+)", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))

        # ── JSON first (preferred) ────────────────────────────────────────
        for jf in sorted(seed_dir.glob("*_results.json")):
            try:
                r   = parse_json(jf, seed)
                key = (r["model"], r["dataset"], seed)
                if key in seen:
                    continue
                seen.add(key)
                all_records.append(r)
                status = (
                    f"clean={r['clean_acc']}  mCA={r['mca']}  mCE={r['mce']}"
                    if r["clean_acc"] is not None else "WARNING: clean_acc missing"
                )
                print(f"  {jf.name:<55} {status}")
            except Exception as e:
                print(f"  [WARN] {jf}: {e}")

        # ── Legacy .txt fallback ──────────────────────────────────────────
        for tf in sorted(seed_dir.glob("*.txt")):
            try:
                records = parse_txt(tf, seed)
                added   = 0
                for r in records:
                    key = (r["model"], r["dataset"], seed)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_records.append(r)
                    added += 1
                if added:
                    print(f"  {tf.name:<55} {added} model(s) from legacy txt")
            except Exception as e:
                print(f"  [WARN] {tf}: {e}")

    if not all_records:
        print("No records found. Check --results_dir.")
        return

    # ── Write summary.csv ─────────────────────────────────────────────────
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "seed", "clean_acc", "mCA", "mCE"])
        for r in all_records:
            w.writerow([r["model"], r["dataset"], r["seed"],
                        r["clean_acc"], r["mca"], r["mce"]])

    # ── Write per_corruption.csv ──────────────────────────────────────────
    corr_path = out_dir / "per_corruption.csv"
    with open(corr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "seed", "corruption",
                    "s1", "s2", "s3", "s4", "s5", "mean"])
        for r in all_records:
            for corr in CORRUPTIONS:
                v = r["per_corruption"].get(corr)
                if v:
                    w.writerow([r["model"], r["dataset"], r["seed"], corr,
                                v["s1"], v["s2"], v["s3"], v["s4"], v["s5"], v["mean"]])

    print(f"\nWrote {summary_path}  ({len(all_records)} model-dataset-seed rows)")
    print(f"Wrote {corr_path}")
    print(f"\nSeeds    : {sorted({r['seed']    for r in all_records})}")
    print(f"Models   : {sorted({r['model']   for r in all_records})}")
    print(f"Datasets : {sorted({r['dataset'] for r in all_records})}")


if __name__ == "__main__":
    main()
