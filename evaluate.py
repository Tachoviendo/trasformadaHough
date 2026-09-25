"""Compare automated Hough counts against manual counts (Section 2.6 / 4 of the paper).

Input: results/counts.csv (from colony_counter.py) and the manual counts, either from a CSV
with columns `image,manual_count` (--manual) or parsed from the file names of the
Rodrigues et al. dataset, e.g. `IMG_7710ecoli_T4_10^-7_59.JPG` -> 59. Plates marked `300`
(too numerous to count) or flagged for recount / spoiled are excluded.

Count-level metrics per species, treating the manual count as ground truth:
    TP = min(auto, manual), FP = max(0, auto - manual), FN = max(0, manual - auto)
    accuracy = TP / (TP + FP + FN), recall = TP / (TP + FN), F1 = 2PR / (P + R)
    error(x) = C(x) * (FPR(x) + FNR(x))  (eq. 4)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXCLUDE = ("contar de novo", "estragad")
COUNT_RE = re.compile(r"10\^-?\d+_(_?)(\d+)", re.I)


def manual_from_filenames(images: pd.Series) -> pd.DataFrame:
    rows = []
    for img in images:
        name = Path(img).name.lower()
        m = COUNT_RE.search(name)
        # "300" is the dataset's too-numerous-to-count cap, with one or two underscores
        if m and not m.group(1) and m.group(2) != "300" and not any(k in name for k in EXCLUDE):
            rows.append({"image": img, "manual_count": int(m.group(2))})
    return pd.DataFrame(rows, columns=["image", "manual_count"])


def metrics(df: pd.DataFrame) -> dict:
    tp = np.minimum(df.auto_count, df.manual_count).sum()
    fp = np.clip(df.auto_count - df.manual_count, 0, None).sum()
    fn = np.clip(df.manual_count - df.auto_count, 0, None).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    rel = (df.auto_count - df.manual_count).abs() / df.manual_count.replace(0, np.nan)
    return {
        "n_images": len(df),
        "accuracy": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_abs_diff_pct": 100 * rel.mean(),
        "over_pct": 100 * (df.auto_count > df.manual_count).mean(),
        "under_pct": 100 * (df.auto_count < df.manual_count).mean(),
        "pearson_r": df.auto_count.corr(df.manual_count) if len(df) > 1 else np.nan,
        "mean_seconds": df.seconds.mean(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto", type=Path, default=Path("results/counts.csv"))
    ap.add_argument("--manual", type=Path, help="CSV image,manual_count (default: parse file names)")
    ap.add_argument("--max-count", type=int, default=None,
                    help="only evaluate plates with manual count <= this (paper's table uses 0-200)")
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    auto = pd.read_csv(args.auto)
    manual = pd.read_csv(args.manual) if args.manual else manual_from_filenames(auto.image)
    if args.max_count is not None:
        manual = manual[manual.manual_count <= args.max_count]
    df = auto.merge(manual[["image", "manual_count"]], on="image")
    if df.empty:
        raise SystemExit("no images in common between the two CSVs")
    df["species"] = df.species.fillna("unknown")
    df["error"] = (df.auto_count - df.manual_count).abs()
    df.to_csv(args.out / "comparison.csv", index=False)

    groups = {sp: g for sp, g in df.groupby("species")}
    table = pd.DataFrame({sp: metrics(g) for sp, g in groups.items()} | {"all": metrics(df)}).T
    table.to_csv(args.out / "metrics.csv")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table)

    n = len(groups)
    # Fig 4: manual vs automated counts and error per plate, sorted by manual count
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)
    for i, (sp, g) in enumerate(groups.items()):
        g = g.sort_values("manual_count").reset_index(drop=True)
        ax = axes[i, 0]
        ax.plot(g.manual_count, label="manual")
        ax.plot(g.auto_count, label="Hough")
        ax.plot(g.error, label="|error|", alpha=0.6)
        ax.set_title(f"{sp}: counts per plate")
        ax.legend()
        axes[i, 1].plot(g.seconds * 1000)
        axes[i, 1].set_title(f"{sp}: processing time (ms)")
    fig.tight_layout()
    fig.savefig(args.out / "fig4_counts_error_time.png", dpi=120)

    # Fig 5: histogram of automated counts
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.5), squeeze=False)
    for ax, (sp, g) in zip(axes[0], groups.items()):
        ax.hist(g.auto_count, bins=20)
        ax.set_title(f"{sp}: Hough colony counts")
    fig.tight_layout()
    fig.savefig(args.out / "fig5_histograms.png", dpi=120)

    # Fig 6: Pearson correlation (scatter with r per species)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, (sp, g) in zip(axes[0], groups.items()):
        ax.scatter(g.manual_count, g.auto_count, s=10)
        m = max(g.manual_count.max(), g.auto_count.max())
        ax.plot([0, m], [0, m], "k--", lw=1)
        ax.set(xlabel="manual", ylabel="Hough", title=f"{sp}  r={g.auto_count.corr(g.manual_count):.2f}")
    fig.tight_layout()
    fig.savefig(args.out / "fig6_pearson.png", dpi=120)
    print(f"\nwrote {args.out}/metrics.csv, comparison.csv and fig4/5/6 PNGs")


if __name__ == "__main__":
    main()
