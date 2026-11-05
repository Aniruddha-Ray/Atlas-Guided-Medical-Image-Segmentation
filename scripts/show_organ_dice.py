"""
Per-organ validation Dice of a run with organ names, read from its metrics.csv (read-only: safe on a live run).

    python scripts/show_organ_dice.py chunk_150_50_e100_swin_prior            # latest epoch and best epoch
    python scripts/show_organ_dice.py chunk_150_50_e100_swin_prior --epoch 48
    python scripts/show_organ_dice.py chunk_150_50_e100_swin_prior --csv      # also writes metrics_named.csv next to metrics.csv

Names follow the AMOS ordering used in experiments/registration_log.md (not confirmed against the dataset documentation).
"""
from __future__ import annotations

import argparse
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["spleen", "r_kidney", "l_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "ivc", "pancreas",
         "r_adrenal", "l_adrenal"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_name")
    p.add_argument("--experiment", default="exp_04_pipeline")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--csv", action="store_true", help="write metrics_named.csv (columns o1(spleen) ... o12(l_adrenal))")
    a = p.parse_args()
    path = os.path.join(REPO, "experiments", a.experiment, a.run_name, "metrics.csv")
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("val_dice")]
    cols = [f"o{k}" for k in range(1, 13)]
    renamed = {c: f"{c}({n})" for c, n in zip(cols, NAMES)}
    if a.csv:
        out = os.path.join(os.path.dirname(path), "metrics_named.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([renamed.get(h, h) for h in rows[0].keys()])
            for r in rows:
                w.writerow(r.values())
        print(f"wrote {out}")
    shown = [("epoch " + str(a.epoch), next(r for r in rows if int(r["epoch"]) == a.epoch))] if a.epoch else \
        [("latest", rows[-1]), ("best", max(rows, key=lambda r: float(r["val_dice"])))]
    print(f"{'organ':12s}" + "".join(f"{f'{t} (ep {r['epoch']})':>20s}" for t, r in shown))
    for c, n in zip(cols, NAMES):
        print(f"{c + ' ' + n:12s}" + "".join(f"{float(r[c]):20.3f}" for _, r in shown))
    print(f"{'mean':12s}" + "".join(f"{float(r['val_dice']):20.3f}" for _, r in shown))


if __name__ == "__main__":
    main()
