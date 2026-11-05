"""
Case split of the cascade study (train on AMOS, test on BTCV).

  atlases   30 AMOS cases, the 30 most typical that pass scripts/select_quality_atlases.py's checks (same funnel as
            scripts/select_amos_atlases.py, on its cached case metrics; the 50 validation cases of that study were not
            scanned and are never atlases). They contain the 10 atlases of the first AMOS
            study (atlas/atlas_selection_amos.json), whose CT-to-CT registrations every other AMOS case already has.
  stage1    42 of the remaining 242 AMOS cases (seed --seed): trains the first-stage model (CT + 10-atlas CT-to-CT
            prior) that predicts the "labels" the 30 atlases are registered to. Spent on stage 1: never a stage-2 case.
  stage2    the other 200 AMOS cases: stage-2 training (image-only, plain cascade, atlas cascade).
  test      the 30 BTCV cases (Dataset062_Synapse), never trained on: nnU-Net's validation fold and the reported test.

Writes (refuses to overwrite without --force):
    atlas/atlas_selection_amos30.json            the 30 atlases, in atlas_selection.json's format
    experiments/exp_cascade/cascade_split.json   {"atlases", "atlases_stage1", "stage1", "stage2", "test", "seed"}

    python scripts/cascade_split.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.splits import case_name, paired_cases  # noqa: E402
from scripts.select_quality_atlases import NUM_ORGANS, apply_filters, load_rows, pick  # noqa: E402

DATA = os.path.dirname(REPO)
AMOS_DIR, BTCV_DIR = os.path.join(DATA, "Dataset061_AMOS"), os.path.join(DATA, "Dataset062_Synapse")
FILTERS = dict(max_slice_mm=3.0, max_inplane_mm=1.6, min_z_mm=300.0, min_lcc=0.90, vol_lo=0.3, vol_hi=3.0, kidney_ratio=2.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlases", type=int, default=30)
    p.add_argument("--stage1", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics-csv", default=os.path.join(REPO, "experiments", "exp_amos", "amos_case_metrics.csv"))
    p.add_argument("--old-selection", default=os.path.join(REPO, "atlas", "atlas_selection_amos.json"))
    p.add_argument("--out", default=os.path.join(REPO, "atlas", "atlas_selection_amos30.json"))
    p.add_argument("--split-out", default=os.path.join(REPO, "experiments", "exp_cascade", "cascade_split.json"))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    for path in (a.out, a.split_out):
        if os.path.exists(path) and not a.force:
            raise SystemExit(f"{path} exists; the split must stay fixed (use --force to replace)")

    survivors, funnel = apply_filters(load_rows(a.metrics_csv), SimpleNamespace(**FILTERS))
    chosen = pick(survivors, a.atlases, {"amos": a.atlases})
    if len(chosen) < a.atlases:
        raise SystemExit(f"only {len(chosen)} cases pass the atlas checks")
    atlases = sorted(r["name"] for r in chosen)
    old = sorted(case_name(c["image"]) for c in json.load(open(a.old_selection, encoding="utf-8"))["cases"])
    if not set(old) <= set(atlases):
        raise SystemExit("the 10 first-study atlases are not all among the new atlases; stage 1 needs their registrations")

    amos = {case_name(c["image"]): c for c in paired_cases(AMOS_DIR)}
    pool = sorted(n for n in amos if n not in set(atlases))
    random.Random(a.seed).shuffle(pool)
    stage1, stage2 = sorted(pool[: a.stage1]), sorted(pool[a.stage1:])
    test = sorted(case_name(c["image"]) for c in paired_cases(BTCV_DIR))
    print("atlas funnel:", ", ".join(f"{label}: {n}" for label, n in funnel))
    print(f"AMOS {len(amos)} = atlases {len(atlases)} + stage1 {len(stage1)} + stage2 {len(stage2)}; BTCV test {len(test)}")

    os.makedirs(os.path.dirname(a.split_out), exist_ok=True)
    json.dump({"atlases": atlases, "atlases_stage1": old, "stage1": stage1, "stage2": stage2, "test": test,
               "seed": a.seed}, open(a.split_out, "w", encoding="utf-8"), indent=1)
    json.dump({"atlas_count": len(atlases), "num_organs": NUM_ORGANS,
               "selection_criteria": (f"AMOS only, the {len(atlases)} most typical cases passing: all 12 organs, "
                                      f"slice<=3mm, in-plane<=1.6mm, z-extent>=300mm, largest component>=90%, organ "
                                      "volumes 0.3-3x pool median, kidney ratio<=2. Includes the 10 atlases of "
                                      "atlas_selection_amos.json."),
               "cases": [{"image": amos[n]["image"], "label": amos[n]["label"]} for n in atlases]},
              open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {a.out}\nwrote {a.split_out}")


if __name__ == "__main__":
    main()
