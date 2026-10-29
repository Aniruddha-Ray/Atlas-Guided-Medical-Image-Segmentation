"""
Validates registration and prior insertion for a set of cases, before any training.

For every case it loads the sample exactly as the trainer does (build_transforms, prior_source="atlas") and checks:

  registration  per-atlas statuses (fallbacks), RANSAC inliers, registration time, and the atlas-only Dice that
                scripts/register_targets.py --eval-gt stored on the 2 mm registration grid
  insertion     invariants of what the network will receive (shapes, finite, P sums to 1, P / D / w in [0, 1], D is
                zero on the atlas organ region it is defined from) and alignment of the priors with the CT/label:
                  - atlas-vote Dice on the 96^3 training grid (argmax P against the label; evaluation only)
                  - the same vote after the training augmentation (random flips/rotations): a misaligned prior
                    would lose Dice here
                  - median distance between vote and label organ centroids, in mm
                  - support recall: share of each true organ that lies where at least one atlas voted for it

The label is used only for these scores, never to build priors. Exit code 2 if an invariant fails (run.py stops
before training in that case), 0 otherwise. Writes <out-prefix>.json and <out-prefix>.md.

    python scripts/validate_priors.py --chunk experiments/exp_04_pipeline/chunk.json --out-prefix experiments/exp_04_pipeline/validation
    python scripts/validate_priors.py --cases amos_0296 s0339 --out-prefix experiments/validation_demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from data.splits import (DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, attach_atlas_priors, case_name, paired_cases,
                         train_val_split)
from data.transforms import _voxel_spacing, build_transforms

IMG_SIZE = (96, 96, 96)
NUM_CLASSES = 13
LOW_DICE = 0.25


def source_of(name: str) -> str:
    return "amos" if name.startswith("amos") else "img" if name.startswith("img") else "s"


def dice_per_organ(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = np.full(NUM_CLASSES - 1, np.nan)
    for c in range(1, NUM_CLASSES):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        if denom > 0:
            out[c - 1] = 2.0 * (p & g).sum() / denom
    return out


def check_invariants(prob: np.ndarray, dist: np.ndarray, conf: np.ndarray, relative_threshold: float = 0.5) -> list:
    """Returns a list of human-readable failures (empty = all good). prob (13, X, Y, Z), dist (12, ...), conf (1, ...)."""
    bad = []
    if prob.shape[0] != NUM_CLASSES or dist.shape[0] != NUM_CLASSES - 1 or conf.shape[0] != 1:
        bad.append(f"channel counts P/D/w = {prob.shape[0]}/{dist.shape[0]}/{conf.shape[0]}, expected 13/12/1")
        return bad
    if not (np.isfinite(prob).all() and np.isfinite(dist).all() and np.isfinite(conf).all()):
        bad.append("non-finite values in P, D or w")
        return bad
    if np.abs(prob.sum(0) - 1.0).max() > 1e-4:
        bad.append(f"P does not sum to 1 (max deviation {np.abs(prob.sum(0) - 1.0).max():.2e})")
    if prob.min() < -1e-6 or prob.max() > 1 + 1e-6:
        bad.append("P outside [0, 1]")
    if dist.min() < -1e-6 or dist.max() > 1 + 1e-6:
        bad.append("D outside [0, 1]")
    if conf.min() < -1e-6 or conf.max() > 1 + 1e-6:
        bad.append("w outside [0, 1]")
    for c in range(1, NUM_CLASSES):
        peak = prob[c].max()
        if peak <= 0:
            if dist[c - 1].min() < 1 - 1e-6:
                bad.append(f"organ {c}: no atlas votes it but D is not 1 everywhere")
            continue
        region = prob[c] >= relative_threshold * peak
        if dist[c - 1][region].max() > 1e-6:
            bad.append(f"organ {c}: D is not 0 on the atlas organ region it is defined from")
    return bad


def centroid_error_mm(vote: np.ndarray, gt: np.ndarray, spacing) -> float:
    errs = []
    for c in range(1, NUM_CLASSES):
        pv, gv = np.argwhere(vote == c), np.argwhere(gt == c)
        if len(pv) and len(gv):
            errs.append(float(np.linalg.norm((pv.mean(0) - gv.mean(0)) * np.asarray(spacing))))
    return float(np.median(errs)) if errs else float("nan")


def support_recall(prob: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = np.full(NUM_CLASSES - 1, np.nan)
    for c in range(1, NUM_CLASSES):
        g = gt == c
        if g.any():
            out[c - 1] = float((prob[c][g] > 0).mean())
    return out


def registration_record(case: str, cache_dir: str, variant: str) -> dict:
    p = os.path.join(cache_dir, "registrations", variant, f"{case}.json")
    if not os.path.exists(p):
        return {"registered": False}
    r = json.load(open(p, encoding="utf-8"))
    inl = [a.get("ransac", {}).get("num_inliers", 0) for a in r["atlases"]]
    ev = r.get("evaluation_only") or {}
    return {
        "registered": True,
        "status_counts": r["status_counts"],
        "inliers_median": float(np.median(inl)) if inl else 0.0,
        "elapsed_s": r.get("elapsed_s"),
        "vote_dice_2mm": ev.get("majority_vote_mean_dice"),
    }


def validate_case(item: dict, tf_val, tf_train, seed: int) -> dict:
    name = case_name(item["image"])
    d = tf_val(item)
    prob, dist, conf = (np.asarray(d[k]) for k in ("prob", "dist", "conf"))
    gt = np.asarray(d["label"]).argmax(0)
    spacing = _voxel_spacing(d["image"])
    vote = prob.argmax(0)
    row = {
        "case": name,
        "source": source_of(name),
        "invariant_failures": check_invariants(prob, dist, conf),
        "vote_dice": dice_per_organ(vote, gt).tolist(),
        "support_recall": support_recall(prob, gt).tolist(),
        "centroid_error_mm": centroid_error_mm(vote, gt, spacing),
        "w_mean": float(conf.mean()),
        "voxel_spacing_mm": [round(float(s), 3) for s in spacing],
    }
    row["vote_dice_mean"] = float(np.nanmean(row["vote_dice"]))
    tf_train.set_random_state(seed=seed)
    a = tf_train(item)
    aug_vote = np.asarray(a["prob"]).argmax(0)
    row["vote_dice_mean_augmented"] = float(np.nanmean(dice_per_organ(aug_vote, np.asarray(a["label"]).argmax(0))))
    return row


def summarise(rows: list) -> dict:
    out = {}
    for split in ("train", "val", "all"):
        rs = [r for r in rows if split == "all" or r["split"] == split]
        if not rs:
            continue
        d = np.array([r["vote_dice_mean"] for r in rs])
        da = np.array([r["vote_dice_mean_augmented"] for r in rs])
        out[split] = {
            "n": len(rs),
            "vote_dice_mean": float(d.mean()), "vote_dice_median": float(np.median(d)),
            "vote_dice_min": float(d.min()), "vote_dice_max": float(d.max()),
            "vote_dice_augmented_mean": float(da.mean()),
            "below_0.25": [r["case"] for r in rs if r["vote_dice_mean"] < LOW_DICE],
            "per_organ_vote_dice": np.nanmean([r["vote_dice"] for r in rs], axis=0).round(3).tolist(),
            "per_organ_support_recall": np.nanmean([r["support_recall"] for r in rs], axis=0).round(3).tolist(),
            "centroid_error_median_mm": float(np.nanmedian([r["centroid_error_mm"] for r in rs])),
            "by_source": {s: {"n": int(sum(r["source"] == s for r in rs)),
                              "vote_dice_mean": float(np.mean([r["vote_dice_mean"] for r in rs if r["source"] == s]))}
                          for s in sorted({r["source"] for r in rs})},
            "fallback_atlases": int(sum(r["registration"].get("status_counts", {}).get("coarse_affine_only", 0) for r in rs)),
            "targets_with_3plus_fallbacks": [r["case"] for r in rs
                                             if r["registration"].get("status_counts", {}).get("coarse_affine_only", 0) >= 3],
        }
    return out


def to_markdown(rows: list, summary: dict, failures: list) -> str:
    lines = ["# Registration and prior-insertion validation", ""]
    lines.append("Atlas-vote Dice is the overlap of argmax(P) with the label on the 96^3 training grid (evaluation only).")
    lines.append("")
    lines.append("| Split | Cases | Vote Dice mean / median | Min / max | After augmentation | Centroid error (median) | Fallback atlases |")
    lines.append("|---|---|---|---|---|---|---|")
    for split, s in summary.items():
        lines.append(f"| {split} | {s['n']} | {s['vote_dice_mean']:.3f} / {s['vote_dice_median']:.3f} | "
                     f"{s['vote_dice_min']:.3f} / {s['vote_dice_max']:.3f} | {s['vote_dice_augmented_mean']:.3f} | "
                     f"{s['centroid_error_median_mm']:.1f} mm | {s['fallback_atlases']} |")
    a = summary["all"]
    lines += ["", "Per organ (1 to 12), all cases:", "",
              "| | " + " | ".join(str(i) for i in range(1, 13)) + " |", "|---" * 13 + "|",
              "| vote Dice | " + " | ".join(f"{v:.2f}" for v in a["per_organ_vote_dice"]) + " |",
              "| support recall | " + " | ".join(f"{v:.2f}" for v in a["per_organ_support_recall"]) + " |", ""]
    lines.append("By source: " + ", ".join(f"{k} {v['vote_dice_mean']:.3f} (n={v['n']})" for k, v in a["by_source"].items()))
    if a["below_0.25"]:
        lines.append(f"\nCases with vote Dice below {LOW_DICE}: {', '.join(a['below_0.25'])}")
    if a["targets_with_3plus_fallbacks"]:
        lines.append(f"\nTargets with 3 or more fallback atlases: {', '.join(a['targets_with_3plus_fallbacks'])}")
    lines.append("\n**Invariants:** " + ("all passed" if not failures else "FAILED: " + "; ".join(failures)))
    lines += ["", "| Case | Split | Vote Dice | Augmented | 2 mm Dice | Fallbacks | Inliers (median) | Centroid err mm | w mean |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        reg = r["registration"]
        d2 = reg.get("vote_dice_2mm")
        lines.append(f"| {r['case']} | {r['split']} | {r['vote_dice_mean']:.3f} | {r['vote_dice_mean_augmented']:.3f} | "
                     f"{'' if d2 is None else f'{d2:.3f}'} | {reg.get('status_counts', {}).get('coarse_affine_only', 0)} | "
                     f"{reg.get('inliers_median', 0):.0f} | {r['centroid_error_mm']:.1f} | {r['w_mean']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunk", default=None, help="chunk.json with 'train' and 'val' case names")
    p.add_argument("--cases", nargs="*", default=None, help="explicit case names (split is detected)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--variant", default="tps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args()

    by_name = {case_name(c["image"]): c for c in paired_cases(a.data_dir)}
    _, val_pool = train_val_split(a.data_dir)
    val_names = {case_name(c["image"]) for c in val_pool}
    if a.chunk:
        chunk = json.load(open(a.chunk, encoding="utf-8"))
        names = list(chunk["train"]) + list(chunk["val"])
    elif a.cases:
        names = list(a.cases)
    else:
        raise SystemExit("give --chunk or --cases")

    items, missing = attach_atlas_priors([by_name[n] for n in names], a.variant, a.cache_dir)
    if missing:
        print(f"skipping {len(missing)} cases without a registration cache: {missing}")
    if not items:
        raise SystemExit("no registered case to validate")

    kw = dict(prior_source="atlas")
    tf_val = build_transforms(IMG_SIZE, NUM_CLASSES, train=False, **kw)
    tf_train = build_transforms(IMG_SIZE, NUM_CLASSES, train=True, **kw)
    rows, failures = [], []
    for i, item in enumerate(items, 1):
        name = case_name(item["image"])
        r = validate_case(item, tf_val, tf_train, a.seed)
        r["split"] = "val" if name in val_names else "train"
        r["registration"] = registration_record(name, a.cache_dir, a.variant)
        rows.append(r)
        failures += [f"{name}: {m}" for m in r["invariant_failures"]]
        print(f"[{i}/{len(items)}] {name:10s} {r['split']:5s} vote Dice {r['vote_dice_mean']:.3f} "
              f"(augmented {r['vote_dice_mean_augmented']:.3f})  centroid {r['centroid_error_mm']:.1f} mm  "
              f"fallbacks {r['registration'].get('status_counts', {}).get('coarse_affine_only', 0)}"
              f"{'  INVARIANT FAILURE' if r['invariant_failures'] else ''}", flush=True)

    summary = summarise(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    with open(a.out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": rows, "missing_registration": missing, "invariant_failures": failures},
                  f, indent=1)
    with open(a.out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(rows, summary, failures))
    s = summary["all"]
    print(f"\n{s['n']} cases | vote Dice mean {s['vote_dice_mean']:.3f} median {s['vote_dice_median']:.3f} | "
          f"after augmentation {s['vote_dice_augmented_mean']:.3f} | centroid error {s['centroid_error_median_mm']:.1f} mm | "
          f"{len(s['below_0.25'])} below {LOW_DICE}")
    print("invariants:", "all passed" if not failures else f"{len(failures)} FAILED")
    print(f"wrote {a.out_prefix}.json / .md")
    sys.exit(2 if failures else 0)


if __name__ == "__main__":
    main()
