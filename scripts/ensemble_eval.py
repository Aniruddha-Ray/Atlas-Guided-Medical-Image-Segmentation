"""
Quick check of how much averaging a Swin UNETR and a UNETR checkpoint helps, on the CPU (safe next to a GPU run).

Both models see the same validation samples (same chunk, cached 96^3 grid, same priors). For each weight alpha the
prediction is argmax(alpha * softmax(Swin) + (1 - alpha) * softmax(UNETR)); alpha 1 is Swin alone and 0 is UNETR alone,
which double as a check against the Dice values logged in training. Both checkpoints were picked on these same cases,
so every number here is slightly optimistic, and alpha is chosen on them too: a rough check, not a result.

    python scripts/ensemble_eval.py --swin <run>/checkpoint_e67_copy.pth --unetr <run>/checkpoint_best.pth \
        --chunk experiments/exp_04_pipeline/chunk_150_50_e100_chunk.json
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only: the 4 GB GPU belongs to the training run

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from monai.data import DataLoader, PersistentDataset
from monai.metrics import DiceMetric

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.splits import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, attach_atlas_priors, case_name, paired_cases  # noqa: E402
from data.transforms import build_transforms  # noqa: E402
from models.swin_atlas import AtlasAwareSwinUNETR  # noqa: E402
from scripts.train_prototype_v2 import IMG_SIZE, NUM_CLASSES, compact_cached, load_prototype_defs  # noqa: E402

NAMES = ["spleen", "r_kidney", "l_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "ivc", "pancreas",
         "r_adrenal", "l_adrenal"]


def lower_priority():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # below normal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--swin", required=True)
    p.add_argument("--unetr", required=True)
    p.add_argument("--chunk", required=True)
    p.add_argument("--alphas", default="1,0.9,0.8,0.7,0.6,0.5,0.3,0")
    p.add_argument("--swin-feature-size", type=int, default=48)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="first N validation cases (smoke test)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    lower_priority()
    alphas = [float(x) for x in a.alphas.split(",")]

    ns = load_prototype_defs(os.path.join(REPO, "prototype.ipynb"))
    kw_model = ns["MODEL_KW"]
    unetr = ns["AtlasAwareViTUNETR"](**kw_model, conf_gating=True, voxel_prior=True, output_fusion=True)
    swin = AtlasAwareSwinUNETR(kw_model["in_channels"], kw_model["out_channels"], IMG_SIZE,
                               feature_size=a.swin_feature_size, hidden_size=kw_model["hidden_size"],
                               conf_gating=True, voxel_prior=True, output_fusion=True)
    unetr.load_state_dict(torch.load(a.unetr, map_location="cpu", weights_only=True))
    swin.load_state_dict(torch.load(a.swin, map_location="cpu", weights_only=True))
    unetr.eval(), swin.eval()

    chunk = json.load(open(a.chunk, encoding="utf-8"))
    by_name = {case_name(c["image"]): c for c in paired_cases(DEFAULT_DATA_DIR)}
    cases, missing = attach_atlas_priors([by_name[n] for n in chunk["val"]], "tps", DEFAULT_CACHE_DIR)
    if missing:
        raise SystemExit(f"missing registration cache for {missing[:5]}")
    cases = cases[: a.limit]
    tf = compact_cached(build_transforms(IMG_SIZE, NUM_CLASSES, train=False, prior_source="atlas",
                                         rotate_range=tuple(float(x) for x in torch.tensor([-0.2617994, 0.2617994]))))
    ds = PersistentDataset(cases, tf, cache_dir=os.path.join(DEFAULT_CACHE_DIR, "persistent_v2_u8"), track_meta=True,
                           weights_only=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    metrics = {al: DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True) for al in alphas}
    agree, t0 = 0.0, time.time()
    with torch.no_grad():
        for i, b in enumerate(loader, 1):
            img, prob, dist, conf = (b[k] for k in ("image", "prob", "dist", "conf"))
            ps = F.softmax(swin(img, prob, dist, conf).float(), 1)
            pu = F.softmax(unetr(img, prob, dist, conf).float(), 1)
            agree += (ps.argmax(1) == pu.argmax(1)).float().mean().item()
            for al, m in metrics.items():
                pred = F.one_hot((al * ps + (1 - al) * pu).argmax(1), NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
                m(y_pred=pred, y=b["label"])
            print(f"case {i}/{len(ds)}  {time.time() - t0:.0f} s", flush=True)

    res = {}
    for al, m in metrics.items():
        per, nn_ = m.aggregate()
        res[al] = {"mean": per[nn_.bool()].mean().item(), "organs": [float(x) for x in per]}
    lines = [f"cases {len(ds)}, voxel agreement between the two models {agree / len(ds):.3f}",
             "| alpha (Swin weight) | mean Dice |", "|---|---|"]
    lines += [f"| {al:g} | {r['mean']:.4f} |" for al, r in res.items()]
    best = max(res, key=lambda k: res[k]["mean"])
    lines += ["", f"per organ: Swin alone (alpha 1), UNETR alone (alpha 0), best alpha {best:g}", "",
              "| organ | Swin | UNETR | ensemble | gain vs Swin |", "|---|---|---|---|---|"]
    for k, n in enumerate(NAMES):
        s, u, e = (res[x]["organs"][k] for x in (1.0, 0.0, best)) if 1.0 in res and 0.0 in res else (0, 0, 0)
        lines.append(f"| {n} | {s:.3f} | {u:.3f} | {e:.3f} | {e - s:+.3f} |")
    text = "\n".join(lines)
    print("\n" + text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
