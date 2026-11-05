"""
Test-time physical refinement of a trained Swin UNETR's predictions, on the CPU (safe next to a GPU run). No training.

For each validation case the network's softmax is refined and scored against the true label on the 96^3 grid:

  base      argmax of the network
  cc        keep only the largest connected component of every organ (stray blobs become background)
  hu<b>     p_c <- p_c * exp(b * valid_c * LL_c), renormalised: Bayes update with the per-scan organ intensity likelihood
            (the same statistics as the HU loss; from the atlas core only, never from the label)
  diff<k>   k steps of edge-guided diffusion of the organ probabilities (Perona-Malik: smoothing is stopped by CT edges)
  and combinations of them.

The checkpoint was picked on these same cases, so every number is slightly optimistic and the refinement settings are
tried on them too: a quick check of which idea is worth building into the model, not a result.

    python scripts/refine_eval.py --ckpt <run>/checkpoint_best.pth --chunk experiments/exp_04_pipeline/chunk_150_50_e100_chunk.json
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only: the 4 GB GPU belongs to the training run

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, PersistentDataset
from monai.metrics import DiceMetric
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.splits import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, attach_atlas_priors, case_name, paired_cases  # noqa: E402
from data.transforms import build_transforms  # noqa: E402
from models.swin_atlas import AtlasAwareSwinUNETR  # noqa: E402
from scripts.train_prototype_v2 import IMG_SIZE, NUM_CLASSES, compact_cached, load_prototype_defs  # noqa: E402
from training.physics_losses import gaussian_blur3d, hu_likelihood_maps  # noqa: E402

NAMES = ["spleen", "r_kidney", "l_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "ivc", "pancreas",
         "r_adrenal", "l_adrenal"]


def lower_priority():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # below normal


def largest_component(pred: torch.Tensor) -> torch.Tensor:
    """pred (X, Y, Z) integer labels: every organ keeps only its largest 3-D connected component, the rest becomes 0."""
    arr, out = pred.numpy(), pred.clone().numpy()
    for k in range(1, NUM_CLASSES):
        lab, n = ndimage.label(arr == k)
        if n > 1:
            sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            out[(arr == k) & (lab != 1 + int(np.argmax(sizes)))] = 0
    return torch.from_numpy(out)


def edge_weight(image: torch.Tensor, sigma_vox: float = 1.0, body_hu: float = -500.0) -> torch.Tensor:
    """exp(-|grad I_s| / kappa), kappa = median gradient magnitude inside the body (image in [0, 1])."""
    smooth = gaussian_blur3d(image.float(), sigma_vox)[0, 0]
    g = torch.gradient(smooth)
    mag = torch.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    body = (image[0, 0] * 2000 - 1000) > body_hu
    kappa = torch.clamp(mag[body].median() if bool(body.any()) else mag.median(), min=1e-4)
    return torch.exp(-mag / kappa)


def diffuse(p: torch.Tensor, w: torch.Tensor, steps: int, tau: float = 0.12) -> torch.Tensor:
    """Explicit edge-guided diffusion of p (C, X, Y, Z) with conductance w (X, Y, Z); no-flux boundary; tau <= 1/6 is stable."""
    for _ in range(steps):
        upd = torch.zeros_like(p)
        for ax in (1, 2, 3):
            d = p.narrow(ax, 1, p.shape[ax] - 1) - p.narrow(ax, 0, p.shape[ax] - 1)
            wf = 0.5 * (w.narrow(ax - 1, 1, w.shape[ax - 1] - 1) + w.narrow(ax - 1, 0, w.shape[ax - 1] - 1))
            flux = d * wf
            upd.narrow(ax, 0, p.shape[ax] - 1).add_(flux)
            upd.narrow(ax, 1, p.shape[ax] - 1).sub_(flux)
        p = (p + tau * upd).clamp_min(0)
        p = p / p.sum(0, keepdim=True).clamp_min(1e-8)
    return p


def hu_update(p: torch.Tensor, ll: torch.Tensor, valid: torch.Tensor, beta: float) -> torch.Tensor:
    """p (C, ...), ll (C-1, ...), valid (C-1,): p_c <- p_c exp(beta valid_c ll_c) for organs, background untouched."""
    term = torch.cat([torch.zeros_like(ll[:1]), valid.view(-1, 1, 1, 1) * ll], 0)
    q = p * torch.exp(beta * term)
    return q / q.sum(0, keepdim=True).clamp_min(1e-8)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--ckpt", required=True)
    a_.add_argument("--chunk", required=True)
    a_.add_argument("--swin-feature-size", type=int, default=48)
    a_.add_argument("--hu-input", action="store_true", help="the checkpoint was trained with --hu-input")
    a_.add_argument("--betas", default="0.25,0.5,1,2")
    a_.add_argument("--diff-steps", default="3,8")
    a_.add_argument("--threads", type=int, default=6)
    a_.add_argument("--limit", type=int, default=None, help="first N validation cases (smoke test)")
    a_.add_argument("--out", default=None)
    a = a_.parse_args()
    torch.set_num_threads(a.threads)
    lower_priority()
    betas = [float(x) for x in a.betas.split(",") if x]
    steps = [int(x) for x in a.diff_steps.split(",") if x]

    kw = load_prototype_defs(os.path.join(REPO, "prototype.ipynb"))["MODEL_KW"]
    model = AtlasAwareSwinUNETR(kw["in_channels"], kw["out_channels"], IMG_SIZE, feature_size=a.swin_feature_size,
                                hidden_size=kw["hidden_size"], conf_gating=True, voxel_prior=True, output_fusion=True,
                                hu_input=a.hu_input)
    for attempt in range(5):  # the live run may be rewriting checkpoint_best.pth
        try:
            state = torch.load(a.ckpt, map_location="cpu", weights_only=True)
            break
        except Exception:
            time.sleep(10)
    else:
        raise SystemExit(f"could not read {a.ckpt}")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

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

    variants = ["base", "cc"] + [f"hu{b:g}" for b in betas] + [f"hu{b:g}+cc" for b in betas] \
        + [f"diff{k}" for k in steps] + [f"diff{k}+cc" for k in steps] \
        + [f"hu{b:g}+diff{k}+cc" for b in betas for k in steps]
    metrics = {v: DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True) for v in variants}

    def refined(name, p, ll, valid, w):
        beta = next((float(t[2:]) for t in name.split("+") if t.startswith("hu")), None)
        k = next((int(t[4:]) for t in name.split("+") if t.startswith("diff")), None)
        q = p
        if beta is not None:
            q = hu_update(q, ll, valid, beta)
        if k is not None:
            q = diffuse(q, w, k)
        pred = q.argmax(0)
        return largest_component(pred) if "cc" in name.split("+") else pred

    t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(loader, 1):
            img, prob, dist, conf = (b[k] for k in ("image", "prob", "dist", "conf"))
            p = F.softmax(model(img, prob, dist, conf).float(), 1)[0]
            ll, valid = hu_likelihood_maps(img, prob, conf)
            ll, valid = ll[0], valid[0]
            w = edge_weight(img)
            for v in variants:
                pred = refined(v, p, ll, valid, w)
                onehot = F.one_hot(pred.long(), NUM_CLASSES).permute(3, 0, 1, 2).unsqueeze(0).float()
                metrics[v](y_pred=onehot, y=b["label"])
            print(f"case {i}/{len(ds)}  {time.time() - t0:.0f} s", flush=True)

    res = {}
    for v, m in metrics.items():
        per, nn_ = m.aggregate()
        res[v] = {"mean": per[nn_.bool()].mean().item(), "organs": [float(x) for x in per]}
    base = res["base"]["mean"]
    lines = [f"checkpoint {a.ckpt}", f"{len(ds)} validation cases, 96^3 grid", "",
             "| variant | mean Dice | change vs base |", "|---|---|---|"]
    lines += [f"| {v} | {r['mean']:.4f} | {r['mean'] - base:+.4f} |" for v, r in res.items()]
    best = max(res, key=lambda k: res[k]["mean"])
    lines += ["", f"per organ: base against the best variant ({best})", "", "| organ | base | best | change |", "|---|---|---|---|"]
    for k, n in enumerate(NAMES):
        s, e = res["base"]["organs"][k], res[best]["organs"][k]
        lines.append(f"| {n} | {s:.3f} | {e:.3f} | {e - s:+.3f} |")
    text = "\n".join(lines)
    print("\n" + text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
