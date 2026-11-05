"""
From-scratch training of the redesigned atlas-aware ViT-UNETR (prototype.ipynb, section 9) on a small chunk.

The model, loss and prior-dropout code are NOT duplicated here: the "#v2-cell definitions" cell of
prototype.ipynb is executed, so the code that passed the notebook's 21 checks is the code that trains.
They move into models/ and training/ once the prototype is accepted.

Fits a 4 GB GPU: micro-batch 1 in fp16 (a 96^3 volume through the 122 M-parameter ViT uses ~3 GB), with gradient
accumulation to the effective batch size. The network has only instance/layer norm, so accumulating N micro-batches
is mathematically the same as a batch of N.

    python scripts/train_prototype_v2.py --bench                      # timing and memory only, no training
    python scripts/train_prototype_v2.py --run-name v2_chunk --epochs 40 --amp
    python scripts/train_prototype_v2.py --full-data --run-name v2_full --epochs 100 --amp   # all 534 + 136 targets

The deterministic part of the data pipeline (load, 1 mm resample, resize, atlas-stack resampling) is cached on disk
once per case. Label and atlas stack are stored as uint8 (they are integer labels), ~13 MB per case instead of ~60 MB;
they are cast back to float32 before augmentation, so the samples are bit-identical to the uncached pipeline.

Patch mode (--patch-size 96): instead of resizing each whole volume to 96^3 (3-5 mm voxels), volumes keep --spacing mm
(default 1.5) and training uses class-balanced 96^3 patches, --patches-per-case per case and epoch; validation runs on the
whole volumes by sliding window (training/patch_eval.py). Its Dice is on the --spacing grid, so it is not comparable with
the 96^3-resize numbers. The cache (~0.1-0.2 GB per case) goes to <cache-dir>/persistent_patch_s<spacing>_u8.

    python scripts/train_prototype_v2.py --backbone swin --patch-size 96 --spacing 1.5 --run-name swin_patch --epochs 100 --amp
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import monai.transforms as mt
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, PersistentDataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from data.splits import (DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, attach_atlas_priors, case_name, paired_cases,
                         split_with_atlases_removed)
from data.transforms import build_patch_transforms, build_transforms
from training.patch_eval import atlas_vote_volumes, validate_volumes
from training.physics_losses import edge_aware_loss, hu_consistency_loss, spacing_from_affine

IMG_SIZE = (96, 96, 96)
NUM_CLASSES = 13
ORGAN_IDS = list(range(1, NUM_CLASSES))


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def _cache_boundary(d):
    # A plain callable (not a monai Transform): PersistentDataset caches everything before it and nothing after.
    return d


def compact_cached(tf):
    """Same pipeline, with label and atlas stack stored as uint8 in the on-disk cache."""
    ts = list(tf.transforms)
    i = next(k for k, t in enumerate(ts) if isinstance(t, mt.ResampleToMatchd)) + 1
    keys = ["label", "atlas_labels"]
    ts[i:i] = [mt.CastToTyped(keys=keys, dtype=torch.uint8), _cache_boundary,
               mt.CastToTyped(keys=keys, dtype=torch.float32)]
    return mt.Compose(ts)


def load_prototype_defs(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    src = next("".join(c["source"]) for c in cells
               if c["cell_type"] == "code" and "#v2-cell definitions" in "".join(c["source"]))
    ns = {"torch": torch, "IMG_SIZE": IMG_SIZE, "NUM_CLASSES": NUM_CLASSES}
    exec(compile(src, "prototype.ipynb#definitions", "exec"), ns)
    return ns


def atlas_vote_dice(loader):
    """Dice of argmax(P) against the label: what the atlases alone give on this grid (reference, no model)."""
    metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    for b in loader:
        pred = F.one_hot(b["prob"].argmax(1), NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        metric(y_pred=pred, y=b["label"])
    per, nn_ = metric.aggregate()
    return per[nn_.bool()].mean().item()


def without_prior(loader):
    for b in loader:
        yield {**b, "prob": torch.full_like(b["prob"], 1.0 / NUM_CLASSES), "dist": torch.ones_like(b["dist"]),
               "conf": torch.zeros_like(b["conf"])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="exp_02_prototype_v2")
    p.add_argument("--run-name", default=None)
    p.add_argument("--chunk", default=None, help="chunk.json from scripts/select_chunk.py")
    p.add_argument("--full-data", action="store_true", help="every non-atlas training target + the full validation split")
    p.add_argument("--persist-dir", default=None, help="on-disk sample cache (default <cache-dir>/persistent_v2_u8)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--variant", default="tps")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step (effective batch = micro x accum)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--prior-dropout", type=float, default=0.3)
    p.add_argument("--consistency-weight", type=float, default=0.1)
    p.add_argument("--ramp-epochs", type=int, default=10)
    # CT-physics regularisers (training/physics_losses.py, model_walkthrough.md 6.3); 0 = off
    p.add_argument("--hu-weight", type=float, default=0.0, help="weight of the HU-consistency loss (ramped like --consistency-weight)")
    p.add_argument("--edge-weight", type=float, default=0.0, help="weight of the edge-aware loss (ramped like --consistency-weight)")
    p.add_argument("--hu-delta", type=float, default=2.0, help="Huber threshold, in units of the organ's HU sigma")
    p.add_argument("--hu-core-p", type=float, default=0.9, help="atlas core for HU statistics: P_c >= this")
    p.add_argument("--hu-core-w", type=float, default=0.8, help="atlas core for HU statistics: w >= this")
    p.add_argument("--hu-min-voxels", type=int, default=50, help="skip an organ whose atlas core is smaller")
    p.add_argument("--hu-sigma-min", type=float, default=10.0, help="floor on the per-organ HU sigma")
    p.add_argument("--edge-sigma-vox", type=float, default=1.0, help="Gaussian smoothing of the CT before its gradient")
    p.add_argument("--rotate-range-deg", type=float, default=15.0, help="degrees (the baseline's was radians by mistake)")
    p.add_argument("--conf-gating", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--voxel-prior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-fusion", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--image-only", action="store_true",
                   help="baseline: the network only ever sees uninformative priors (P=1/13, D=1, w=0), in training and validation")
    p.add_argument("--gamma-init", type=float, default=0.0,
                   help="initial output-fusion weight (all classes): 1.0 starts the network at logits = z + w*log(P), i.e. at the atlas vote")
    p.add_argument("--backbone", choices=("vit", "swin"), default="vit", help="vit = UNETR (default), swin = Swin UNETR (models/swin_atlas.py)")
    p.add_argument("--swin-feature-size", type=int, default=48, help="Swin UNETR feature size (multiple of 12; 48 is the published size)")
    p.add_argument("--hu-input", action="store_true",
                   help="Swin only: per-scan organ HU-likelihood maps (from the atlas core) as 12 extra input channels, plus a "
                        "learnable per-organ weight on the log-likelihood in the output fusion (init 0)")
    p.add_argument("--swin-checkpoint", action="store_true", help="activation checkpointing in the Swin stages (less memory, slower)")
    p.add_argument("--patch-size", type=int, default=None,
                   help="patch mode: train on PxPxP patches of volumes kept at --spacing mm, validate by sliding window "
                        "(default: off, every volume resized to 96^3 as before)")
    p.add_argument("--spacing", type=float, default=1.5, help="patch mode: isotropic voxel size in mm")
    p.add_argument("--patches-per-case", type=int, default=4, help="patch mode: patches drawn from each training case per epoch")
    p.add_argument("--sw-overlap", type=float, default=0.25, help="patch mode: sliding-window overlap at validation")
    p.add_argument("--val-every", type=int, default=1, help="validate every N epochs (and at the last epoch)")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--tune-holdout", type=int, default=0,
                   help="hyperparameter tuning: hold N of the TRAINING cases out as the validation set (fixed shuffle) and "
                        "leave the real validation split untouched, so it is never used to choose settings")
    p.add_argument("--bench", action="store_true", help="time the data pipeline, a few steps and validation; then exit")
    p.add_argument("--resume", default=None, help="checkpoint_last.pth of a previous run of this script")
    p.add_argument("--init-from", default=None, help="checkpoint of a previous run: start a NEW run from its model weights only "
                                                     "(fresh optimizer, schedule and best-epoch tracking)")
    args = p.parse_args()

    if args.full_data:
        train_pool, val_pool = split_with_atlases_removed(args.data_dir)
    else:
        chunk_path = args.chunk or os.path.join(_REPO_ROOT, "experiments", args.experiment, "chunk.json")
        chunk = json.load(open(chunk_path, encoding="utf-8"))
        by_name = {case_name(c["image"]): c for c in paired_cases(args.data_dir)}
        train_pool, val_pool = [by_name[n] for n in chunk["train"]], [by_name[n] for n in chunk["val"]]
    train_cases, miss_t = attach_atlas_priors(train_pool, args.variant, args.cache_dir)
    val_cases, miss_v = attach_atlas_priors(val_pool, args.variant, args.cache_dir)
    if (miss_t or miss_v) and not (args.allow_missing or args.bench):
        raise SystemExit(f"no registration cache for {len(miss_t)} train / {len(miss_v)} val targets "
                         f"(e.g. {(miss_t + miss_v)[:5]}); run scripts/register_targets.py first")
    if args.tune_holdout:
        order = sorted(train_cases, key=lambda c: case_name(c["image"]))
        random.Random(12345).shuffle(order)
        if not 0 < args.tune_holdout < len(order):
            raise SystemExit(f"--tune-holdout {args.tune_holdout} needs 0 < N < {len(order)} training cases")
        val_cases, train_cases = order[: args.tune_holdout], order[args.tune_holdout:]
    if args.bench:  # timing needs only a few cases
        train_cases, val_cases = train_cases[:4], val_cases[:2]
    train_cases, val_cases = train_cases[: args.limit_train], val_cases[: args.limit_val]
    if not train_cases or not val_cases:
        raise SystemExit("no registered training or validation target to use")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    amp = args.amp and device.type == "cuda"

    run_name = args.run_name or ("bench" if args.bench else time.strftime("%Y%m%d_%H%M%S"))
    out_dir = os.path.join(_REPO_ROOT, "experiments", args.experiment, run_name)
    if os.path.exists(out_dir) and not args.bench and not args.resume:
        raise SystemExit(f"run directory already exists, refusing to overwrite: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    log = Tee(os.path.join(out_dir, "training.log"))

    ns = load_prototype_defs(os.path.join(_REPO_ROOT, "prototype.ipynb"))
    AtlasAwareViTUNETR, MODEL_KW = ns["AtlasAwareViTUNETR"], ns["MODEL_KW"]
    PriorDropoutd, atlas_consistency_loss = ns["PriorDropoutd"], ns["atlas_consistency_loss"]
    consistency_lambda, v2_validate = ns["consistency_lambda"], ns["v2_validate"]

    def validate(m, loader):
        if patch:
            return validate_volumes(m, loader, device, patch, amp=amp, overlap=args.sw_overlap)
        return v2_validate(m, loader, device, amp=amp)

    rot = tuple(float(x) for x in np.deg2rad([-args.rotate_range_deg, args.rotate_range_deg]))
    workers = args.num_workers
    patch = (args.patch_size,) * 3 if args.patch_size else None
    if patch:
        if args.patch_size % 32:
            raise SystemExit("--patch-size must be a multiple of 32 (five 2x downsamplings)")
        if args.backbone == "vit" and patch != IMG_SIZE:
            raise SystemExit(f"the ViT backbone is built for {IMG_SIZE[0]}^3 inputs; use --patch-size {IMG_SIZE[0]}")
        pkw = dict(spacing=(args.spacing,) * 3, num_classes=NUM_CLASSES)
        persist = args.persist_dir or os.path.join(args.cache_dir, f"persistent_patch_s{args.spacing:g}_u8")
        # the case list repeated: each copy is one patch per epoch (same cache file, a new random crop)
        train_ds = PersistentDataset(train_cases * args.patches_per_case,
                                     build_patch_transforms(patch, train=True, rotate_range=rot, **pkw),
                                     cache_dir=persist, track_meta=True, weights_only=False)
        val_ds = PersistentDataset(val_cases, build_patch_transforms(patch, train=False, **pkw),
                                   cache_dir=persist, track_meta=True, weights_only=False)
        val_workers = min(workers, 1)  # a whole 1.5 mm volume with its priors is ~1-2 GB in RAM
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=val_workers,
                                persistent_workers=val_workers > 0, **({"prefetch_factor": 1} if val_workers else {}))
    else:
        kw = dict(prior_source="atlas", rotate_range=rot)
        persist = args.persist_dir or os.path.join(args.cache_dir, "persistent_v2_u8")
        train_ds = PersistentDataset(train_cases, compact_cached(build_transforms(IMG_SIZE, NUM_CLASSES, train=True, **kw)),
                                     cache_dir=persist, track_meta=True, weights_only=False)
        val_ds = PersistentDataset(val_cases, compact_cached(build_transforms(IMG_SIZE, NUM_CLASSES, train=False, **kw)),
                                   cache_dir=persist, track_meta=True, weights_only=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True, num_workers=workers,
                              persistent_workers=workers > 0)

    if args.backbone == "swin":
        from models.swin_atlas import AtlasAwareSwinUNETR
        model = AtlasAwareSwinUNETR(MODEL_KW["in_channels"], MODEL_KW["out_channels"], IMG_SIZE,
                                    feature_size=args.swin_feature_size, hidden_size=MODEL_KW["hidden_size"],
                                    conf_gating=args.conf_gating, voxel_prior=args.voxel_prior,
                                    output_fusion=args.output_fusion, use_checkpoint=args.swin_checkpoint,
                                    hu_input=args.hu_input).to(device)
    else:
        model = AtlasAwareViTUNETR(**MODEL_KW, conf_gating=args.conf_gating, voxel_prior=args.voxel_prior,
                                   output_fusion=args.output_fusion).to(device)
    if args.gamma_init != 0.0 and args.output_fusion:
        with torch.no_grad():
            model.output_fusion_weight.fill_(args.gamma_init)
    init_note = None
    if args.init_from and not args.resume:
        ck_init = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ck_init["model"] if "model" in ck_init else ck_init  # checkpoint_last.pth or checkpoint_best.pth (plain state dict)
        if args.backbone == "swin":
            from models.swin_atlas import load_state_dict_expanding
            grown, absent = load_state_dict_expanding(model, state)
            init_note = (f"initialised from {args.init_from}: {len(grown)} tensors gained zero-initialised input channels, "
                         f"{len(absent)} tensors kept their fresh initialisation ({', '.join(absent) or 'none'})")
        else:
            model.load_state_dict(state)
            init_note = f"initialised from {args.init_from}"
        del ck_init, state
    n_params = sum(q.numel() for q in model.parameters())
    criterion = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=2.0 ** 12)
    dropper = PriorDropoutd(args.prior_dropout) if args.prior_dropout > 0 and not args.image_only else None
    if dropper is not None:
        dropper.set_random_state(seed=args.seed)

    n_micro = len(train_loader)
    opt_steps_per_epoch = math.ceil(n_micro / args.accum)
    total_steps, warm = opt_steps_per_epoch * args.epochs, int(opt_steps_per_epoch * args.warmup_epochs)
    floor = args.min_lr / args.lr

    def lr_factor(s):
        if s < warm:
            return (s + 1) / max(warm, 1)
        t = (s - warm) / max(total_steps - warm, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    cfg = {**vars(args), "num_train": len(train_cases), "num_val": len(val_cases), "params": n_params,
           "effective_batch": args.micro_batch * args.accum, "optimizer_steps_per_epoch": opt_steps_per_epoch,
           "rotate_range_rad": rot, "missing_train": miss_t, "missing_val": miss_v, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"), indent=2, default=str)
    log(f"Run {out_dir}")
    if init_note:
        log(init_note)
    log(f"{n_params / 1e6:.2f} M params | train {len(train_cases)} val {len(val_cases)} | micro-batch {args.micro_batch} x accum "
        f"{args.accum} = effective {args.micro_batch * args.accum} | {opt_steps_per_epoch} optimizer steps/epoch | amp {amp}")

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)

    zero = torch.zeros((), device=device)
    physics_state = {"logged": False}

    def forward_loss(batch, lam, hu_w=0.0, edge_w=0.0):
        spacing = spacing_from_affine(batch["image"])
        img, lab = batch["image"].to(device), batch["label"].to(device)
        prob, dist, conf = (batch[k].to(device) for k in ("prob", "dist", "conf"))
        if args.image_only:
            prob, dist, conf = torch.full_like(prob, 1.0 / NUM_CLASSES), torch.ones_like(dist), torch.zeros_like(conf)
        # The HU statistics are not a network input, so they use the priors before prior dropout.
        prob_raw, conf_raw = (prob.clone(), conf.clone()) if hu_w > 0 else (prob, conf)
        if dropper is not None:
            for i in range(prob.shape[0]):
                s = dropper({"prob": prob[i], "dist": dist[i], "conf": conf[i]})
                prob[i], dist[i], conf[i] = s["prob"], s["dist"], s["conf"]
        with autocast():
            logits = model(img, prob, dist, conf)
        sup = criterion(logits.float(), lab)
        cons = atlas_consistency_loss(logits, prob, conf) if lam > 0 else zero
        hu, n_hu = zero, 0
        if hu_w > 0:
            hu, n_hu = hu_consistency_loss(logits, img, prob_raw, conf_raw, delta=args.hu_delta, core_p=args.hu_core_p,
                                           core_w=args.hu_core_w, min_voxels=args.hu_min_voxels,
                                           sigma_min=args.hu_sigma_min)
        edge = edge_aware_loss(logits, img, spacing, sigma_vox=args.edge_sigma_vox) if edge_w > 0 else zero
        if (hu_w > 0 or edge_w > 0) and not physics_state["logged"]:
            log(f"physics losses: first-batch voxel spacing {[round(float(s), 3) for s in spacing[0]]} mm, "
                f"HU organs used {n_hu}, HU loss {float(hu):.4f}, edge loss {float(edge):.4f}")
            physics_state["logged"] = True
        return sup, cons, hu, edge, n_hu

    def total_loss(sup, cons, hu, edge, lam, hu_w=0.0, edge_w=0.0):
        return sup + lam * cons + hu_w * hu + edge_w * edge

    if args.bench:
        return bench(args, log, device, model, train_ds, val_loader, forward_loss, total_loss, scaler, optimizer, amp,
                     validate, train_loader)

    if patch:
        ref = atlas_vote_volumes(val_loader, NUM_CLASSES)
        grid = f"whole volumes at {args.spacing:g} mm"
    else:
        ref, grid = atlas_vote_dice(val_loader), "96^3 grid"
    log(f"reference: atlas-only majority vote on the validation chunk ({grid}): Dice {ref:.4f}")

    csv_path = os.path.join(out_dir, "metrics.csv")
    new_file = not os.path.exists(csv_path)
    if new_file:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "lr", "lambda", "train_dice_ce", "train_kl", "val_dice", "val_dice_no_prior",
                                    "gate_mean", "gamma_mean"] + [f"o{k}" for k in ORGAN_IDS] + ["epoch_s", "skipped_steps",
                                                                                                  "train_hu", "train_edge",
                                                                                                  "hu_organs_per_sample"])
    start_epoch, best = 1, {"epoch": 0, "val_dice": -1.0}
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["sched"]); start_epoch, best = ck["epoch"] + 1, ck["best"]
        log(f"resumed from {args.resume} at epoch {start_epoch}")

    t_run = time.time()
    epoch_times = []
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        lam = consistency_lambda(epoch - 1, args.consistency_weight, args.ramp_epochs)
        # physics terms judge the prediction, which is meaningless at the start: same ramp as the consistency term
        hu_w = consistency_lambda(epoch - 1, args.hu_weight, args.ramp_epochs)
        edge_w = consistency_lambda(epoch - 1, args.edge_weight, args.ramp_epochs)
        sup_sum = kl_sum = hu_sum = edge_sum = 0.0
        hu_organs = 0
        skipped = 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            group = min(args.accum, n_micro - (i // args.accum) * args.accum)
            sup, cons, hu, edge, n_hu = forward_loss(batch, lam, hu_w, edge_w)
            scaler.scale(total_loss(sup, cons, hu, edge, lam, hu_w, edge_w) / group).backward()
            sup_sum += sup.item(); kl_sum += cons.item(); hu_sum += float(hu); edge_sum += float(edge)
            hu_organs += n_hu
            if (i + 1) % args.accum == 0 or i == n_micro - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                skipped += int(scaler.get_scale() < before)
                optimizer.zero_grad(set_to_none=True)
                sched.step()
        do_val = epoch % args.val_every == 0 or epoch == args.epochs
        val, per, val_np = None, [None] * len(ORGAN_IDS), None
        if do_val:
            val, per = validate(model, without_prior(val_loader) if args.image_only else val_loader)
        if do_val and (epoch % 5 == 0 or epoch == args.epochs):
            val_np, _ = validate(model, without_prior(val_loader))
        newp = model.new_parameters()
        gate = float(newp["cross_gates"].mean()) if "cross_gates" in newp else float("nan")
        gamma = float(newp["gamma"][1:].mean()) if "gamma" in newp else float("nan")
        secs = time.time() - t0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, optimizer.param_groups[0]["lr"], lam, sup_sum / n_micro, kl_sum / n_micro,
                                    "" if val is None else val,
                                    val_np if val_np is not None else "", gate, gamma]
                                   + [("" if s is None else round(s, 4)) for s in per] + [round(secs, 1), skipped]
                                   + [hu_sum / n_micro, edge_sum / n_micro, hu_organs / max(n_micro * args.micro_batch, 1)])
        extra = f" | no-prior Dice {val_np:.4f}" if val_np is not None else ""
        if args.hu_weight > 0 or args.edge_weight > 0:
            extra += (f" | HU {hu_sum / n_micro:.3f} (x{hu_w:.4f}, {hu_organs / max(n_micro * args.micro_batch, 1):.1f} "
                      f"organs/sample) edge {edge_sum / n_micro:.4f} (x{edge_w:.4f})")
        epoch_times.append(secs)
        steady = epoch_times[1:] or epoch_times  # epoch 1 also builds the sample cache
        eta_h = (args.epochs - epoch) * float(np.mean(steady[-5:])) / 3600
        log(f"epoch {epoch:3d}/{args.epochs}  loss {sup_sum / n_micro:.4f} (+{lam:.3f} x KL {kl_sum / n_micro:.3f})  "
            f"val Dice {'-' if val is None else f'{val:.4f}'}{extra}  gate {gate:.3f} gamma {gamma:+.3f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
            f"{secs:.0f}s  ETA {eta_h:.1f} h{'  skipped ' + str(skipped) if skipped else ''}")
        if val is not None and val > best["val_dice"]:
            best = {"epoch": epoch, "val_dice": val, "per_organ": per}
            torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_best.pth"))
            log("   new best")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "sched": sched.state_dict(), "best": best, "config": cfg},
                   os.path.join(out_dir, "checkpoint_last.pth"))

    json.dump({"best": best, "atlas_only_reference": ref, "train_minutes": (time.time() - t_run) / 60, "config": cfg},
              open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), indent=2, default=str)
    log(f"Done in {(time.time() - t_run) / 60:.1f} min. Best val Dice {best['val_dice']:.4f} at epoch {best['epoch']} "
        f"(atlas-only reference {ref:.4f})")


def bench(args, log, device, model, train_ds, val_loader, forward_loss, total_loss, scaler, optimizer, amp, validate,
          train_loader):
    n = len(train_ds)
    log(f"--- bench: {n} cached train cases, {len(val_loader)} cached val cases ---")
    t0 = time.time()
    for i in range(n):
        train_ds[i]
    first_s = (time.time() - t0) / max(n, 1)
    log(f"first pass (builds the on-disk cache unless already there): {first_s:.1f} s/case")
    t0 = time.time()
    for _ in range(2):
        for i in range(n):
            train_ds[i]
    data_s = (time.time() - t0) / max(2 * n, 1)
    log(f"cached sample (flip/rotate + AtlasPriorsd), one process: {data_s:.2f} s/sample")

    model.train()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        for batch in train_loader:
            torch.cuda.synchronize(); t = time.time()
            sup, cons, hu, edge, _ = forward_loss(batch, 0.1, args.hu_weight, args.edge_weight)
            scaler.scale(total_loss(sup, cons, hu, edge, 0.1, args.hu_weight, args.edge_weight)).backward()
            scaler.unscale_(optimizer); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.time() - t)
    step_s, peak = float(np.mean(times[1:] or times)), torch.cuda.max_memory_allocated() / 2**30
    log(f"train step (fwd+bwd+AdamW, micro-batch {args.micro_batch}): {step_s:.2f} s over {len(times) - 1} steps "
        f"(first {times[0]:.1f} s), peak GPU {peak:.2f} GiB")
    for _ in val_loader:  # builds the validation cache
        pass
    t0 = time.time(); validate(model, val_loader); torch.cuda.synchronize()
    val_s = (time.time() - t0) / max(len(val_loader), 1)
    log(f"validation (fetch + forward): {val_s:.2f} s/case")
    res = {"first_pass_s_per_case": first_s, "data_s_per_sample_main_process": data_s, "step_s_per_micro_batch": step_s,
           "micro_batch": args.micro_batch, "val_s_per_case": val_s, "num_workers": args.num_workers,
           "peak_gpu_gib": peak, "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    with open(os.path.join(os.path.dirname(log.f.name), "bench.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
