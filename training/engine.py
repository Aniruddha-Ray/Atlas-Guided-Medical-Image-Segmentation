"""
Train / validate loops, identical in behaviour to the notebook's cell 27 so
that runs are comparable with the epoch-101 baseline:

  - loss:   DiceCELoss(softmax=True, to_onehot_y=False) on the one-hot label
  - metric: argmax -> one-hot -> DiceMetric(include_background=False,
            reduction="mean_batch"); mean over organs that are present

What is fed to the model as the anatomical prior is chosen by `prior_keys`
(see data.transforms.PRIOR_KEYS). The supervised target is always the label.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric

NUM_CLASSES = 13
ORGAN_NAMES = [f"Organ {i}" for i in range(1, NUM_CLASSES)]


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, optimizer, criterion, device, prior_keys, scaler=None, amp=False, max_steps=None, log_every=0, log=print):
    model.train()
    prob_key, dist_key = prior_keys
    total, steps = 0.0, 0
    t0 = time.time()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        prob = batch[prob_key].to(device)
        dist = batch[dist_key].to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(images, prob, dist)
            loss = criterion(logits.float(), labels)
        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item()
        steps += 1
        if log_every and steps % log_every == 0:
            log(f"    step {steps}: loss {loss.item():.4f} ({(time.time() - t0) / steps:.2f}s/step)")
        if max_steps is not None and steps >= max_steps:
            break
    return total / max(steps, 1)


@torch.no_grad()
def validate(model, loader, device, prior_keys, amp=False, max_steps=None):
    """Returns (mean_dice, per_organ list[float|None])."""
    model.eval()
    prob_key, dist_key = prior_keys
    metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    metric.reset()
    steps = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with _autocast(device, amp):
            logits = model(images, batch[prob_key].to(device), batch[dist_key].to(device))
        pred = F.one_hot(logits.float().argmax(dim=1), NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        metric(y_pred=pred, y=labels)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    per_class, not_nans = metric.aggregate()
    mean = per_class[not_nans.bool()].mean().item()
    per_organ = [float(s) if n else None for s, n in zip(per_class.tolist(), not_nans.tolist())]
    return mean, per_organ
