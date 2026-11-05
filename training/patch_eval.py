"""
Whole-volume evaluation for patch-trained models (data/transforms.build_patch_transforms, train=False).

The model sees [CT, P, D, w] in windows of the training patch size (sliding-window inference); Dice is computed per
organ on the resampled volume with the rule of v2_validate / MONAI DiceMetric(include_background=False,
reduction="mean_batch"): a case counts for an organ only if the organ is in its label, an organ's score is the mean over
those cases, and the overall score is the mean over organs. Done on the CPU with index maps, because one-hot volumes of
~13 M voxels x 13 classes do not fit next to the model on a 4 GB GPU.
"""
from __future__ import annotations

import numpy as np
import torch
from monai.inferers import sliding_window_inference


def organ_dice(pred: torch.Tensor, label: torch.Tensor, num_classes: int) -> list:
    """Per-organ Dice (organs 1..C-1) of one case from two index maps of the same shape; None where the label lacks the organ."""
    out = []
    for c in range(1, num_classes):
        g = label == c
        n_g = int(g.sum())
        if n_g == 0:
            out.append(None)
            continue
        p = pred == c
        out.append(2.0 * int((p & g).sum()) / (int(p.sum()) + n_g))
    return out


class DiceTally:
    def __init__(self, num_classes: int):
        self.total, self.count = np.zeros(num_classes - 1), np.zeros(num_classes - 1, dtype=np.int64)

    def add(self, per_organ: list):
        for i, v in enumerate(per_organ):
            if v is not None:
                self.total[i] += v
                self.count[i] += 1

    def result(self):
        """(mean over organs seen at least once, per-organ list with None for unseen organs)."""
        per = [float(t / n) if n else None for t, n in zip(self.total, self.count)]
        seen = [v for v in per if v is not None]
        return (float(np.mean(seen)) if seen else float("nan")), per


def sliding_window_logits(model, image, prob, dist, conf, roi_size, device, amp=False, overlap=0.25, mode="gaussian"):
    """Logits (B, C, ...) on the CPU for a whole volume. Inputs stay on the CPU; each window goes to `device`."""
    n_cls = prob.shape[1]
    # float32: sliding_window_inference accumulates in the input dtype
    x = torch.cat([image, prob, dist, conf], dim=1).float()

    def predictor(w):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(w[:, :1], w[:, 1:1 + n_cls], w[:, 1 + n_cls:2 * n_cls], w[:, 2 * n_cls:2 * n_cls + 1])
        return out.float()

    return sliding_window_inference(x, roi_size, 1, predictor, overlap=overlap, mode=mode, sw_device=device, device="cpu")


def validate_volumes(model, loader, device, roi_size, amp=False, overlap=0.25):
    """(mean Dice, per-organ Dice) over the loader's volumes; same return format as v2_validate."""
    model.eval()
    tally = None
    with torch.no_grad():
        for b in loader:
            logits = sliding_window_logits(model, b["image"], b["prob"], b["dist"], b["conf"], roi_size, device, amp=amp,
                                           overlap=overlap)
            n_cls = logits.shape[1]
            tally = tally or DiceTally(n_cls)
            pred, lab = logits.argmax(1), b["label"][:, 0]
            for i in range(pred.shape[0]):
                tally.add(organ_dice(pred[i], lab[i], n_cls))
    return tally.result()


def atlas_vote_volumes(loader, num_classes: int) -> float:
    """Mean Dice of argmax(P) on the whole volumes: what the atlases alone give (reference, no model)."""
    tally = DiceTally(num_classes)
    for b in loader:
        pred, lab = b["prob"].float().argmax(1), b["label"][:, 0]
        for i in range(pred.shape[0]):
            tally.add(organ_dice(pred[i], lab[i], num_classes))
    return tally.result()[0]
