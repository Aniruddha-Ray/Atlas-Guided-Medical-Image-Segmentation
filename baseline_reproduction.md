# Baseline Reproduction — Epoch 101 (Experiment 0)

## Command

```bash
pip install -r requirements.txt
cd Atlas-Guided-Medical-Image-Segmentation
python scripts/evaluate_checkpoint.py \
    --checkpoint best_model.pth \
    --data-dir "../Dataset060_Merged_Def" \
    --legacy-buggy-distance
```

`--legacy-buggy-distance` is required here: `best_model.pth` was trained
against the original (buggy) `CustomSpatialDistanceTransformd` semantics
(research_audit.md, Section G). Without this flag, `evaluate_checkpoint.py`
would feed the model correctly-computed distance maps that its weights were
never trained to interpret, and the reproduced Dice would not match the
historical number — that is a distribution-shift experiment, not a
reproduction of Experiment 0.

## Checkpoint

`best_model.pth` — state_dict only, saved whenever validation Dice improved.
Verified directly: Dice peaked at epoch 101 (0.6505) and epochs 102–103 were
both lower (0.6482, 0.6395; see `training_metrics.csv`), so this file was
never overwritten after epoch 101. It is the epoch-101 checkpoint; there is
no separate file literally named "epoch_101" (see `checkpoints/epoch_101/README.md`).

`latest_checkpoint.pth` is NOT epoch 101 — inspected directly:
`torch.load(...)["epoch"] == 102` (0-indexed) → "Epoch 103/200" as printed by
the training loop, with `best_dice == 0.6505...` carried over from epoch 101.
Training continued 2 epochs past the reported stopping point before the run
was halted.

## Dataset

`Dataset060_Merged_Def/` — 680 paired `{image: *_0000.nii.gz, label:
*.nii.gz}` cases (24 GB), 100% labeled (AMOS-style). Split: `random.seed(42)`
+ shuffle + 80/20 → 544 train / 136 validation. `scripts/evaluate_checkpoint.py`
reproduces this split exactly (same seed, same shuffle order, same fraction).

## Configuration

`AtlasGuidedViTUNETR(in_channels=1, out_channels=13, img_size=(96,96,96),
feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
proj_type="perceptron", norm_name="instance", res_block=True)`. Validation
transforms: orientation RAS → foreground crop → 1mm spacing → HU[-1000,1000]
scaled to [0,1] → resize to 96³ → one-hot label (13 ch) → distance map
(12 ch, legacy semantics for this reproduction only). `DiceMetric(include_background=False,
reduction="mean_batch")`.

## Expected metrics (from training_log.txt / training_metrics.csv, epoch 101)

| Organ | Dice |
|---|---|
| Mean | 0.6505 |
| Organ 1 | 0.8315 |
| Organ 2 | 0.7619 |
| Organ 3 | 0.7614 |
| Organ 4 | 0.5678 |
| Organ 5 | 0.5272 |
| Organ 6 | 0.9038 |
| Organ 7 | 0.6899 |
| Organ 8 | 0.7195 |
| Organ 9 | 0.6398 |
| Organ 10 | 0.5509 |
| Organ 11 | 0.4523 |
| Organ 12 | 0.4001 |

## Status

**Pipeline verified end-to-end on GPU; full 136-case run intentionally not
executed here.** `requirements.txt` was installed (with CUDA torch — see the
`--extra-index-url`/`+cu130` pins added after the first install silently
resolved to CPU-only wheels), and `scripts/evaluate_checkpoint.py --limit 3`
was run locally as a smoke test only:

```
Device: cuda
[1/3] amos_0278_0000.nii.gz ... done
[2/3] s0703_0000.nii.gz ... done
[3/3] s0945_0000.nii.gz ... done

[PARTIAL, 3/136 cases] Validation Dice: 0.6757
  Organ 1   0.8744   Organ 5   0.5274   Organ 9    0.6447
  Organ 2   0.8659   Organ 6   0.9182   Organ 10   0.6245
  Organ 3   0.8600   Organ 7   0.6465   Organ 11   0.5038
  Organ 4   0.5311   Organ 8   0.7430   Organ 12   0.3694
```

This confirms data loading, checkpoint loading, the model forward pass, and
`DiceMetric` are all wired correctly (val split matches exactly: same 3
filenames as a fresh notebook run), and the 3-case Dice (0.6757) is in the
same ballpark as the expected full-set mean (0.6505) — consistent with
sampling noise on n=3, not a red flag.

**This 3-case number is NOT the reported baseline metric** — this machine
(RTX 3050, 4 GB) is too slow to finish the full 136-case set in reasonable
time. The full run (drop `--limit`) should be executed on a remote GPU:

```bash
python scripts/evaluate_checkpoint.py \
    --checkpoint best_model.pth \
    --data-dir "../Dataset060_Merged_Def" \
    --legacy-buggy-distance
```

Once run, replace this section with the actual full-set output (mean Dice +
per-organ table) and compare against the "Expected metrics" table above
before proceeding to Phase 2 (atlas-derived priors).

If the full run fails or drifts by more than noise (~±0.01 Dice), check in
this order before touching the atlas pipeline: (1) `--legacy-buggy-distance`
was passed, (2) the val split matches (log `len(val_files)` and the first 3
filenames — should be `amos_0278`, `s0703`, `s0945`), (3) `monai` version is
1.6.0 as pinned, (4) `DiceMetric` reduction/include_background match the
training loop exactly.
