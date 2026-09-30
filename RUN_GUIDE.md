# Run guide: atlas pipeline and Experiment 1 on a server

Every command runs from the repository root. The dataset is expected at
`../Dataset060_Merged_Def`; pass `--data-dir` if yours is elsewhere.

## 0. Environment

```bash
pip install -r requirements.txt     # pins torch +cu130; change the tag to match your server's CUDA driver
python -m pytest tests/ -q          # expect 48 passed
```

## 1. Atlas library (≈1 min, CPU)

```bash
python scripts/build_atlas_library.py
```

Reads only the 10 atlas labels listed in `atlas/atlas_selection.json`. It
scores landmarks with golden transformations over all 90 ordered atlas pairs
and writes `cache/atlas_library/atlas_library.npz` plus a summary JSON.

## 2. Register the atlases onto every target (the expensive step)

Each target takes about 2 min (10 atlases, TPS variant). That is about 670
targets in total: 534 train (atlases already excluded) and 136 val. Split the
work into shards and run one process per shard:

```bash
for i in 0 1 2 3 4 5 6 7; do
  python scripts/register_targets.py --split train --num-shards 8 --shard-index $i > logs/reg_train_$i.log 2>&1 &
done
for i in 0 1 2 3; do
  python scripts/register_targets.py --split val --num-shards 4 --shard-index $i --eval-gt > logs/reg_val_$i.log 2>&1 &
done
wait
```

- Output goes to `cache/warped_labels/tps/`, `cache/registrations/tps/` and
  (with `--qa-png`) `cache/qa/tps/`.
- Finished targets are skipped on a rerun. Add `--force` to recompute them.
- `--eval-gt` on validation writes an **evaluation-only** atlas-only Dice
  into each QA JSON. That is the registration quality diagnostic. Ground
  truth never feeds the priors.
- Every QA JSON has a `status_counts` entry. Any `coarse_affine_only`
  fallback deserves a look before trusting the result.

## 3. Experiment 1: atlas priors, existing model and loss

```bash
python scripts/train.py --experiment exp_01_atlas_prior --run-name ft_from_101 --epochs 50 --amp
```

- Starts from `best_model.pth` (θ101). Only the prior source changes: loss,
  optimizer, lr and augmentation are the notebook's.
- **Epoch 0** in `metrics.csv` is θ101 evaluated with atlas priors before any
  fine-tuning. It is the reference point for how much the baseline relied on
  ground-truth priors.
- The script refuses to start if some cases have no registration cache.
  `--allow-missing` is for smoke tests only.
- Each run writes to `experiments/<experiment>/<run_name>/`: `config.json`,
  `command.txt`, `git.txt`, `git_diff.patch`, `metrics.csv`, `summary.json`,
  `training.log` and the checkpoints. An existing run directory is never
  overwritten.

Controls worth running alongside it, each changing one thing:

```bash
# same priors, trained from scratch instead of from θ101
python scripts/train.py --run-name scratch --init-checkpoint none --epochs 200 --amp
# corrected rotation augmentation (±15 degrees instead of ±15 radians)
python scripts/train.py --run-name ft_rot15deg --rotate-range-deg 15 --epochs 50 --amp
```

## 4. Evaluation of any checkpoint

```bash
# baseline reproduction (ground-truth priors, legacy distance semantics)
python scripts/evaluate_checkpoint.py --checkpoint best_model.pth --legacy-buggy-distance
# atlas-prior evaluation
python scripts/evaluate_checkpoint.py --checkpoint experiments/exp_01_atlas_prior/ft_from_101/checkpoint_best.pth --prior-source atlas
```

## What is still open

- **Registration tuning.** Tune TPS λ, the RANSAC threshold and patch size on
  non-atlas *training* targets, and try organ-aware refinement. See
  `experiments/registration_log.md`.
- **Later experiments.** Exp 2 adds the confidence-weighted atlas-consistency
  loss (w(v) is already produced as `conf`). Exp 3 adds the CT edge loss.
  Exp 4 adds the HU loss, only if it is justified.
