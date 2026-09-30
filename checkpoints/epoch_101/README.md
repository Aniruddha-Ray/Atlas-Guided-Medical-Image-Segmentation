# Epoch-101 checkpoint

There is no file physically named `epoch_101` (the checkpoint isn't
duplicated here — `best_model.pth` is 467 MB and copying it would just
double the repo's binary weight for no benefit). The canonical epoch-101
weights are:

```
../../best_model.pth
```

Verified in `research_audit.md` (Section E): validation Dice peaked at
epoch 101 (0.6505) and never improved again in epochs 102–103, so
`best_model.pth` (saved only when Dice improves) was never overwritten
after epoch 101 and is exactly that checkpoint.

Load it with:

```python
model.load_state_dict(torch.load("best_model.pth", map_location=device))
```

or via `scripts/evaluate_checkpoint.py --checkpoint best_model.pth` (see
`baseline_reproduction.md`).

`../../latest_checkpoint.pth` is a different, later checkpoint (epoch 103,
full training state including optimizer) — do not confuse the two.
