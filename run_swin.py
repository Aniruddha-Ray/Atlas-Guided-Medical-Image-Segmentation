"""
Swin UNETR counterpart of run.py: the same pipeline (select, register, validate, library, train, report) with the
atlas-prior Swin UNETR (models/swin_atlas.py) as the backbone. It is the recipe of the laptop run
chunk_150_50_e100_swin_prior: prior in the input and as tokens, gated cross-attention, output fusion with gamma
initialised to 1, Dice + CE plus the KL atlas-consistency loss (weight 0.1, ramped over 10 epochs), prior dropout 0.3,
lr 2e-4 -> 1e-5. Every other flag is run.py's, so the two scripts behave the same way.

    python run_swin.py --mode full --epochs 200 --reg-shards 8       # whole dataset, e.g. on a server
    python run_swin.py --mode chunk --chunk-file experiments/exp_04_pipeline/chunk_150_50_e100_chunk.json --epochs 100
    python run_swin.py --mode full --dry-run                         # plan only

Run name defaults to swin_full_e<epochs> or swin_chunk_<train>_<val>_e<epochs>, so it never collides with run.py's.
The time estimate printed by the plan is calibrated on the UNETR backbone; Swin feature size 48 took about 1.2 times
longer per epoch on the 4 GB laptop card (about 1200 s against about 600 s for UNETR on the 150-case chunk), so scale it.
Extra trainer flags go through --train-args, for example --train-args "--hu-weight 0.03 --edge-weight 50".
"""
from __future__ import annotations

import argparse
import shlex

import run


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--swin-feature-size", type=int, default=48)
    p.add_argument("--swin-checkpoint", action="store_true", help="activation checkpointing (less memory, slower)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--gamma-init", type=float, default=1.0)
    p.add_argument("--train-args", default="")
    p.add_argument("--run-name", default=None)
    own, rest = p.parse_known_args(argv)
    if "-h" in rest or "--help" in rest:
        print(__doc__)
        run.parse_args(["--help"])
    swin = ["--backbone", "swin", f"--swin-feature-size={own.swin_feature_size}", f"--lr={own.lr:g}",
            f"--min-lr={own.min_lr:g}", f"--gamma-init={own.gamma_init:g}"]
    if own.swin_checkpoint:
        swin.append("--swin-checkpoint")
    train_args = " ".join(swin) + (" " + own.train_args if own.train_args else "")
    forwarded = rest + [f"--train-args={train_args}"]
    a = run.parse_args(forwarded)
    if own.run_name is None:
        name = f"swin_chunk_{a.train_size}_{a.val_size}_e{a.epochs}" if a.mode == "chunk" else f"swin_full_e{a.epochs}"
    else:
        name = own.run_name
    run.main(forwarded + [f"--run-name={name}"])


if __name__ == "__main__":
    main()
