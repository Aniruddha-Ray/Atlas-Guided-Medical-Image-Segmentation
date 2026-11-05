"""
Copy nnUNetTrainerAtlasPrior.py into the installed nnunetv2 package, where nnU-Net looks trainers up by name (for
nnUNetv2_train and for inference). Run again after every edit of the trainer file; --check only compares.

    python nnunet_ext/install.py
    python nnunet_ext/install.py --check
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

import nnunetv2

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnUNetTrainerAtlasPrior.py")
DST = os.path.join(list(nnunetv2.__path__)[0], "training", "nnUNetTrainer", "nnUNetTrainerAtlasPrior.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    same = os.path.exists(DST) and filecmp.cmp(SRC, DST, shallow=False)
    if a.check:
        print(f"{DST}: {'up to date' if same else 'missing or outdated'}")
        sys.exit(0 if same else 1)
    if not same:
        shutil.copyfile(SRC, DST)
    print(f"{'installed' if not same else 'already installed'}: {DST}")


if __name__ == "__main__":
    main()
