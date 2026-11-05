"""
Run an nnU-Net v2 command with this project's data folders set (nnUNet_raw / nnUNet_preprocessed / nnUNet_results under
NNUNET_ROOT, default ~/nnunet_data, outside OneDrive) and torch.compile off (no Triton on Windows). The nnUNetv2_*
executables need not be on PATH.

    python nnunet_ext/nn.py plan_and_preprocess -d 71 -c 3d_fullres -gpu_memory_target 4 -overwrite_plans_name nnUNetPlans_4G
    python nnunet_ext/nn.py train 71 3d_fullres 0 -tr nnUNetTrainerAtlasPrior_100epochs -p nnUNetPlans_4G
    python nnunet_ext/nn.py --root C:/Users/Asus/nnunet_trial train ...
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig

DEFAULT_ROOT = os.environ.get("NNUNET_ROOT", os.path.join(os.path.expanduser("~"), "nnunet_data"))


def nnunet_env(root: str = DEFAULT_ROOT) -> dict:
    env = dict(os.environ)
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        env[k] = os.path.join(root, k)
        os.makedirs(env[k], exist_ok=True)
    env.setdefault("nnUNet_compile", "f")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def executable(command: str) -> str:
    name = f"nnUNetv2_{command}" + (".exe" if os.name == "nt" else "")
    for scheme in (f"{os.name}_user", sysconfig.get_default_scheme()):
        try:
            path = os.path.join(sysconfig.get_path("scripts", scheme), name)
        except KeyError:
            continue
        if os.path.exists(path):
            return path
    raise SystemExit(f"{name} not found; is nnunetv2 installed for this Python ({sys.executable})?")


def main():
    args = sys.argv[1:]
    root = DEFAULT_ROOT
    if args[:1] == ["--root"]:
        root, args = args[1], args[2:]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(subprocess.run([executable(args[0])] + args[1:], env=nnunet_env(root)).returncode)


if __name__ == "__main__":
    main()
