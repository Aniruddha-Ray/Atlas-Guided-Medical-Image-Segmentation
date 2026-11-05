#!/usr/bin/env bash
# Server setup for the atlas-guided segmentation project, one stage at a time. No sudo, no system Python, no driver
# changes: everything lives in a user-space Miniforge install and a project folder.
#
# Target machine (checked on `teesta`): Ubuntu 18.04 (glibc 2.27), 2 x Tesla V100 32 GB (sm_70), NVIDIA driver 530
# (CUDA 12.1), x86-64. That fixes the choices below: PyTorch 2.5.1 + cu121 wheels (they include Volta and install on
# glibc 2.27; cu126/cu128/cu130 builds either need glibc 2.28 or dropped Volta), Python 3.11, MONAI 1.6.0 and the other
# scientific packages pinned to the laptop versions.
#
#   bash server/setup_server.sh preflight    # machine facts and warnings, changes nothing
#   bash server/setup_server.sh miniforge    # install Miniforge under $MINIFORGE_DIR
#   bash server/setup_server.sh env          # conda env with Python 3.11
#   bash server/setup_server.sh torch        # torch 2.5.1 / torchvision 0.20.1 / torchaudio 2.5.1 (cu121) + constraints file
#   bash server/setup_server.sh gpu          # sm_70 present? fp16 autocast + GradScaler forward/backward on every GPU
#   bash server/setup_server.sh repo         # git clone of the project into $ROOT
#   bash server/setup_server.sh data         # checks dataset + cache are in place, prints the rsync commands if not
#   bash server/setup_server.sh deps-check   # pip --dry-run --only-binary: which pinned packages have a usable wheel
#   bash server/setup_server.sh deps         # install the rest (after deps-check passes)
#   bash server/setup_server.sh tests        # pytest -q
#   bash server/setup_server.sh plan         # run.py --mode full --dry-run (plans only, writes nothing)
#   bash server/setup_server.sh bench        # one Swin benchmark step on one GPU: speed and peak memory
#   bash server/setup_server.sh all          # every stage above in order, stopping at the first failure
#
# Settings (environment variables):
#   ROOT              project folder          default $HOME/atlas   (put it on LOCAL disk if you have any: NFS is slow)
#   MINIFORGE_DIR     Miniforge install       default $HOME/miniforge3
#   CONDA_FORGE_PKGS  packages to take from conda-forge instead of pip, e.g. "simpleitk pillow" (use when deps-check
#                     reports a package with no wheel for glibc 2.27); a version can be given: "simpleitk=2.5.*"
#   GPU               GPU index for the benchmark   default 0
# Training is never started by this script.
set -euo pipefail

case "${1:-}" in
  preflight|miniforge|env|torch|gpu|repo|data|deps-check|deps|tests|plan|bench|all) ;;
  *) sed -n '2,36p' "$0"; exit 2;;  # usage: creates nothing
esac

ROOT="${ROOT:-$HOME/atlas}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-atlas-server}"
PY_VER="3.11"
TORCH_VER="2.5.1"; TV_VER="0.20.1"; TA_VER="2.5.1"; CUDA_TAG="cu121"
REPO_URL="https://github.com/Aniruddha-Ray/Atlas-Guided-Medical-Image-Segmentation.git"
REPO="$ROOT/Atlas-Guided-Medical-Image-Segmentation"
DATA="$ROOT/Dataset060_Merged_Def"
REQ="$ROOT/requirements-server.txt"
CONSTRAINTS="$ROOT/constraints.txt"
CONDA_FORGE_PKGS="${CONDA_FORGE_PKGS:-}"
GPU="${GPU:-0}"
CHUNK="experiments/exp_04_pipeline/chunk_150_50_e100_chunk.json"

mkdir -p "$ROOT"
LOG="$ROOT/setup_server.log"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

activate() {
  [ -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ] || die "Miniforge not found at $MINIFORGE_DIR (run the miniforge stage)"
  set +u; source "$MINIFORGE_DIR/etc/profile.d/conda.sh"; conda activate "$ENV_NAME" || die "env $ENV_NAME missing (run the env stage)"; set -u
}

write_requirements() {
  # Laptop versions, minus torch/torchvision/torchaudio (installed from the cu121 index in the torch stage).
  cat > "$REQ" <<'EOF'
monai==1.6.0
numpy==2.3.5
scipy==1.16.2
nibabel==5.4.2
SimpleITK==2.5.6
pandas==3.0.3
einops==0.8.2
matplotlib==3.11.2
pytest==9.1.1
pytorch-lightning==2.6.6
scikit-learn==1.7.2
psutil==7.2.2
EOF
}

filtered_requirements() {
  # Requirements without the packages that come from conda-forge, so pip does not try to replace them.
  local out="$ROOT/requirements-server.filtered.txt"
  cp "$REQ" "$out"
  for spec in $CONDA_FORGE_PKGS; do
    local name="${spec%%[=<>]*}"
    sed -i "/^${name}==/Id" "$out"
  done
  echo "$out"
}

stage_preflight() {
  say "preflight (changes nothing)"
  uname -a
  echo "glibc: $(ldd --version | head -1)"
  command -v curl git >/dev/null || echo "WARNING: curl and git are both needed"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    local drv; drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
    [ "${drv:-0}" -ge 525 ] || echo "WARNING: driver $drv is older than 525; CUDA 12.x wheels may not run"
  else
    echo "WARNING: nvidia-smi not found"
  fi
  echo "ROOT=$ROOT  MINIFORGE_DIR=$MINIFORGE_DIR"
  df -hT "$ROOT" "$(dirname "$MINIFORGE_DIR")" | cat
  case "$(df -T "$ROOT" | awk 'NR==2{print $2}')" in
    nfs*|cifs|fuse*) echo "WARNING: $ROOT is on a network filesystem; the dataset cache is read heavily, local disk is much faster";;
  esac
  echo "need about 60 GB free for data (24 GB) + cache (5 GB) + outputs, and about 8 GB for the environment"
}

stage_miniforge() {
  say "miniforge"
  if [ -x "$MINIFORGE_DIR/bin/conda" ]; then echo "already installed at $MINIFORGE_DIR"; return; fi
  local inst="$ROOT/Miniforge3-Linux-x86_64.sh"
  curl -fL --retry 3 -o "$inst" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  [ -s "$inst" ] || die "download is empty"
  bash "$inst" -b -p "$MINIFORGE_DIR"
  "$MINIFORGE_DIR/bin/conda" --version
}

stage_env() {
  say "conda env $ENV_NAME (Python $PY_VER)"
  set +u; source "$MINIFORGE_DIR/etc/profile.d/conda.sh"; set -u
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env exists"
  else
    conda create -y -n "$ENV_NAME" "python=$PY_VER" pip
  fi
  activate
  which python; python --version; python -m pip --version
  case "$(which python)" in "$MINIFORGE_DIR"*) ;; *) die "python is not the Miniforge one";; esac
}

stage_torch() {
  say "PyTorch $TORCH_VER ($CUDA_TAG)"
  activate
  printf 'torch==%s\ntorchvision==%s\ntorchaudio==%s\n' "$TORCH_VER" "$TV_VER" "$TA_VER" > "$CONSTRAINTS"
  python -m pip install "torch==$TORCH_VER" "torchvision==$TV_VER" "torchaudio==$TA_VER" \
    --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())"
}

stage_gpu() {
  say "GPU smoke test (architecture + fp16 autocast / GradScaler forward and backward on every GPU)"
  activate
  python - <<'PY'
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FAIL: CUDA is not available")
arch = torch.cuda.get_arch_list()
print("arch list:", arch)
bad = False
for i in range(torch.cuda.device_count()):
    dev = torch.device("cuda", i)
    cc = torch.cuda.get_device_capability(i)
    tag = f"sm_{cc[0]}{cc[1]}"
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({tag}) {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GiB")
    if tag not in arch:
        print(f"FAIL: {tag} is not in this build's architecture list")
        bad = True
        continue
    torch.manual_seed(0)
    conv = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.InstanceNorm3d(8), nn.GELU(), nn.Conv3d(8, 13, 1)).to(dev)
    attn = nn.MultiheadAttention(64, 4, batch_first=True).to(dev)
    opt = torch.optim.AdamW(list(conv.parameters()) + list(attn.parameters()), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(1, 1, 32, 32, 32, device=dev)
    y = torch.randint(0, 13, (1, 32, 32, 32), device=dev)
    tok = torch.randn(1, 20, 64, device=dev)
    for step in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(conv(x).float(), y) + attn(tok, tok, tok, need_weights=False)[0].float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    ok = bool(torch.isfinite(loss))
    print(f"  forward/backward x3 under autocast fp16: loss {loss.item():.4f} finite={ok}")
    bad |= not ok
sys.exit(1 if bad else 0)
PY
  echo "GPU smoke test passed"
}

stage_repo() {
  say "repository"
  if [ -d "$REPO/.git" ]; then echo "already cloned at $REPO"; git -C "$REPO" log --oneline | head -1; return; fi
  git clone "$REPO_URL" "$REPO"
  git -C "$REPO" log --oneline | head -1
}

stage_data() {
  say "dataset and cache"
  local ok=1
  if [ -d "$DATA" ]; then echo "dataset: $(find "$DATA" -name '*.nii.gz' | wc -l) NIfTI files in $DATA ($(du -sh "$DATA" | cut -f1))"; else echo "MISSING: $DATA"; ok=0; fi
  if [ -d "$REPO/cache" ]; then echo "cache: $(du -sh "$REPO/cache" | cut -f1) in $REPO/cache"; else echo "MISSING: $REPO/cache (optional: it saves re-running the atlas registration)"; fi
  if [ "$ok" = 0 ]; then
    cat <<EOF

Copy from the laptop (Git Bash or WSL there), resumable:
  rsync -avP "/c/Users/Asus/OneDrive/Desktop/Atlas Based Medical Segmentation/Dataset060_Merged_Def/" USER@teesta:$DATA/
  rsync -avP "/c/Users/Asus/OneDrive/Desktop/Atlas Based Medical Segmentation/Atlas-Guided-Medical-Image-Segmentation/cache/" USER@teesta:$REPO/cache/
The dataset must sit next to the repo (data/splits.py looks for ../Dataset060_Merged_Def).
EOF
    die "dataset missing"
  fi
}

stage_deps_check() {
  say "dependency dry run (wheels only)"
  activate
  write_requirements
  local req; req="$(filtered_requirements)"
  if [ -n "$CONDA_FORGE_PKGS" ]; then echo "taking from conda-forge instead of pip: $CONDA_FORGE_PKGS"; fi
  if python -m pip install --dry-run --only-binary=:all: -c "$CONSTRAINTS" -r "$req"; then
    echo "every pinned package has a usable wheel"
  else
    cat <<EOF

A package has no wheel that this machine (glibc 2.27) accepts, or two pins conflict. Read the error above. If it names a
package with no matching distribution (Pillow and SimpleITK are the likely ones), take it from conda-forge:
  CONDA_FORGE_PKGS="simpleitk pillow" bash server/setup_server.sh deps-check
EOF
    exit 1
  fi
}

stage_deps() {
  say "install the remaining packages"
  activate
  write_requirements
  if [ -n "$CONDA_FORGE_PKGS" ]; then
    # conda-forge first, then pip, so conda never overwrites pip-installed files
    conda install -y -c conda-forge $CONDA_FORGE_PKGS
  fi
  local req; req="$(filtered_requirements)"
  python -m pip install -c "$CONSTRAINTS" -r "$req"
  python -c "import torch, monai, numpy, scipy, nibabel, SimpleITK, pandas; print('torch', torch.__version__, '| monai', monai.__version__, '| numpy', numpy.__version__, '| SimpleITK', SimpleITK.Version_VersionString())"
  python -c "import torch; assert torch.__version__.startswith('$TORCH_VER'), torch.__version__; print('torch still', torch.__version__)"
}

stage_tests() {
  say "pytest -q"
  activate
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="" python -m pytest -q
}

stage_plan() {
  say "run.py --mode full --dry-run (plan only)"
  activate
  cd "$REPO"
  python run.py --mode full --dry-run
}

stage_bench() {
  say "Swin benchmark step on GPU $GPU"
  activate
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/train_prototype_v2.py --experiment exp_04_pipeline --run-name bench_server \
    --bench --amp --backbone swin --gamma-init=1.0 --chunk "$CHUNK"
  cat experiments/exp_04_pipeline/bench_server/bench.json 2>/dev/null || true
}

case "${1:-}" in
  preflight)   stage_preflight;;
  miniforge)   stage_miniforge;;
  env)         stage_env;;
  torch)       stage_torch;;
  gpu)         stage_gpu;;
  repo)        stage_repo;;
  data)        stage_data;;
  deps-check)  stage_deps_check;;
  deps)        stage_deps;;
  tests)       stage_tests;;
  plan)        stage_plan;;
  bench)       stage_bench;;
  all)         stage_preflight; stage_miniforge; stage_env; stage_torch; stage_gpu; stage_repo; stage_data
               stage_deps_check; stage_deps; stage_tests; stage_plan; stage_bench
               say "all stages passed; no training was started";;
  *)           sed -n '2,36p' "$0"; exit 2;;
esac
