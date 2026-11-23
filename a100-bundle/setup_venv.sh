#!/bin/bash
# Makes a private Python environment in .venv with torch 2.5.1 (the system torch is too old and pip is read only).
# Run from the bundle folder with  bash setup_venv.sh   Takes about 10 minutes (downloads about 3 GB).
set -e
# Works from inside another (read only) venv too. uv is installed into a folder of this bundle, not into the active environment.
echo "== active python $(command -v python) $(python --version 2>&1), VIRTUAL_ENV=${VIRTUAL_ENV:-none}"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install -q --target .uvpkg uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
  [ -x .uvpkg/bin/uv ] && export PATH="$PWD/.uvpkg/bin:$PATH"
fi
if command -v uv >/dev/null 2>&1 && uv venv --python 3.12 .venv 2>/dev/null; then
  echo "== venv made with uv, python 3.12"; INSTALL="uv pip install --python .venv/bin/python"
else
  echo "== uv or python 3.12 not available, using python -m venv with $(python --version 2>&1) (torch 2.5.1 needs python 3.9 or newer)"
  python -m venv .venv && INSTALL=".venv/bin/python -m pip install"
fi
$INSTALL torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$INSTALL timm huggingface_hub scikit-learn pandas numpy pillow matplotlib cleanlab nbconvert ipykernel
# kagglehub needs a recent kagglesdk, so install them last and without the old kaggle command line tool
$INSTALL -U kagglesdk kagglehub
.venv/bin/python -c "import kagglehub" 2>/dev/null || $INSTALL "kagglehub==0.3.13"
.venv/bin/python -m ipykernel install --user --name a100venv --display-name "a100venv"
.venv/bin/python - <<'PY'
import sys, torch, timm
print("python", sys.version.split()[0], "torch", torch.__version__, "timm", timm.__version__, "cuda available", torch.cuda.is_available())
x = torch.randn(1024, 1024, device="cuda"); print("gpu matmul ok", float((x @ x).sum().isfinite()))
PY
echo "== done, now run  bash run.sh"
