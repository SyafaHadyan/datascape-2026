#!/bin/bash
# One time setup in the Jupyter terminal. Run it from the folder that holds this bundle, with bash setup.sh
set -u
echo "== GPU"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "== CPUs $(nproc)"
echo "== python $(python --version 2>&1)"
python - <<'PY'
import importlib
need = {"torch": "torch", "torchvision": "torchvision", "timm": "timm", "sklearn": "scikit-learn", "pandas": "pandas",
        "numpy": "numpy", "PIL": "pillow", "matplotlib": "matplotlib", "kagglehub": "kagglehub", "cleanlab": "cleanlab",
        "nbconvert": "nbconvert", "ipykernel": "ipykernel", "huggingface_hub": "huggingface_hub"}
missing = []
for mod, pkg in need.items():
    try:
        m = importlib.import_module(mod)
        print("ok     ", mod, getattr(m, "__version__", ""))
    except Exception:
        print("MISSING", mod)
        missing.append(pkg)
open("missing.txt", "w").write(" ".join(missing))
PY
missing=$(cat missing.txt); rm -f missing.txt
if [ -n "$missing" ]; then echo "== installing $missing"; pip install -q $missing; fi
python -c "import torch; print('cuda available', torch.cuda.is_available(), 'torch', torch.__version__)"
echo "== internet check"
python - <<'PY'
import socket
for host in ("huggingface.co", "www.kaggle.com", "pypi.org"):
    try:
        socket.getaddrinfo(host, 443); print("resolves", host)
    except OSError as e:
        print("CANNOT RESOLVE", host, e)
PY
