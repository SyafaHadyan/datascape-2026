# Sourced by run.sh and by hand. Edit DATA if the data folder is somewhere else.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HERE/.venv/bin:$HOME/.local/bin:$PATH"
export VIRTUAL_ENV="$HERE/.venv"
export DS_DATA_ROOT="${DATA:-$HERE/archive}"
export DS_CACHE_DIR="${DS_CACHE_DIR:-$HERE/ds-cache}"
export DS_NOTEBOOK=cv-notebook.ipynb
export DS_KERNEL=a100venv
export CUDA_VISIBLE_DEVICES=0
export DS_NUM_WORKERS="${DS_NUM_WORKERS:-4}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
