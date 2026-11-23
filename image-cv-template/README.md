# Image classification CV template

A reusable setup for Kaggle image classification with a timm backbone, 5-fold cross-validation, saved out-of-fold probabilities, a queue driver for many runs, and a blend checker. It was extracted from the DATASCAPE 2026 project. Read `LESSONS.md` first, it is the short version of what was learned.

## Files

| File | Purpose |
|---|---|
| `cv-notebook.ipynb` | The full pipeline. Config cell, duplicate check, folds, training workers, OOF report, test inference, submission and probability dump. |
| `run_queue.py` | Runs many notebook runs back to back from `queue.jsonl`, each with its own settings. |
| `blend_check.py` | Checks a finished run and tests blends from the saved probability files. |
| `runs.example.json` | Example input for `blend_check.py`. |
| `diagnostics/adversarial_validation_cells.py` | Notebook cells that test whether train and test differ (paste into the notebook). |
| `LESSONS.md` | What worked, what did not, and process lessons. |

## Start a new competition

1. Copy this folder. Put the data in a folder with one subfolder per class for train and one folder for test, or edit `TRAIN_SUBDIR` and `TEST_SUBDIR` in the config cell.
2. In the config cell, set `COMP_SLUG` (or the `DS_COMP_SLUG` environment variable), `TRAIN_SUBDIR` and `TEST_SUBDIR`.
3. Check these parts of `cv-notebook.ipynb` for the new task, because they were written for a 3-class macro F1 problem. The metric in the OOF report and the best-epoch pick, the sample submission format in the submission cell, the letterbox size and the class weighting.
4. Set `RUN_TAG`, then do a sanity run with `DS_SANITY=1` before the full run.
5. Never change `SEED` after the first run. Use `SEED_OFFSET` for other seeds.

## Run on a GPU server

```
export DS_DATA_ROOT=$HOME/data/competition
export DS_CACHE_DIR=$HOME/ds-cache
export CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 cv-notebook.ipynb
```

For several runs, write one JSON object per line in `queue.jsonl` and start the driver in `screen` or `tmux`.

```
{"RUN_TAG": "run02-siglip2-seed1", "SEED_OFFSET": 1}
{"RUN_TAG": "run03-dinov3-b16", "MODEL_NAME": "vit_base_patch16_dinov3.lvd1689m", "NORM_MEAN": [0.485, 0.456, 0.406], "NORM_STD": [0.229, 0.224, 0.225]}
```

```
python run_queue.py queue.jsonl
```

The driver heals itself from temporary failures. A run that fails because the server cannot reach the model hub, a timeout, or CUDA out of memory is retried later (every 5 minutes, at most 200 times, or with doubling waits when `RETRY_BACKOFF=1`) while the queue moves on. A run whose weights are not cached is not started while the hub is unreachable. Flags from earlier temporary failures are removed at start-up. See the top of `run_queue.py` for the settings `RETRY_WAIT_MIN` and `MAX_RETRIES`.

The driver also reloads itself. Copy a new `run_queue.py` to `run_queue.py.new` and `mv` it over, and the driver restarts in place between runs if the new file passes a self check, so a training run is never interrupted. The notebook and the queue file are read fresh before each run too.

Every key except `RUN_TAG` and `_env` must be a config name of the notebook, and an unknown name stops the run. The file is read again before each run, so lines can be added while it runs. Edit it with an atomic copy (write to a temporary name, then `mv`).

## Check runs and blends

Each run saves `probs-<RUN_TAG>.npz` and `submission-<RUN_TAG>.csv`. Put them in a `runs.json` (see `runs.example.json`) and run the following.

```
python blend_check.py runs.json table
python blend_check.py runs.json check 03 --base "01,02" --ref submission-best.csv --vs 01
python blend_check.py runs.json blend "01,02,03:2" --base "01,02" --write submission-blend.csv
```

`check` prints OOF F1 per class, test class counts, how many labels differ from a reference file, and the effect of adding the run to a base blend with a bootstrap interval. Trust a blend gain only when the interval stays above zero.

## Requirements

Python 3 with torch, torchvision, timm, scikit-learn, pandas, numpy and Pillow. On a server with an older NVIDIA driver, use a CUDA build of PyTorch that matches the driver (CUDA 12.1 for driver 535).
