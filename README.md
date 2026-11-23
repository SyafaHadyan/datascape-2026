# DATASCAPE 2026

Archive of the work of team Mistersippy for the DATASCAPE 2026 Data Mining Competition on Kaggle (1 to 10 October 2026).

The task was to classify package condition images into three classes, `Aman`, `Rusak` and `Lainnya`. The metric was macro F1, with 30% of the test images scored publicly and 70% privately. The competition data is not part of this repository, see the Kaggle page of the competition.

## Approach

- Pretrained image backbones from `timm` (ConvNeXt, EfficientNet, CLIP, SigLIP, SigLIP 2, DINOv2, DINOv3, PE core, EVA02), fine-tuned with letterbox resizing, a weighted sampler, focal loss, AdamW and a cosine schedule.
- 5-fold stratified cross-validation with a fixed fold seed, the best epoch of each fold, and test predictions averaged over the folds with horizontal flip.
- More than 80 runs were queued and logged. Learning rate and the pretrained family mattered most. Out-of-fold macro F1 of the best single runs is about 0.92, and equal weight blends reach about 0.925.
- The out-of-fold score was far above the public leaderboard (a gap of 0.09 to 0.13), and a longer discussion of why is in `RUN_HISTORY.md`.

`RUN_HISTORY.md` has the full table of runs, blends, leaderboard scores and lessons.

## Layout

| Path | What it is |
|---|---|
| `RUN_HISTORY.md` | Readable log of every run, blend, public score and lesson |
| `run_queue.py`, `queue.jsonl`, `queue-log.txt` | Queue driver that runs one notebook run after another with retries, its queue and its log |
| `datascape-2026-cv-notebook-l40.ipynb`, `datascape-2026-cv-notebook-local.ipynb` | The cross-validation notebooks used for the runs |
| `scripts/` | Worker that trains one fold (`worker/`), Modal training, weight download and benchmark scripts |
| `a100-bundle/` | Scripts to run the queue on a shared A100 |
| `image-cv-template/` | Reusable template for an image classification competition with lessons learned |
| `scratchpad/` | Analysis scripts for blends, shift and buckets, run configs and server logs |
| `probabilities/` | Saved out-of-fold and test probabilities of 59 runs, see below |
| `submission/` | The notebook, requirements and write-up that were handed in |
| `submission-*.csv`, `late-probes/`, `result.zip` | Prediction files of single runs and blends, one file for each submission |
| `sub_*.csv`, `datascape-2026-sota-gpu-training*.ipynb` | Files from a teammate, see the note below |

## Saved probabilities

`probabilities/probs-run<N>-<name>` holds the predictions of 59 runs as they were saved, so blends and scores can be recomputed without training again. Runs 18 to 23, 25, 31, 34 to 73 and 77 to 81 are `.npz` files. The earliest runs (14 to 17, 24, 26, 28, 29, 30 and 32) are `.zip` files that hold the same arrays as `.npy` files.

| Array | Shape | Meaning |
|---|---|---|
| `classes` | 3 | class names, `Aman`, `Rusak`, `Lainnya` |
| `oof_filepaths` | 12,172 | training image paths in the row order of the arrays below |
| `oof_targets` | 12,172 | training labels as class indices |
| `oof_probs_flip`, `oof_probs_tta` | 12,172 x 3 | out-of-fold probabilities |
| `test_filenames` | 3,057 | test image names |
| `test_probs` | 3,057 x 3 | test probabilities averaged over the 5 fold models |

The files contain the training image paths and the training labels of the competition, exactly as saved. Compare rows by the last two parts of the path, because the folder prefix differs between Modal, Kaggle and the shared server.

## Not included

- The competition images, the data zip and the model checkpoints (over 140 GB). They are too large for GitHub and the data belongs to the organizers.
- Virtual environments and caches.

## Note on the teammate files

The `sub_*.csv` files and the `datascape-2026-sota-gpu-training*.ipynb` notebooks come from a teammate. They embed hardcoded test labels ("anchors") of unknown origin, so their leaderboard scores (0.755 to 0.845) are not comparable to the model scores in this repository. They are archived as they were created. The runs documented in `RUN_HISTORY.md` do not use them.

## Reproducing

The notebooks and workers expect the competition images in a local data folder and read their settings from environment variables such as `DS_DATA_ROOT` and `DS_OVERRIDES` (see `a100-bundle/env.sh` and the config cell of the notebook). Pretrained weights are downloaded from the Hugging Face hub through `timm`.
