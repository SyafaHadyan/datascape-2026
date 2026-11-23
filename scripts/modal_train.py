"""Run the 5-fold training pipeline on Modal, one GPU container per fold.

The same cv_worker.py that the notebook writes is used here (copied into scripts/worker).
Every fold trains in its own T4 container at the same time, so a full run takes about as
long as one fold. Nothing here is run automatically, you start each step yourself.

One-time setup on your own computer.
  pip install modal numpy
  modal setup

Steps.
  1. Upload the dataset and the fold order (about 3 GB, once).
       modal run scripts/modal_train.py::upload
  2. Train all folds in parallel and build the submission.
       modal run scripts/modal_train.py::train --run-tag run18-effnetb0-nsjft --model-name tf_efficientnet_b0.ns_jft_in1k
     Useful options are --shift-aug off, --img-size 288, --n-folds 6, --epochs 40, --lr 0.0003.
     ViTs such as CLIP and SigLIP also need --layer-decay 0.75, --norm clip or --norm half, and
     --multi-scale-tta off (a ViT rejects any size other than its own). For example run 31 would be
       scripts/train.sh run31-siglip-b16-256 --model-name vit_base_patch16_siglip_256.webli --img-size 256 \\
         --shift-aug off --epochs 30 --lr 3e-5 --layer-decay 0.75 --norm half --multi-scale-tta off
     Keep the terminal open until it finishes.
     To see the run tag as the app name in the Modal dashboard, use the wrapper, which passes
     the tag to modal run -n as well.
       scripts/train.sh run18-effnetb0-nsjft --model-name tf_efficientnet_b0.ns_jft_in1k
  3. Download the results from the runs volume.
       modal volume get datascape-runs run18-effnetb0-nsjft/submission.csv .
       modal volume get datascape-runs run18-effnetb0-nsjft/probs-run18-effnetb0-nsjft.npz .

The fold order comes from an earlier run's saved out-of-fold file, so the splits match that
run exactly and the scores can be compared directly. Without it the images are sorted by name.
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

# These paths are overridable only so the functions can be tested on a laptop.
DATA_DIR = Path(os.environ.get("DS_DATA_DIR", "/data"))
RUNS_DIR = Path(os.environ.get("DS_RUNS_DIR", "/runs"))
WORK_DIR = Path(os.environ.get("DS_WORK_DIR", "/tmp/work"))
APP_DIR = Path(os.environ.get("DS_APP_DIR", "/app"))

# Two persistent folders on Modal's side. One holds the dataset and the image cache, the
# other holds each run's checkpoints, predictions and submission.
data_vol = modal.Volume.from_name("datascape-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("datascape-runs", create_if_missing=True)
VOLUMES = {str(DATA_DIR): data_vol, str(RUNS_DIR): runs_vol}

# The container is described in code. Start from a slim Debian with Python 3.12, install the
# packages the worker imports, and copy the worker files in. This replaces writing a Dockerfile.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "timm", "scikit-learn", "pandas", "numpy", "pillow")
    .add_local_dir(HERE / "worker", remote_path="/app")
)
app = modal.App("datascape-2026", image=image)

# Same settings as the notebook config. Anything can be overridden from the command line.
DEFAULT_CFG = dict(
    SEED=42, CLASSES=["Aman", "Lainnya", "Rusak"], IMG_SIZE=224, EPOCHS_TO_RUN=50,
    BATCH_SIZE=32, NUM_WORKERS=5, MODEL_NAME="efficientnet_b0", DROP_PATH_RATE=0.0,
    USE_FOCAL_LOSS=True, FOCAL_USE_ALPHA=False, FOCAL_GAMMA=2.0, LR=3e-4,
    WEIGHT_DECAY=1e-2, EARLY_STOP_PATIENCE=30, TTA_SCALES=[192, 224, 256],
    FIRST_EPOCH_TIME_LIMIT_S=None, N_FOLDS=5, USE_SHIFT_AUG=True,
)
SHIFT_AUG = dict(
    p_down=0.3, down_scale=[0.4, 0.8], p_blur=0.2, blur_radius=[0.3, 1.2],
    p_noise=0.2, noise_std=[2.0, 8.0], p_jpeg=0.5, jpeg_quality=[25, 95],
)


def _commit(volume):
    # Skipped in local tests, where no volume is mounted.
    if os.environ.get("DS_LOCAL_TEST") != "1":
        volume.commit()


def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(cpu=8.0, memory=16384, timeout=3600, volumes=VOLUMES)
def prepare(run_tag: str, overrides: dict) -> dict:
    """Build the image list, the folds, the image cache and the config for one run."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    cfg = {**DEFAULT_CFG, **overrides}
    classes = cfg["CLASSES"]
    train_root = DATA_DIR / "Train_Set"
    order_file = DATA_DIR / "fold_order.txt"

    if order_file.exists():
        rel = [line for line in order_file.read_text().split("\n") if line]
        print(f"Using the saved fold order, {len(rel)} images")
    else:
        rel = sorted(f"{c}/{p.name}" for c in classes for p in (train_root / c).iterdir() if p.is_file())
        print(f"No saved fold order found, using {len(rel)} images sorted by name")
    df = pd.DataFrame({"filepath": [str(train_root / r) for r in rel], "label": [r.split("/")[0] for r in rel]})

    # Identical files that carry different labels cannot both be right, so drop them.
    with ThreadPoolExecutor(max_workers=16) as pool:
        df["md5"] = list(pool.map(_file_md5, df["filepath"]))
    by_hash = df.groupby("md5")["label"].nunique()
    conflict = df["md5"].isin(by_hash.index[by_hash > 1])
    print(f"Dropping {int(conflict.sum())} images with conflicting duplicate labels")
    df = df[~conflict].reset_index(drop=True)
    print(df["label"].value_counts().to_dict())

    n_folds = int(cfg["N_FOLDS"])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["SEED"])
    splits = list(skf.split(df, df["label"]))

    # One cache file for all folds, built once and reused by later runs on the same images.
    paths_text = "\n".join(df["filepath"])
    key = hashlib.md5(f"{cfg['IMG_SIZE']}\n{paths_text}".encode()).hexdigest()[:10]
    cache_src = DATA_DIR / "cache" / f"imgcache_{cfg['IMG_SIZE']}_{key}.npy"
    if cache_src.exists():
        print("Using the existing image cache", cache_src)
    else:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        (WORK_DIR / "cache_paths.txt").write_text(paths_text)
        tmp_cache = WORK_DIR / "cache.npy"
        print("Building the image cache")
        subprocess.run(
            [sys.executable, str(APP_DIR / "build_cache.py"), "--paths-file", str(WORK_DIR / "cache_paths.txt"),
             "--out", str(tmp_cache), "--size", str(cfg["IMG_SIZE"]), "--procs", str(os.cpu_count() or 4)],
            check=True, cwd=APP_DIR,
        )
        cache_src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_cache, cache_src)
        tmp_cache.unlink()
        _commit(data_vol)

    run_dir = RUNS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["SHIFT_AUG"] = SHIFT_AUG if cfg.pop("USE_SHIFT_AUG") else None
    cfg["n_folds_total"] = n_folds
    cfg["CACHE_SRC"] = str(cache_src)
    (run_dir / "cfg.json").write_text(json.dumps(cfg))
    df.drop(columns="md5").to_csv(run_dir / "fold_df.csv", index=False)
    arrays = {}
    for i, (tr, va) in enumerate(splits):
        arrays[f"train_{i}"], arrays[f"val_{i}"] = tr, va
    np.savez(run_dir / "splits.npz", **arrays)
    _commit(runs_vol)
    return {"images": len(df), "folds": n_folds, "model": cfg["MODEL_NAME"], "shift_aug": cfg["SHIFT_AUG"] is not None}


@app.function(gpu="T4", cpu=6.0, memory=16384, timeout=3 * 3600, volumes=VOLUMES)
def train_fold(run_tag: str, fold: int) -> dict:
    """Train one fold on its own T4 and save its checkpoint and out-of-fold predictions."""
    run_dir = RUNS_DIR / run_tag
    work = WORK_DIR / f"{run_tag}-fold{fold}"
    work.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "cfg.json").read_text())
    for name in ("fold_df.csv", "splits.npz"):
        shutil.copy(run_dir / name, work / name)
    # Read the cache from local disk, a network volume is too slow for random access.
    shutil.copy(cfg["CACHE_SRC"], work / "cache.npy")
    cfg["CACHE_PATH"] = str(work / "cache.npy")
    (work / "cfg.json").write_text(json.dumps(cfg))

    proc = subprocess.Popen(
        [sys.executable, str(APP_DIR / "cv_worker.py"), "--work-dir", str(work), "--folds", str(fold)],
        cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"Fold {fold + 1} failed, see the log above")
    for name in (f"fold{fold}_ckpt.pt", f"fold{fold}_oof.npz", f"fold{fold}_summary.json"):
        shutil.copy(work / name, run_dir / name)
    _commit(runs_vol)
    return json.loads((run_dir / f"fold{fold}_summary.json").read_text())


@app.function(gpu="T4", cpu=4.0, memory=16384, timeout=3600, volumes=VOLUMES)
def finish(run_tag: str) -> dict:
    """Pool the out-of-fold predictions, predict the test set with all folds, and write the files."""
    import numpy as np
    import pandas as pd
    import torch
    import timm
    from sklearn.metrics import classification_report, f1_score
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(APP_DIR))
    import cv_worker as cw

    run_dir = RUNS_DIR / run_tag
    cfg = json.loads((run_dir / "cfg.json").read_text())
    fold_df = pd.read_csv(run_dir / "fold_df.csv")
    classes = cfg["CLASSES"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_folds, n_classes, n = cfg["n_folds_total"], len(classes), len(fold_df)

    targets = np.full(n, -1, dtype=int)
    preds_single = np.full(n, -1, dtype=int)
    probs_flip = np.zeros((n, n_classes), dtype=np.float32)
    probs_tta = np.zeros((n, n_classes), dtype=np.float32)
    for fold in range(n_folds):
        oof = np.load(run_dir / f"fold{fold}_oof.npz")
        idx = oof["val_idx"]
        targets[idx], preds_single[idx] = oof["targets"], oof["preds_single"]
        probs_flip[idx], probs_tta[idx] = oof["probs_flip"], oof["probs_tta"]
    assert (targets >= 0).all(), "Some images never got an out-of-fold prediction"

    f1_none = f1_score(targets, preds_single, average="macro")
    f1_flip = f1_score(targets, probs_flip.argmax(1), average="macro")
    f1_tta = f1_score(targets, probs_tta.argmax(1), average="macro")
    print(f"OOF macro F1 no TTA {f1_none:.4f} | single scale + flip {f1_flip:.4f} | multi-scale + flip {f1_tta:.4f}")
    print(classification_report(targets, probs_flip.argmax(1), target_names=classes, digits=4))
    scales = cfg["TTA_SCALES"] if f1_tta > f1_flip else [cfg["IMG_SIZE"]]
    print("Test-time scales", scales)

    test_dir = DATA_DIR / "Test_Set"
    test_names = sorted(p.name for p in test_dir.iterdir() if p.is_file())
    test_df = pd.DataFrame({"filepath": [str(test_dir / f) for f in test_names], "label": classes[0]})
    # The test images need the same normalization constants the model was trained with.
    _, eval_tf = cw.build_transforms(
        cfg["IMG_SIZE"], cached=False, aug=None,
        mean=cfg.get("NORM_MEAN", cw.IMAGENET_MEAN), std=cfg.get("NORM_STD", cw.IMAGENET_STD),
    )
    loader = DataLoader(cw.PackageDataset(test_df, eval_tf, class_to_idx), batch_size=64, shuffle=False,
                        num_workers=int(cfg["NUM_WORKERS"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_probs = np.zeros((len(test_names), n_classes), dtype=np.float32)
    for fold in range(n_folds):
        model = timm.create_model(
            cfg["MODEL_NAME"], pretrained=False, num_classes=n_classes, **cfg.get("MODEL_KWARGS", {})
        )
        model.load_state_dict(torch.load(run_dir / f"fold{fold}_ckpt.pt", map_location="cpu"))
        model.to(device)
        test_probs += cw.predict_proba_tta(model, loader, scales, device, n_classes).numpy() / n_folds
        del model

    labels = [classes[i] for i in test_probs.argmax(1)]
    pd.DataFrame({"filename": test_names, "label": labels}).to_csv(run_dir / "submission.csv", index=False)
    np.savez_compressed(
        run_dir / f"probs-{run_tag}.npz",
        classes=np.array(classes), oof_filepaths=fold_df["filepath"].values, oof_targets=targets,
        oof_probs_flip=probs_flip, oof_probs_tta=probs_tta, test_filenames=np.array(test_names),
        test_probs=test_probs, test_scales=np.array(scales),
    )
    _commit(runs_vol)
    counts = {c: labels.count(c) for c in classes}
    print("Test label counts", counts)
    return {"oof_f1_no_tta": f1_none, "oof_f1_flip": f1_flip, "oof_f1_multiscale": f1_tta, "test_counts": counts}


def _tta_scales(img_size: int) -> list:
    """Three test-time scales around the training size, as multiples of 32. 224 gives 192, 224 and 256."""
    return sorted({max(32, round(img_size * f / 32) * 32) for f in (6 / 7, 1.0, 8 / 7)})


# Normalization constants by name. ImageNet is what convnext, efficientnet and DINOv2 use.
NORMS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    "half": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


def _build_overrides(model_name, shift_aug, n_folds, epochs, lr, batch_size, num_workers, drop_path,
                     img_size, layer_decay, norm, multi_scale_tta, model_kwargs):
    """Turn the command line options into config overrides. Only options that differ from the
    defaults add keys, so a run with the old options builds exactly the same config as before."""
    if norm not in NORMS:
        raise ValueError(f"--norm must be one of {sorted(NORMS)}, got {norm}")
    if multi_scale_tta not in ("on", "off"):
        raise ValueError("--multi-scale-tta must be on or off")
    overrides = dict(
        MODEL_NAME=model_name, USE_SHIFT_AUG=(shift_aug == "on"), N_FOLDS=n_folds, EPOCHS_TO_RUN=epochs,
        LR=lr, BATCH_SIZE=batch_size, NUM_WORKERS=num_workers, DROP_PATH_RATE=drop_path,
        IMG_SIZE=img_size,
        TTA_SCALES=_tta_scales(img_size) if multi_scale_tta == "on" else [img_size],
    )
    if layer_decay:
        overrides["LAYER_DECAY"] = layer_decay
    if norm != "imagenet":
        overrides["NORM_MEAN"], overrides["NORM_STD"] = NORMS[norm]
    kwargs = json.loads(model_kwargs)
    if kwargs:
        overrides["MODEL_KWARGS"] = kwargs
    return overrides


def _find(path: str) -> Path:
    """Resolve a path from the current folder first, then from the project root, so it works from anywhere."""
    p = Path(path).expanduser()
    if p.exists():
        return p
    alt = HERE.parent / path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find {path} (looked in {Path.cwd()} and {HERE.parent})")


def _order_from_zip(zip_path: Path) -> str:
    """Read the fold order (relative image paths) from an earlier run's probabilities file."""
    import numpy as np

    with zipfile.ZipFile(zip_path) as z, z.open("oof_filepaths.npy") as f:
        paths = np.load(io.BytesIO(f.read()), allow_pickle=True)
    return "\n".join("/".join(str(p).split("/")[-2:]) for p in paths) + "\n"


@app.local_entrypoint()
def upload(archive_dir: str = "archive", order_zip: str = "archive/probs-run17-effnetb0-dedup.zip",
           what: str = "all"):
    """Copy the dataset and the fold order to the data volume. Use --what order to refresh only the order.

    Use --order-zip none to skip the fold order, in which case images are sorted by name.
    """
    archive = _find(archive_dir)
    # Look everything up before the upload starts, so a wrong path fails right away.
    folders = [_find(str(archive / sub)) for sub in ("Train_Set", "Test_Set")] if what == "all" else []
    order = "" if order_zip == "none" else _order_from_zip(_find(order_zip))
    with data_vol.batch_upload(force=True) as batch:
        for folder in folders:
            print("Uploading", folder)
            batch.put_directory(folder, f"/{folder.name}")
        if order:
            print(f"Uploading the fold order from {order_zip}, {order.count(chr(10))} images")
            batch.put_file(io.BytesIO(order.encode()), "/fold_order.txt")
        else:
            print("No fold order, images will be sorted by name")
    print("Upload done")


@app.local_entrypoint()
def train(run_tag: str, model_name: str = "efficientnet_b0", shift_aug: str = "on", n_folds: int = 5,
          epochs: int = 50, lr: float = 3e-4, batch_size: int = 32, num_workers: int = 5,
          drop_path: float = 0.0, img_size: int = 224, layer_decay: float = 0.0, norm: str = "imagenet",
          multi_scale_tta: str = "on", model_kwargs: str = "{}"):
    """Prepare the data, train every fold in parallel, then build the submission.

    A new --img-size builds its own image cache on the first run, then reuses it.
    The test-time scales follow the image size, so 288 tests at 256, 288 and 320, unless
    --multi-scale-tta is off, which tests only at the training size (needed for ViTs).
    --layer-decay turns on layer-wise LR decay (0 means off), --norm is imagenet, clip or half
    (mean and std 0.5, used by SigLIP), and --model-kwargs is a JSON dict for timm.create_model.
    """
    if img_size % 32:
        print(f"Warning, img_size {img_size} is not a multiple of 32, 224 or 256 or 288 or 320 are safer")
    overrides = _build_overrides(
        model_name, shift_aug, n_folds, epochs, lr, batch_size, num_workers, drop_path, img_size,
        layer_decay, norm, multi_scale_tta, model_kwargs,
    )
    start = time.time()
    print("Prepare", prepare.remote(run_tag, overrides))
    fold_results = list(train_fold.starmap([(run_tag, i) for i in range(n_folds)]))
    for r in fold_results:
        print(f"  fold {r['fold'] + 1}: best val macro F1 {r['best_val_f1']:.4f} at epoch {r['best_epoch']}")
    print("Finish", finish.remote(run_tag))
    print(f"Done in {(time.time() - start) / 60:.0f} minutes. Download with")
    print(f"  modal volume get datascape-runs {run_tag}/submission.csv .")
    print(f"  modal volume get datascape-runs {run_tag}/probs-{run_tag}.npz .")
