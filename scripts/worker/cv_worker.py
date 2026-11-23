"""Standalone fold trainer. The notebook launches one of these per GPU as a subprocess."""
import argparse
import copy
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from img_utils import LetterboxResize, RandomDegrade

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size, cached=False, aug=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """cached=True means the images were already letterboxed to img_size by build_cache.py.
    aug is a dict of RandomDegrade settings (or None) applied to training images only.
    mean and std are the normalization constants of the pretrained backbone."""
    pre = [] if cached else [LetterboxResize(img_size)]
    degrade = [RandomDegrade(**aug)] if aug else []
    train_transform = T.Compose(pre + degrade + [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    eval_transform = T.Compose(pre + [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


class PackageDataset(Dataset):
    def __init__(self, df, transform, class_to_idx):
        self.filepaths = df["filepath"].tolist()
        self.transform = transform
        self.labels = [class_to_idx[label] for label in df["label"]]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


class CachedDataset(Dataset):
    """Reads already letterboxed images from the uint8 cache array built by build_cache.py.
    positions index the cache (they are the fold_df row numbers)."""

    def __init__(self, cache, positions, labels, transform):
        self.cache = cache
        self.positions = positions
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        img = Image.fromarray(np.asarray(self.cache[self.positions[idx]]))
        return self.transform(img), int(self.labels[idx])


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017), same as the notebook version."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=True, accum_steps=1):
    """One pass over the loader. Returns avg loss, macro F1, targets, predictions and a
    timing dict with the epoch seconds and the seconds spent waiting on the DataLoader.
    With accum_steps above 1 the optimizer steps once per accum_steps batches, so a small
    batch can reproduce a larger one (the model has no batch norm, so the gradient matches)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_targets = [], []
    data_wait = 0.0
    epoch_start = time.perf_counter()
    fetch_start = epoch_start
    use_amp = use_amp and device.type == "cuda"
    n_batches = len(loader)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, targets) in enumerate(loader):
            data_wait += time.perf_counter() - fetch_start

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                group_size = min(accum_steps, n_batches - (batch_idx // accum_steps) * accum_steps)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss / group_size).backward()
                if (batch_idx + 1) % accum_steps == 0 or batch_idx + 1 == n_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_loss += loss.item() * images.size(0)
            fetch_start = time.perf_counter()

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    timing = {"seconds": time.perf_counter() - epoch_start, "data_wait": data_wait}
    return avg_loss, macro_f1, all_targets, all_preds, timing


def predict_proba_tta(model, loader, scales, device, num_classes, use_amp=True):
    """Mean softmax probabilities over the given scales and horizontal flip."""
    n_passes = len(scales) * 2
    all_probs = []
    use_amp = use_amp and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            probs_sum = torch.zeros(images.size(0), num_classes, device=device)
            for scale in scales:
                scaled = images if scale == images.shape[-1] else F.interpolate(
                    images, size=(scale, scale), mode="bilinear", align_corners=False
                )
                for variant in (scaled, torch.flip(scaled, dims=[3])):
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(variant)
                    probs_sum += torch.softmax(logits.float(), dim=1)
            all_probs.append((probs_sum / n_passes).cpu())
    return torch.cat(all_probs, dim=0)


def train_one_fold(fold_idx, train_idx, val_idx, fold_df, cfg, out_dir, device,
                   is_first_fold, n_folds_for_worker):
    # Reseed per fold so a fold gives the same result whichever worker or order runs it.
    seed = cfg["SEED"] + fold_idx + cfg.get("SEED_OFFSET", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    classes = cfg["CLASSES"]
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    epochs = cfg["EPOCHS_TO_RUN"]
    use_amp = cfg.get("USE_AMP", True)
    accum_steps = cfg.get("ACCUM_STEPS", 1)
    # Every log line carries this tag so lines from the two GPU workers stay readable.
    tag = f"[fold {fold_idx + 1}/{cfg['n_folds_total']}]"
    print(
        f"{tag} started, batch {cfg['BATCH_SIZE']} x accumulation {accum_steps} = {cfg['BATCH_SIZE'] * accum_steps}, "
        f"mixed precision {'on' if use_amp and device.type == 'cuda' else 'off'}, seed offset {cfg.get('SEED_OFFSET', 0)}",
        flush=True,
    )

    cache = np.load(cfg["CACHE_PATH"], mmap_mode="r") if cfg.get("CACHE_PATH") else None
    train_transform, eval_transform = build_transforms(
        cfg["IMG_SIZE"], cached=cache is not None, aug=cfg.get("SHIFT_AUG"),
        mean=cfg.get("NORM_MEAN", IMAGENET_MEAN), std=cfg.get("NORM_STD", IMAGENET_STD),
    )
    fold_train_df = fold_df.iloc[train_idx].reset_index(drop=True)
    fold_val_df = fold_df.iloc[val_idx].reset_index(drop=True)
    if cache is not None:
        all_labels = np.array([class_to_idx[label] for label in fold_df["label"]])
        train_dataset = CachedDataset(cache, train_idx, all_labels[train_idx], train_transform)
        val_dataset = CachedDataset(cache, val_idx, all_labels[val_idx], eval_transform)
    else:
        train_dataset = PackageDataset(fold_train_df, train_transform, class_to_idx)
        val_dataset = PackageDataset(fold_val_df, eval_transform, class_to_idx)
    print(
        f"{tag} data from {'image cache' if cache is not None else 'original files'}, "
        f"shift augmentation {'on' if cfg.get('SHIFT_AUG') else 'off'}",
        flush=True,
    )

    class_counts = fold_train_df["label"].value_counts().reindex(classes)
    class_weights = (class_counts.sum() / (num_classes * class_counts)).values
    sample_weights = fold_train_df["label"].map(dict(zip(classes, class_weights))).values

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["BATCH_SIZE"],
        sampler=sampler,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    model = timm.create_model(
        cfg["MODEL_NAME"], pretrained=True, num_classes=num_classes,
        drop_path_rate=cfg["DROP_PATH_RATE"], **cfg.get("MODEL_KWARGS", {}),
    ).to(device)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    if cfg["USE_FOCAL_LOSS"]:
        alpha = class_weights_tensor if cfg["FOCAL_USE_ALPHA"] else None
        criterion = FocalLoss(alpha=alpha, gamma=cfg["FOCAL_GAMMA"], label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    layer_decay = cfg.get("LAYER_DECAY")
    if layer_decay:
        # Layer-wise LR decay for ViTs, earlier blocks get a smaller LR. timm only stores the
        # decay as a per-group lr_scale that its own schedulers apply, so set each group's lr here
        # for torch's scheduler, which keeps the ratios between groups. Bias and norm weights get
        # no weight decay in this branch.
        optimizer = create_optimizer_v2(
            model, opt="adamw", lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"], layer_decay=layer_decay
        )
        for group in optimizer.param_groups:
            group["lr"] = cfg["LR"] * group.get("lr_scale", 1.0)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and use_amp))

    best_val_f1 = 0.0
    best_state = None
    best_epoch = 0
    best_targets, best_preds = None, None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_f1, _, _, train_timing = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp, accum_steps
        )
        val_loss, val_f1, val_targets, val_preds, val_timing = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        epoch_seconds = train_timing["seconds"] + val_timing["seconds"]
        wait_pct = 100 * (train_timing["data_wait"] + val_timing["data_wait"]) / epoch_seconds
        print(
            f"{tag} Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | "
            f"{epoch_seconds:.0f}s (data wait {wait_pct:.0f}%)",
            flush=True,
        )

        limit = cfg["FIRST_EPOCH_TIME_LIMIT_S"]
        if is_first_fold and epoch == 1 and limit and epoch_seconds > limit:
            raise RuntimeError(
                f"First epoch took {epoch_seconds:.0f}s, over the {limit}s guard. "
                f"Aborting to protect the GPU quota (set FIRST_EPOCH_TIME_LIMIT_S = None to disable)."
            )
        if is_first_fold and epoch == 2:
            est_hours = epoch_seconds * epochs * n_folds_for_worker / 3600
            print(
                f"{tag} Rough estimate for this worker at this speed, about {est_hours:.1f} hours of wall time",
                flush=True,
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_targets, best_preds = val_targets, val_preds
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["EARLY_STOP_PATIENCE"]:
                print(
                    f"{tag} Early stopping, no improvement for {cfg['EARLY_STOP_PATIENCE']} epochs",
                    flush=True,
                )
                break

    print(f"{tag} best val macro F1 {best_val_f1:.4f} at epoch {best_epoch}", flush=True)

    model.load_state_dict(best_state)
    probs_flip = predict_proba_tta(model, val_loader, [cfg["IMG_SIZE"]], device, num_classes, use_amp)
    probs_tta = predict_proba_tta(model, val_loader, cfg["TTA_SCALES"], device, num_classes, use_amp)

    torch.save({k: v.detach().cpu() for k, v in best_state.items()}, out_dir / f"fold{fold_idx}_ckpt.pt")
    np.savez_compressed(
        out_dir / f"fold{fold_idx}_oof.npz",
        val_idx=np.asarray(val_idx),
        targets=np.asarray(best_targets),
        preds_single=np.asarray(best_preds),
        probs_flip=probs_flip.numpy(),
        probs_tta=probs_tta.numpy(),
    )
    # The summary file is written last and doubles as the "this fold finished" marker.
    (out_dir / f"fold{fold_idx}_summary.json").write_text(json.dumps({
        "fold": fold_idx,
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
    }))

    del model, optimizer, scheduler, scaler, best_state
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    cfg = json.loads((work_dir / "cfg.json").read_text())
    fold_df = pd.read_csv(work_dir / "fold_df.csv")
    splits = np.load(work_dir / "splits.npz")
    folds = [int(f) for f in args.folds.split(",")]

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    print(f"Worker started on {gpu_name} for folds {[f + 1 for f in folds]}", flush=True)

    for i, fold_idx in enumerate(folds):
        train_one_fold(
            fold_idx,
            splits[f"train_{fold_idx}"],
            splits[f"val_{fold_idx}"],
            fold_df,
            cfg,
            work_dir,
            device,
            is_first_fold=(i == 0),
            n_folds_for_worker=len(folds),
        )
    print("Worker finished", flush=True)


if __name__ == "__main__":
    main()
