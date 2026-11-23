# Legacy notebook training helpers, removed from the Colab notebook on 2026-10-03.
# The live pipeline trains inside cv_worker.py, which has its own copies of FocalLoss and run_epoch.
# Kept only for the mixup/cutmix code (USE_MIXUP, MIXUP_ALPHA, CUTMIX_ALPHA, MIXUP_PROB) that failed twice, see project memory.

def rand_bbox(size, lam):
    W, H = size[3], size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(images, targets):
    if not USE_MIXUP or np.random.rand() > MIXUP_PROB:
        return images, targets, targets, 1.0

    perm = torch.randperm(images.size(0), device=images.device)
    targets_a, targets_b = targets, targets[perm]

    if np.random.rand() < 0.5:
        lam = float(np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA))
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bby1:bby2, bbx1:bbx2] = images[perm, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    else:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        images = lam * images + (1.0 - lam) * images[perm]

    return images, targets_a, targets_b, lam


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017): down-weights easy/already-correct examples so
    gradient concentrates on hard cases, which tends to help minority classes (Lainnya)
    more directly than class-weighting alone. `alpha` reuses the same per-class weights
    as the class-weighted CrossEntropyLoss it replaces."""

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


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    """Run one pass over the loader. Returns avg loss, macro F1, targets, predictions
    and a timing dict with the epoch duration and the seconds spent waiting for the
    next batch from the DataLoader (a high wait share means data loading, not the GPU,
    is the bottleneck)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_targets = [], []
    data_wait = 0.0
    epoch_start = time.perf_counter()
    fetch_start = epoch_start

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            data_wait += time.perf_counter() - fetch_start

            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if is_train:
                images, targets_a, targets_b, lam = mixup_cutmix(images, targets)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    outputs = model(images)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets_a.cpu().numpy())
            else:
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            total_loss += loss.item() * images.size(0)
            fetch_start = time.perf_counter()

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    timing = {"seconds": time.perf_counter() - epoch_start, "data_wait": data_wait}
    return avg_loss, macro_f1, all_targets, all_preds, timing
