# Adversarial validation cells exported from the Colab notebook (2026-10-03).
# They need the notebook state (CLASSES, DATA_ROOT, eval_transform, MODEL_NAME, DEVICE) so run them inside the notebook, not standalone.

# %% runtime probe (cell vMf_wKK80ftn)
# Runtime probe (adversarial validation section, not part of the training pipeline)
import os, subprocess, shutil
print("cwd", os.getcwd())
root = "/root/.cache/kagglehub/competitions/datascape-2026-data-mining-competition"
print("data present", os.path.isdir(root), os.path.isdir(root + "/Train_Set") if os.path.isdir(root) else None)
print("cpu", os.cpu_count(), "| free disk GB", round(shutil.disk_usage("/content").free / 1e9, 1))
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "no GPU")

# %% step 1 feature extraction (cell ueeUbVk02hBF)
# Adversarial validation (train vs test), not part of the training pipeline.
# Step 1 extracts frozen pretrained features from letterboxed 224 images.
# Needs the config, download, class and transform cells above to have run.
adv_train = sorted(
    (str(p) for c in CLASSES for p in (TRAIN_DIR / c).iterdir() if p.is_file()),
    key=lambda s: (Path(s).parent.name, Path(s).name),
)
adv_test = sorted((str(p) for p in TEST_DIR.iterdir() if p.is_file()), key=lambda s: Path(s).name)
adv_paths = adv_train + adv_test
print("train", len(adv_train), "test", len(adv_test))


class AdvDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_transform(Image.open(self.paths[i]).convert("RGB"))


adv_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(DEVICE).eval()
adv_loader = DataLoader(AdvDataset(adv_paths), batch_size=64, num_workers=2, pin_memory=True)
adv_start = time.time()
adv_chunks = []
with torch.no_grad():
    for i, x in enumerate(adv_loader):
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            f = adv_model(x.to(DEVICE, non_blocking=True))
        adv_chunks.append(f.float().cpu().numpy())
        if i % 50 == 0:
            print(f"  batch {i}/{len(adv_loader)} {time.time() - adv_start:.0f}s")
adv_X = np.concatenate(adv_chunks)
np.save("/content/adv_feats.npy", adv_X)
print("features", adv_X.shape, f"{time.time() - adv_start:.0f}s")

# %% step 2 train vs test classifier (cell ipHb5ItP2hFj)
# Adversarial validation step 2, cross-validated train-vs-test classifier on the features.
import base64
import zlib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

adv_y = np.array([0] * len(adv_train) + [1] * len(adv_test))
adv_clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
adv_p = cross_val_predict(
    adv_clf, adv_X, adv_y,
    cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba",
)[:, 1]
print(f"adversarial AUC (frozen {MODEL_NAME} features) {roc_auc_score(adv_y, adv_p):.3f}")

train_cls = np.array([Path(p).parent.name for p in adv_train])
for c in CLASSES:
    print(f"  mean test-likeness of train {c}: {adv_p[:len(adv_train)][train_cls == c].mean():.3f}")
print(f"  mean test-likeness of test images {adv_p[len(adv_train):].mean():.3f}")

# Compact dump so the per-image scores can be analysed outside Colab (float16, zlib, base64).
# Order is train sorted by (class, filename) followed by test sorted by filename.
blob = base64.b64encode(zlib.compress(adv_p.astype(np.float16).tobytes())).decode()
print("BLOB_START", len(adv_p), adv_train[0].split("/")[-1], adv_test[0].split("/")[-1])
print(blob)
print("BLOB_END")

# %% step 3 score run 14 OOF on test-like images (cell HhU_piys3a2k)
# Adversarial validation step 3, score run 14's out-of-fold predictions on the test-like training images.
# The payload holds run 14's predicted class per training image (2 bits each), ordered by (class, filename).
import base64
import zlib
from sklearn.metrics import f1_score

B64 = "eNp9VkuLJDcMdoveRetTD+zkkJNSJFD4V0wgDTlmQnsPew/kZwhDwNQtkECOpk7GvzKyXQ/Xo9c9Uw9blvR9erh8SunP75RSF3VVTAoUKpIHVvmmWMsdQNYxKgSvYpCVeYBqByrgMsPyOA/aSvJuATcaiNo9jeK8s1w1LOqWrZSf6NPBIZlmWAzDtAFw3p996fcuzD4gNMYhQ2J1HBt0WQBpwYh7fg5bXaMls96ROhiBVYY2tGJDJ+1BHEjkiolWtVyh0Ub6ks0VKV4Z3usmXqm5FcFpwxqRLXDhnjPtQk71AHe50FBWrv0GfoPkesyL5eWU7B2fuLVZ//J/d6BuymSl2xU3aYStfVh1avnBExfrAvGT1fMZdeaxem4jy9GxQCehfp9B5+myybs1iocQ8LeyrrSQYyvg50j5ZIL2CxP+sPUW6CxbYUcVPa3JxfG3sxoktSnVfSKeKw151g1b9XP+09P9SKfMHE0APwnyVkEhquN+bqFy7/rVLsgsrDZo4ye/qR9KzUDbDPdlwuojNBl6m1vtRa3VPmUCSEO+lWm9tI1GZfELslCvwNFYupHrnBppgYy1R9cnVMwDRBWEbLfpF4S4EiHQmpLFW24E1FBYhSrJ8I2OXYYOT9pO7nNr8Zl9xR3yQz81MinVfpTjOenqufM5XPirRKwvYSUZII2V6/GGa4+TBuzo4ycuuVRcNAU675v47Gzuy/60SMa8KcxlyEsFtj0Gy1FQ0wuvqpLdUNuhvy4cYzFdjJGeFIm/ExkBay9+vd+tfbe/vX54fP7D2pf7w1p7/+Vdrvb3n+T27+sXkbCvD3v7Yuu42zxj/84X2XP/YG3eZX9+/fr5/ab+utvLA18elzel3l/s45Ht2H9iioXlOsz6mGLajLh5HpMepqmuXR/KdRQ9Y34xeaKs6MFn/THbGMat6qExX9VEjfULQi4OvfJIEvFrcBI56tgjG0l61iE4LAcmO+8oau88Jhi6AL0b0DMEP3LH2jlXv8xyDDvoJYV8iAqisO1hjCR5hJ0aohSB4JLdo0HqMR8T4HUMjp3E7XvFwVFy4oEZ44Rnhm/YpCBYKriQQllzLrkFo9FVWKXBVaE8IJod1yXrEwi3o3ACEzlcNS/CnAYjb8NC3iC/rjmuiDloHQfxBIyKKdYA5wdzjLfYFF3FLycczEBS9RnqPUw+m1Dh5clRhxLuFKPpi+4SdFNhxwn7UPFKTkRTdERZcbIPps4klnNixI66PmLfI4cfc3fumOR4AwHbJ93nnEr9KAnA4C8ggfMu4xr/y0z4Mchnem/SKMR1acQgyN0wGCdf8LHH/wGZPd0V"
raw = zlib.decompress(base64.b64decode(B64))
assert zlib.adler32(raw) == 2570968341, "payload corrupted in copy"
packed = np.frombuffer(raw, dtype=np.uint8)
pred = np.stack([packed & 3, (packed >> 2) & 3, (packed >> 4) & 3, (packed >> 6) & 3], axis=1).reshape(-1)[: len(adv_train)].astype(int)
yt = np.array([CLASSES.index(Path(p).parent.name) for p in adv_train])
pt = adv_p[: len(adv_train)]
mf1 = lambda m, w=None: f1_score(yt[m], pred[m], average="macro", sample_weight=None if w is None else w[m])
everyone = np.ones(len(yt), bool)
print(f"run 14 OOF macro F1 unweighted {mf1(everyone):.4f}")

prior = len(adv_test) / len(adv_train)
w = np.clip(pt / (1 - pt), 0, 50) / prior
print(f"importance weighted by deep-feature test-likeness {mf1(everyone, w):.4f}, effective n {w.sum() ** 2 / (w ** 2).sum():.0f} of {len(w)}")
for frac in (0.5, 0.25, 0.1):
    m = pt >= np.quantile(pt, 1 - frac)
    per = f1_score(yt[m], pred[m], average=None, labels=[0, 1, 2], zero_division=0)
    print(f"top {int(frac * 100):3d}% test-like n={m.sum():5d} macro F1 {mf1(m):.4f} acc {(yt[m] == pred[m]).mean():.4f} "
          f"class mix {np.round(np.bincount(yt[m], minlength=3) / m.sum(), 3)} per-class F1 {np.round(per, 3)}")
m = pt < np.quantile(pt, 0.5)
print(f"bottom 50% n={m.sum()} macro F1 {mf1(m):.4f} acc {(yt[m] == pred[m]).mean():.4f}")
print("weighted class mix", np.round([w[yt == c].sum() for c in range(3)] / w.sum(), 3), "raw", np.round(np.bincount(yt) / len(yt), 3))
tp_ = adv_p[len(adv_train):]
print(f"test images with test-likeness>0.8 {(tp_ > 0.8).mean():.3f}, <0.2 {(tp_ < 0.2).mean():.3f}")
lain = yt == 1
print(f"Lainnya train images test-likeness mean {pt[lain].mean():.3f}, share above test median ({np.median(tp_):.2f}) {(pt[lain] > np.median(tp_)).mean():.3f}")

# %% step 4 bootstrap and sensitivity (cell ZZ4tGKnh3fZU)
# Adversarial validation step 4, how stable is the test-likeness weighted score, and which class drives it.
per_w = f1_score(yt, pred, average=None, labels=[0, 1, 2], sample_weight=w)
print("weighted per-class F1 (Aman, Lainnya, Rusak)", np.round(per_w, 3))
print("effective n per class", [round(w[yt == c].sum() ** 2 / (w[yt == c] ** 2).sum()) for c in range(3)])

rng = np.random.default_rng(0)
boots = []
for _ in range(300):
    idx = rng.integers(0, len(yt), len(yt))
    boots.append(f1_score(yt[idx], pred[idx], average="macro", sample_weight=w[idx]))
print(f"weighted macro F1 bootstrap mean {np.mean(boots):.3f}, 95% interval {np.percentile(boots, 2.5):.3f} to {np.percentile(boots, 97.5):.3f}")

# Sensitivity to the weight clip (50 above).
for clip in (5, 10, 20, 50, 200):
    wc = np.clip(pt / (1 - pt), 0, clip) / prior
    print(f"clip {clip:3d} weighted macro F1 {f1_score(yt, pred, average='macro', sample_weight=wc):.4f}  effective n {wc.sum() ** 2 / (wc ** 2).sum():.0f}")

# Where the Lainnya mistakes sit in the test-like region (weighted confusion).
from sklearn.metrics import confusion_matrix
print("weighted confusion (rows true Aman, Lainnya, Rusak, cols predicted)")
print(np.round(confusion_matrix(yt, pred, labels=[0, 1, 2], sample_weight=w) / w.sum() * 100, 2))
