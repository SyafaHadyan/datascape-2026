"""Quick speed test for training on Apple silicon (MPS), also runs on CUDA or CPU.

It uses random weights and random images, so nothing is downloaded and no dataset is needed.
Usage, python bench_mps.py [--steps 8] [--batch 32] [--models convnext_tiny,convnext_small] [--dtypes fp32,fp16,bf16]
"""
import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import timm
import torch
import torch.nn.functional as F

# Model name and the extra timm arguments the real runs use.
MODELS = {
    "convnext_tiny": {},
    "convnext_small": {},
    "vit_small_patch14_dinov2.lvd142m": {"img_size": 224},
}
TRAIN_STEPS_PER_EPOCH = 305  # 9737 training images per fold at batch 32
EVAL_STEPS_PER_EPOCH = 77  # 2435 validation images per fold at batch 32
EPOCHS = 30
FOLDS = 5


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench(name, kwargs, device, batch, steps, dtype):
    model = timm.create_model(name, pretrained=False, num_classes=3, **kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, 3, (batch,), device=device)
    use_amp = dtype is not None

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    def eval_step():
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            model(x)

    model.train()
    for _ in range(3):
        train_step()
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        train_step()
    sync(device)
    t_train = (time.perf_counter() - start) / steps

    model.eval()
    for _ in range(2):
        eval_step()
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        eval_step()
    sync(device)
    t_eval = (time.perf_counter() - start) / steps

    # The step counts above are for batch 32, so scale them when a smaller batch is used.
    epoch_s = (TRAIN_STEPS_PER_EPOCH * t_train + EVAL_STEPS_PER_EPOCH * t_eval) * (32 / batch)
    return t_train, t_eval, epoch_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--dtypes", default="fp32,fp16,bf16")
    args = parser.parse_args()

    device = pick_device()
    print(f"torch {torch.__version__} | timm {timm.__version__} | device {device}")
    names = [n for n in MODELS if any(n.startswith(m) for m in args.models.split(","))]
    for name in names:
        for label, dtype in (("fp32", None), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
            if label not in args.dtypes.split(","):
                continue
            try:
                t_train, t_eval, epoch_s = bench(name, MODELS[name], device, args.batch, args.steps, dtype)
            except Exception as e:
                print(f"{name} | {label} | failed, {type(e).__name__}, {str(e)[:100]}")
                continue
            run_h = epoch_s * EPOCHS * FOLDS / 3600
            print(
                f"{name} | {label} | train {t_train:.2f} s/step, eval {t_eval:.2f} s/step | "
                f"epoch about {epoch_s:.0f} s | one 5-fold 30-epoch run about {run_h:.1f} h"
            )


if __name__ == "__main__":
    main()
