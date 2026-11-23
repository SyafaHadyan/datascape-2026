"""Image helpers shared by cv_worker.py and build_cache.py (no torch needed)."""
import io
import random

import numpy as np
from PIL import Image, ImageFilter


class LetterboxResize:
    """Resize preserving aspect ratio, then pad to a square with the ImageNet mean color."""

    def __init__(self, size, fill=(124, 116, 104)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        offset = ((self.size - new_w) // 2, (self.size - new_h) // 2)
        canvas.paste(img, offset)
        return canvas


class RandomDegrade:
    """Randomly lower image quality to mimic the compression and resolution differences
    between the training and test photos (test images have a lower median bytes per pixel).
    Each step fires independently with its own probability, so many images stay clean.
    Randomness comes from python's random and numpy, which seed_worker reseeds per worker."""

    def __init__(self, p_down=0.3, down_scale=(0.4, 0.8), p_blur=0.2, blur_radius=(0.3, 1.2),
                 p_noise=0.2, noise_std=(2.0, 8.0), p_jpeg=0.5, jpeg_quality=(25, 95)):
        self.p_down, self.down_scale = p_down, down_scale
        self.p_blur, self.blur_radius = p_blur, blur_radius
        self.p_noise, self.noise_std = p_noise, noise_std
        self.p_jpeg, self.jpeg_quality = p_jpeg, jpeg_quality

    def __call__(self, img):
        w, h = img.size
        if random.random() < self.p_down:
            s = random.uniform(*self.down_scale)
            small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        if random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(*self.blur_radius)))
        if random.random() < self.p_noise:
            arr = np.asarray(img, dtype=np.float32)
            arr = arr + np.random.normal(0.0, random.uniform(*self.noise_std), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if random.random() < self.p_jpeg:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(int(self.jpeg_quality[0]), int(self.jpeg_quality[1])))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img
