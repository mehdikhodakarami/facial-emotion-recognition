"""
ابزارهای کمکی: نمایش توزیع کلاس‌ها، بارگذاری نمونه‌ها و صحت‌سنجی دیتاست.
(بارگذاری انبوه در NumPy حذف شد؛ آموزش حالا از پایپ‌لاین جریانی tf.data استفاده می‌کند.)
"""

import os
import numpy as np
from PIL import Image

from config import CLASS_NAMES, DISPLAY_NAMES, IMG_SIZE

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def list_image_files(class_dir):
    if not os.path.isdir(class_dir):
        return []
    return [f for f in os.listdir(class_dir)
            if f.lower().endswith(VALID_EXTS)]


def print_class_distribution(counts, title="Class Distribution"):
    """نمایش جدولی توزیع کلاس‌ها از روی dict {class: count}."""
    print(f"\n{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")
    total = sum(counts.values())
    for i, c in enumerate(CLASS_NAMES):
        n = counts.get(c, 0)
        pct = n / total * 100 if total else 0
        bar = '▓' * int(pct / 2)
        print(f"  {DISPLAY_NAMES[i]:10s} [{i}]: {n:6d}  ({pct:5.1f}%)  {bar}")
    print(f"{'─'*52}")
    print(f"  Total: {total}\n")


def load_sample_images(folder, per_class=1):
    """چند نمونه از هر کلاس برای نمایش شبکه‌ای برمی‌گرداند (grayscale, IMG_SIZE)."""
    samples = []
    for c in CLASS_NAMES:
        cdir = os.path.join(folder, c)
        files = list_image_files(cdir)[:per_class]
        for f in files:
            try:
                img = Image.open(os.path.join(cdir, f)).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                samples.append((c, np.array(img, dtype='uint8')))
            except Exception as e:
                print(f"  [Error] {f}: {e}")
    return samples


def find_corrupt_images(folder):
    """تصاویر غیرقابل‌خواندن را پیدا می‌کند (صحت‌سنجی دیتاست)."""
    bad = []
    for c in CLASS_NAMES:
        cdir = os.path.join(folder, c)
        for f in list_image_files(cdir):
            fpath = os.path.join(cdir, f)
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception:
                bad.append(fpath)
    return bad
