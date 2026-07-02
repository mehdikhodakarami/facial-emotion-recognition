"""
═══════════════════════════════════════════════════════════════════════════
  تحلیل و صحت‌سنجی دیتاست  (Exploratory Data Analysis)
═══════════════════════════════════════════════════════════════════════════
  این اسکریپت دیگر فایل .npy نمی‌سازد (آموزش از پایپ‌لاین جریانی tf.data
  استفاده می‌کند). به‌جای آن، برای گزارش پایان‌نامه موارد زیر را تولید می‌کند:
    1. شمارش و توزیع کلاس‌ها در سه split (train/val/test)
    2. نمودار میله‌ای توزیع
    3. شبکهٔ نمونه از هر کلاس
    4. بررسی تصاویر خراب/غیرقابل‌خواندن
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, RESULTS_DIR,
    CLASS_NAMES, DISPLAY_NAMES,
)
from data_pipeline import count_per_class
from utils import print_class_distribution, load_sample_images, find_corrupt_images


def plot_distribution(splits):
    """نمودار میله‌ای توزیع کلاس‌ها برای هر split."""
    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]
    fig.suptitle('Class Distribution — New 8-Emotion Dataset',
                 fontsize=14, fontweight='bold')

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(CLASS_NAMES))]

    for ax, (name, counts) in zip(axes, splits.items()):
        values = [counts.get(c, 0) for c in CLASS_NAMES]
        bars = ax.bar(DISPLAY_NAMES, values, color=colors,
                      edgecolor='black', lw=0.7)
        ax.set_title(f'{name}  ({sum(values):,})', fontweight='bold')
        ax.set_xlabel('Emotion'); ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=0.3)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    str(v), ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'class_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ نمودار توزیع: {path}")


def plot_sample_grid(folder):
    """یک نمونه از هر کلاس را در یک شبکه نمایش می‌دهد."""
    samples = load_sample_images(folder, per_class=1)
    if not samples:
        return
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    fig.suptitle('Sample per Class', fontsize=14, fontweight='bold')
    axes_flat = np.array(axes).flatten()
    for ax, (cls, img) in zip(axes_flat, samples):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(DISPLAY_NAMES[CLASS_NAMES.index(cls)], fontsize=10)
        ax.axis('off')
    for j in range(len(samples), len(axes_flat)):
        axes_flat[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, 'sample_grid.png')
    plt.savefig(path, dpi=140, bbox_inches='tight'); plt.close()
    print(f"  ✓ شبکهٔ نمونه: {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    splits = {}
    for name, d in [('Train', TRAIN_DIR), ('Validation', VAL_DIR), ('Test', TEST_DIR)]:
        if os.path.isdir(d):
            counts = count_per_class(d)
            splits[name] = counts
            print_class_distribution(counts, f"{name} distribution")
        else:
            print(f"  [WARNING] پوشه پیدا نشد: {d}")

    if splits:
        plot_distribution(splits)
        plot_sample_grid(TRAIN_DIR)

    print("\n  بررسی تصاویر خراب در Train ...")
    bad = find_corrupt_images(TRAIN_DIR)
    if bad:
        print(f"  [WARNING] {len(bad)} تصویر خراب پیدا شد:")
        for b in bad[:10]:
            print(f"    {b}")
    else:
        print("  ✓ هیچ تصویر خرابی در Train پیدا نشد.")

    print("\n  تحلیل دیتاست کامل شد.\n")


if __name__ == '__main__':
    main()
