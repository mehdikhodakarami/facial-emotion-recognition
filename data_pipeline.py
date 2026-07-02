"""
═══════════════════════════════════════════════════════════════════════════
  پایپ‌لاین داده مبتنی بر tf.data
═══════════════════════════════════════════════════════════════════════════
  چرا tf.data به‌جای بارگذاری همهٔ تصاویر در آرایهٔ NumPy؟
    • دیتاست جدید ~۶۶هزار تصویر دارد؛ نگه‌داشتن همه در RAM روی سیستم ۸ گیگ
      باعث کرش می‌شود.
    • tf.data تصاویر را به‌صورت جریانی از دیسک می‌خواند، فقط یک batch در لحظه
      در حافظه است، و با prefetch هم‌زمان با آموزش داده آماده می‌کند.
    • نرمال‌سازی و Augmentation داخل خود مدل انجام می‌شود تا train و inference
      دقیقاً یکسان باشند (هیچ نشتی/ناهماهنگی).

  خروجی: سه شیء tf.data.Dataset برای train / val / test.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import tensorflow as tf

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASS_NAMES, NUM_CLASSES, IMG_SIZE, CHANNELS,
    BATCH_SIZE, SHUFFLE_BUFFER,
)

AUTOTUNE = tf.data.AUTOTUNE


def _make_dataset(directory, shuffle, batch_size=BATCH_SIZE):
    """ساخت یک tf.data.Dataset از پوشه‌ای با ساختار directory/<class>/*.png"""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',          # one-hot → برای label smoothing
        class_names=CLASS_NAMES,           # قفل‌کردن ترتیب کلاس‌ها
        color_mode='grayscale' if CHANNELS == 1 else 'rgb',
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='area',              # برای کوچک‌کردن کیفیت بهتر
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )
    return ds


def build_datasets(batch_size=BATCH_SIZE, cache_eval=True, weighted_train=False):
    """
    سه دیتاست آماده برای fit/evaluate برمی‌گرداند.

    train : shuffle + prefetch (بدون cache؛ برای حفظ RAM)
    val   : cache + prefetch (کوچک است، در RAM جا می‌شود)
    test  : cache + prefetch

    weighted_train=True : به هر نمونهٔ train بر اساس وزن کلاسش sample_weight اضافه
    می‌کند (برای جبران عدم‌توازن خفیف؛ مثلاً neutral پرشمارتر است). خروجی train
    آنگاه (x, y, sample_weight) خواهد بود — قالبی که Keras مستقیماً پشتیبانی می‌کند.
    """
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"پوشهٔ train پیدا نشد: {TRAIN_DIR}\n"
            f"  مسیر دیتاست را در config.py (DATA_ROOT) یا متغیر محیطی "
            f"FER_DATA_ROOT تنظیم کنید."
        )

    train_ds = _make_dataset(TRAIN_DIR, shuffle=True,  batch_size=batch_size)
    val_ds   = _make_dataset(VAL_DIR,   shuffle=False, batch_size=batch_size)
    test_ds  = _make_dataset(TEST_DIR,  shuffle=False, batch_size=batch_size)

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER, seed=42,
                                reshuffle_each_iteration=True)

    if weighted_train:
        cw = compute_class_weights(TRAIN_DIR)
        w_vec = tf.constant([cw[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

        def _attach_weight(x, y):
            idx = tf.argmax(y, axis=-1)
            return x, y, tf.gather(w_vec, idx)

        train_ds = train_ds.map(_attach_weight, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(AUTOTUNE)

    if cache_eval:
        val_ds  = val_ds.cache().prefetch(AUTOTUNE)
        test_ds = test_ds.cache().prefetch(AUTOTUNE)
    else:
        val_ds  = val_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


def count_per_class(directory=TRAIN_DIR):
    """تعداد فایل تصویری در هر کلاس را برمی‌گرداند (برای آمار/گزارش)."""
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    counts = {}
    for c in CLASS_NAMES:
        cdir = os.path.join(directory, c)
        if os.path.isdir(cdir):
            counts[c] = sum(
                1 for f in os.listdir(cdir)
                if f.lower().endswith(valid_exts)
            )
        else:
            counts[c] = 0
    return counts


def compute_class_weights(directory=TRAIN_DIR):
    """
    وزن کلاس‌ها به روش 'balanced' بر اساس تعداد فایل‌ها.
    دیتاست تقریباً متوازن است، اما این تابع برای گزارش/استفادهٔ اختیاری مفید است.
    """
    counts = count_per_class(directory)
    total  = sum(counts.values())
    weights = {}
    for i, c in enumerate(CLASS_NAMES):
        n = counts[c]
        weights[i] = (total / (NUM_CLASSES * n)) if n > 0 else 1.0
    return weights


def steps_per_epoch(directory=TRAIN_DIR, batch_size=BATCH_SIZE):
    """تعداد گام در هر epoch (برای زمان‌بندی نرخ یادگیری Cosine)."""
    total = sum(count_per_class(directory).values())
    return int(np.ceil(total / batch_size))


if __name__ == '__main__':
    # تست سریع پایپ‌لاین
    print("شمارش train :", count_per_class(TRAIN_DIR))
    print("steps/epoch :", steps_per_epoch())
    tr, va, te = build_datasets()
    for x, y in tr.take(1):
        print("batch x:", x.shape, x.dtype, "min/max:",
              float(tf.reduce_min(x)), float(tf.reduce_max(x)))
        print("batch y:", y.shape, y.dtype)
