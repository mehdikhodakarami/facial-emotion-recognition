"""
═══════════════════════════════════════════════════════════════════════════
  آموزش مدل تشخیص احساسات چهره — SE-ResNet (نسخهٔ تقویت‌شده)
═══════════════════════════════════════════════════════════════════════════
  ویژگی‌های حرفه‌ای این نسخه:
    ✓ معماری SE-ResNet (Residual + Squeeze-and-Excitation attention)
    ✓ Augmentation به‌صورت لایه‌های Keras داخل مدل (GPU، و یکسان با inference)
    ✓ نرمال‌سازی داخل مدل (Rescaling) → inference بدون دردسر
    ✓ AdamW (weight decay) + زمان‌بندی Cosine با Warmup
    ✓ Label Smoothing برای کاهش بیش‌اطمینانی
    ✓ Top-2 Accuracy، EarlyStopping، Checkpoint، TensorBoard، CSVLogger
    ✓ پایپ‌لاین tf.data (مناسب RAM محدود)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    NUM_CLASSES, INPUT_SHAPE, IMG_SIZE, CHANNELS,
    BATCH_SIZE, EPOCHS, BASE_LR, WARMUP_EPOCHS,
    WEIGHT_DECAY, LABEL_SMOOTHING, DROPOUT_HEAD,
    EARLY_STOP_PATIENCE, DISPLAY_NAMES,
    STEM_FILTERS, STAGE_FILTERS, BLOCKS_PER_STAGE, HEAD_UNITS,
    STEPS_PER_EPOCH_CAP, SEED,
)
from data_pipeline import build_datasets, steps_per_epoch, count_per_class


# ══════════════════════════════════════════════════════════════════════════
#  بلوک‌های معماری
# ══════════════════════════════════════════════════════════════════════════

def se_block(x, ratio=16, name=None):
    """Squeeze-and-Excitation: به مدل اجازه می‌دهد روی کانال‌های مهم تمرکز کند."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D(keepdims=True, name=f'{name}_squeeze')(x)
    se = layers.Dense(max(filters // ratio, 4), activation='relu',
                      use_bias=False, name=f'{name}_fc1')(se)
    se = layers.Dense(filters, activation='sigmoid',
                      use_bias=False, name=f'{name}_fc2')(se)
    return layers.Multiply(name=f'{name}_scale')([x, se])


def residual_se_block(x, filters, stride=1, drop=0.0, name='res'):
    """بلوک باقیمانده + SE attention با projection shortcut در صورت نیاز."""
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)

    x = layers.Conv2D(filters, 3, padding='same',
                      use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    x = se_block(x, name=f'{name}_se')

    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, name=f'{name}_proj')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name}_out')(x)
    if drop > 0:
        x = layers.SpatialDropout2D(drop, name=f'{name}_drop')(x)
    return x


def build_augmentation():
    """Augmentation به‌صورت لایه — فقط هنگام training فعال است."""
    return tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15, value_range=(0.0, 255.0)),
    ], name='augmentation')


def build_se_resnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    SE-ResNet با عمق قابل‌تنظیم از روی config (STAGE_FILTERS).
    اولین مرحله stride=1 و مراحل بعدی stride=2 دارند (کاهش تدریجی ابعاد).
    """
    inputs = layers.Input(shape=input_shape, name='input_image')

    # Augmentation + نرمال‌سازی داخل مدل (ورودی خام 0..255)
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0, name='rescale')(x)

    # Stem
    x = layers.Conv2D(STEM_FILTERS, 3, padding='same', use_bias=False, name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_relu')(x)
    x = layers.MaxPooling2D(2, name='stem_pool')(x)

    # مراحل residual + SE (تعداد و عرض از config)
    for si, filters in enumerate(STAGE_FILTERS, start=1):
        stride = 1 if si == 1 else 2
        drop = 0.10 if filters < 256 else 0.15
        for bi in range(1, BLOCKS_PER_STAGE + 1):
            name = f's{si}b{bi}'
            x = residual_se_block(
                x, filters,
                stride=stride if bi == 1 else 1,
                drop=(drop if bi == BLOCKS_PER_STAGE else 0.0),
                name=name,
            )

    # Head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(DROPOUT_HEAD, name='head_drop1')(x)
    x = layers.Dense(HEAD_UNITS, use_bias=False, name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.Activation('relu', name='fc1_relu')(x)
    x = layers.Dropout(DROPOUT_HEAD, name='head_drop2')(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return Model(inputs, outputs, name='FER_SE_ResNet')


# ══════════════════════════════════════════════════════════════════════════
#  کمکی‌ها
# ══════════════════════════════════════════════════════════════════════════

def save_model_summary(model, path):
    with open(path, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda line: f.write(line + '\n'))
    print(f"  معماری ذخیره شد: {path}")


def plot_history(history, save_dir):
    h = history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FER SE-ResNet — Training History', fontsize=14, fontweight='bold')

    best_val = max(h['val_accuracy'])
    best_ep  = h['val_accuracy'].index(best_val)

    axes[0].plot(h['accuracy'],     label='Train', lw=2, color='#2980b9')
    axes[0].plot(h['val_accuracy'], label='Val',   lw=2, color='#e74c3c', ls='--')
    axes[0].axvline(best_ep, color='green', ls=':', lw=1.5,
                    label=f'Best (Ep {best_ep+1}, {best_val*100:.2f}%)')
    axes[0].set_title(f'Accuracy — Best Val: {best_val*100:.2f}%')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1)

    axes[1].plot(h['loss'],     label='Train', lw=2, color='#2980b9')
    axes[1].plot(h['val_loss'], label='Val',   lw=2, color='#e74c3c', ls='--')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  نمودار آموزش ذخیره شد: {path}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    # بازتولیدپذیری کامل: seed یکسان برای python / numpy / tensorflow
    tf.keras.utils.set_random_seed(SEED)

    for d in (MODELS_DIR, LOGS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ساخت پایپ‌لاین داده (tf.data) ...")
    print("=" * 60)
    # weighted_train=True → جبران عدم‌توازن خفیف (neutral پرشمارتر است)
    train_ds, val_ds, _ = build_datasets(batch_size=BATCH_SIZE, weighted_train=True)

    train_counts = count_per_class()
    print("  توزیع کلاس‌های Train (با sample-weight متعادل می‌شود):")
    for c, n in train_counts.items():
        print(f"    {c:10s}: {n}")

    full_spe = steps_per_epoch(batch_size=BATCH_SIZE)
    if STEPS_PER_EPOCH_CAP and STEPS_PER_EPOCH_CAP < full_spe:
        spe = STEPS_PER_EPOCH_CAP
        # برای دیدن زیرمجموعهٔ متفاوت در هر epoch، دیتاست را تکرار می‌کنیم
        train_ds = train_ds.repeat()
        print(f"\n  حالت سبک: هر epoch {spe} گام از {full_spe} گام کامل "
              f"(~{spe*BATCH_SIZE:,} تصویر، با reshuffle کل داده پوشش داده می‌شود)")
    else:
        spe = full_spe
        print(f"\n  هر epoch کل داده: {spe} گام")

    total_steps  = spe * EPOCHS
    warmup_steps = spe * WARMUP_EPOCHS
    print(f"  total_steps={total_steps}  |  warmup={warmup_steps}")

    # ─── زمان‌بندی نرخ یادگیری: Cosine با Warmup ───────────────────────────
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-2,                      # کف LR = 1٪ مقدار اولیه
        warmup_target=BASE_LR,
        warmup_steps=warmup_steps,
    )

    print("\n  ساخت مدل SE-ResNet ...")
    model = build_se_resnet()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2'),
        ],
    )
    model.summary()
    save_model_summary(model, os.path.join(MODELS_DIR, 'model_summary.txt'))

    # ─── Callbacks ────────────────────────────────────────────────────────
    best_path = os.path.join(MODELS_DIR, 'fer_model_best.keras')
    log_dir   = os.path.join(LOGS_DIR, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    callbacks = [
        ModelCheckpoint(best_path, monitor='val_accuracy', mode='max',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', mode='max',
                      patience=EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False),
        CSVLogger(os.path.join(RESULTS_DIR, 'training_log.csv')),
    ]

    print("\n" + "=" * 60)
    print(f"  شروع آموزش  (تا {EPOCHS} epoch، EarlyStopping={EARLY_STOP_PATIENCE})")
    print(f"  TensorBoard: tensorboard --logdir {LOGS_DIR}")
    print("=" * 60 + "\n")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=spe,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # ─── ذخیره خروجی‌ها ───────────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, 'fer_model_final.keras')
    model.save(final_path)
    print(f"\n  مدل نهایی ذخیره شد: {final_path}")

    np.save(os.path.join(MODELS_DIR, 'history.npy'), history.history)
    with open(os.path.join(MODELS_DIR, 'history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    plot_history(history.history, RESULTS_DIR)

    best_val = max(history.history['val_accuracy'])
    best_ep  = history.history['val_accuracy'].index(best_val) + 1

    # ذخیرهٔ کانفیگ کامل آموزش (هایپرپارامترها + نتایج) — برای گزارش و بازتولید
    run_config = {
        'seed': SEED,
        'img_size': IMG_SIZE,
        'channels': CHANNELS,
        'architecture': 'SE-ResNet',
        'stem_filters': STEM_FILTERS,
        'stage_filters': list(STAGE_FILTERS),
        'blocks_per_stage': BLOCKS_PER_STAGE,
        'head_units': HEAD_UNITS,
        'total_params': int(model.count_params()),
        'batch_size': BATCH_SIZE,
        'epochs_planned': EPOCHS,
        'epochs_run': len(history.history['accuracy']),
        'steps_per_epoch': spe,
        'base_lr': BASE_LR,
        'warmup_epochs': WARMUP_EPOCHS,
        'weight_decay': WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'optimizer': 'AdamW',
        'lr_schedule': 'CosineDecay+Warmup',
        'best_val_accuracy': round(float(best_val), 4),
        'best_epoch': best_ep,
        'best_val_top2': round(float(max(history.history.get('val_top2', [0]))), 4),
    }
    with open(os.path.join(MODELS_DIR, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    print(f"  کانفیگ آموزش ذخیره شد: {os.path.join(MODELS_DIR, 'training_config.json')}")

    print("\n" + "=" * 60)
    print(f"  آموزش کامل شد.")
    print(f"  بهترین Val Accuracy : {best_val*100:.2f}%  (Epoch {best_ep})")
    print(f"  تعداد Epoch اجراشده : {len(history.history['accuracy'])}")
    print(f"  بهترین مدل          : {best_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
