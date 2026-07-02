"""
═══════════════════════════════════════════════════════════════════════════
  ارزیابی کامل و حرفه‌ای مدل FER (8 کلاس)
═══════════════════════════════════════════════════════════════════════════
  ✓ Top-1 و Top-2 Accuracy + Macro/Weighted F1
  ✓ Classification Report (Precision/Recall/F1/Support)
  ✓ Confusion Matrix خام + نرمال‌شده
  ✓ نمودار دقت هر کلاس
  ✓ نمایش بدترین پیش‌بینی‌ها (تحلیل خطا)
  ✓ گزارش متنی کامل

  نکته: test set دیتاست جدید کوچک و نامتوازن است (مثلاً disgust=21, contempt=30)؛
  بنابراین Macro-F1 و per-class recall مهم‌تر از accuracy کلی هستند.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, classification_report,
    top_k_accuracy_score, f1_score,
)

from config import (
    MODELS_DIR, RESULTS_DIR,
    DISPLAY_NAMES, NUM_CLASSES,
)
from data_pipeline import build_datasets


def _gather_xy(dataset):
    """یک tf.data.Dataset (one-hot) را به آرایه‌های NumPy تبدیل می‌کند."""
    xs, ys = [], []
    for xb, yb in dataset:
        xs.append(xb.numpy())
        ys.append(np.argmax(yb.numpy(), axis=1))
    return np.concatenate(xs), np.concatenate(ys)


# ══════════════════════════════════════════════════════════════════════════
#  نمودارها
# ══════════════════════════════════════════════════════════════════════════

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FER SE-ResNet — Training History', fontsize=14, fontweight='bold')

    best_val = max(history['val_accuracy'])
    best_ep  = history['val_accuracy'].index(best_val)

    axes[0].plot(history['accuracy'],     label='Train Acc', lw=2, color='#2980b9')
    axes[0].plot(history['val_accuracy'], label='Val Acc',   lw=2, color='#e74c3c', ls='--')
    axes[0].axvline(best_ep, color='green', ls=':', lw=1.5,
                    label=f'Best (Ep {best_ep+1}, {best_val*100:.1f}%)')
    axes[0].set_title(f'Accuracy — Best Val: {best_val*100:.2f}%')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1)

    axes[1].plot(history['loss'],     label='Train Loss', lw=2, color='#2980b9')
    axes[1].plot(history['val_loss'], label='Val Loss',   lw=2, color='#e74c3c', ls='--')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {path}")


def plot_confusion_matrices(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                        where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle('Confusion Matrix — FER SE-ResNet', fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Sample Counts', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    axes[0].tick_params(axis='x', rotation=35); axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Normalized (Recall per class)', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    axes[1].tick_params(axis='x', rotation=35); axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {path}")


def plot_per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1)
    per_class_acc = np.divide(cm.diagonal(), row_sums,
                              out=np.zeros(NUM_CLASSES), where=row_sums != 0)
    mean_acc = float(np.mean(per_class_acc))

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(DISPLAY_NAMES, per_class_acc * 100,
                  color=colors, edgecolor='black', lw=0.7, zorder=3)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{acc*100:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    ax.axhline(mean_acc * 100, color='red', ls='--', lw=2,
               label=f'Mean: {mean_acc*100:.1f}%', zorder=4)
    ax.set_ylim(0, 115)
    ax.set_title('Per-Class Accuracy (Recall)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion'); ax.set_ylabel('Accuracy (%)')
    ax.legend(); ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'per_class_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {path}")


def plot_worst_predictions(X, y_true, y_pred, probs, n=20):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("  ! هیچ پیش‌بینی اشتباهی پیدا نشد.")
        return
    confs = probs[wrong, y_pred[wrong]]
    top = wrong[np.argsort(confs)[::-1][:n]]

    cols = 5
    rows = (len(top) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    fig.suptitle('Worst Predictions (Highest-Confidence Errors)',
                 fontsize=14, fontweight='bold')
    axes_flat = np.array(axes).flatten()

    for i, idx in enumerate(top):
        img = X[idx]
        if img.shape[-1] == 1:
            img = img.squeeze(); cmap = 'gray'
        else:
            cmap = None
        axes_flat[i].imshow(img.astype('uint8'), cmap=cmap, vmin=0, vmax=255)
        axes_flat[i].set_title(
            f"True: {DISPLAY_NAMES[y_true[idx]]}\n"
            f"Pred: {DISPLAY_NAMES[y_pred[idx]]}\n"
            f"Conf: {probs[idx, y_pred[idx]]*100:.0f}%",
            fontsize=9, color='red', fontweight='bold')
        axes_flat[i].axis('off')
    for j in range(len(top), len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(RESULTS_DIR, 'worst_predictions.png')
    plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"  ✓ {path}")


def save_report(report_str, top1, top2, macro_f1, weighted_f1, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  FER SE-ResNet — Evaluation Report\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Top-1 Accuracy : {top1*100:.2f}%\n")
        f.write(f"  Top-2 Accuracy : {top2*100:.2f}%\n")
        f.write(f"  Macro    F1    : {macro_f1*100:.2f}%\n")
        f.write(f"  Weighted F1    : {weighted_f1*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, 'fer_model_best.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, 'fer_model_final.keras')
    print(f"\n  بارگذاری مدل: {model_path}")
    model = load_model(model_path)

    print("  ساخت test set ...")
    _, _, test_ds = build_datasets()

    print("  جمع‌آوری تصاویر و برچسب‌ها ...")
    X_test, y_test = _gather_xy(test_ds)
    print(f"  X_test: {X_test.shape}")

    print("  اجرای پیش‌بینی ...")
    y_pred_probs = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    labels = list(range(NUM_CLASSES))
    top1 = float(np.mean(y_pred == y_test))
    top2 = float(top_k_accuracy_score(y_test, y_pred_probs, k=2, labels=labels))
    macro_f1    = float(f1_score(y_test, y_pred, average='macro', labels=labels, zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted', labels=labels, zero_division=0))
    report_str = classification_report(
        y_test, y_pred, labels=labels,
        target_names=DISPLAY_NAMES, zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"  Top-1 Accuracy : {top1*100:.2f}%")
    print(f"  Top-2 Accuracy : {top2*100:.2f}%")
    print(f"  Macro    F1    : {macro_f1*100:.2f}%")
    print(f"  Weighted F1    : {weighted_f1*100:.2f}%")
    print("=" * 60)
    print("\n  Classification Report:")
    print(report_str)

    print("\n  ساخت نمودارها ...")
    history_path = os.path.join(MODELS_DIR, 'history.npy')
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_history(history)
    plot_confusion_matrices(y_test, y_pred)
    plot_per_class_accuracy(y_test, y_pred)
    plot_worst_predictions(X_test, y_test, y_pred, y_pred_probs, n=20)

    save_report(report_str, top1, top2, macro_f1, weighted_f1,
                os.path.join(RESULTS_DIR, 'evaluation_report.txt'))

    print("\n" + "=" * 60)
    print(f"  همهٔ نتایج در: {RESULTS_DIR}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
