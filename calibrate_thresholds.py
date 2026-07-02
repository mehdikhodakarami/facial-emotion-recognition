"""
═══════════════════════════════════════════════════════════════════════════
  کالیبراسیون آستانهٔ اطمینان به‌ازای هر کلاس (Per-Class Confidence Threshold)
═══════════════════════════════════════════════════════════════════════════
  چرا؟  یک آستانهٔ سراسری (مثلاً 0.50) برای همهٔ کلاس‌ها بد است:
    • با 8 کلاس + Label Smoothing، حداکثر softmax سقف ~0.91 دارد.
    • کلاس‌های آسان (happy/surprise) اطمینان بالا دارند → آستانهٔ بالا اشکالی ندارد.
    • کلاس‌های سخت (contempt/disgust/fear) اطمینان پایین‌تری دارند →
      آستانهٔ بالا باعث می‌شود حتی پیش‌بینی‌های درست هم «Uncertain» شوند و recall نابود شود.

  این اسکریپت روی validation set برای هر کلاس آستانه‌ای پیدا می‌کند که
  F1 تصمیمِ «قبول/Uncertain» را بیشینه کند، و نتیجه را در models/thresholds.json
  ذخیره می‌کند تا webcam_inference.py از آن استفاده کند.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from config import (
    MODELS_DIR, RESULTS_DIR, THRESHOLDS_PATH,
    CLASS_NAMES, DISPLAY_NAMES, NUM_CLASSES, CONFIDENCE_THRESHOLD,
)
from data_pipeline import build_datasets

# دامنهٔ مجاز آستانه (جلوگیری از مقادیر افراطی)
T_MIN, T_MAX = 0.15, 0.85
T_GRID = np.round(np.arange(T_MIN, T_MAX + 1e-9, 0.01), 3)


def gather_predictions(model, dataset):
    probs, y_true = [], []
    for xb, yb in dataset:
        p = model.predict(xb, verbose=0)
        probs.append(p)
        y_true.append(np.argmax(yb.numpy(), axis=1))
    return np.concatenate(probs), np.concatenate(y_true)


def best_threshold_for_class(c, y_pred, conf, y_true):
    """آستانه‌ای که F1 تصمیم قبول برای کلاس c را بیشینه می‌کند."""
    pred_c = (y_pred == c)
    n_true_c = int(np.sum(y_true == c))
    if pred_c.sum() == 0 or n_true_c == 0:
        return CONFIDENCE_THRESHOLD, 0.0, 0.0, 0.0

    best_t, best_f1, best_p, best_r = CONFIDENCE_THRESHOLD, -1.0, 0.0, 0.0
    for t in T_GRID:
        accept = pred_c & (conf >= t)
        tp = int(np.sum(accept & (y_true == c)))
        fp = int(np.sum(accept & (y_true != c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / n_true_c
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, float(t), precision, recall
    return best_t, best_f1, best_p, best_r


def coverage_stats(thresholds, y_pred, conf, y_true):
    """آمار کلی: درصد پذیرفته‌شده و دقت روی پذیرفته‌شده‌ها با آستانه‌های داده‌شده."""
    th_vec = np.array([thresholds[c] for c in range(NUM_CLASSES)])
    accept = conf >= th_vec[y_pred]
    n = len(y_true)
    n_acc = int(np.sum(accept))
    acc_on_accepted = (float(np.mean(y_pred[accept] == y_true[accept]))
                       if n_acc > 0 else 0.0)
    return n_acc / n, acc_on_accepted


def plot_thresholds(per_class, metrics):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Per-Class Confidence Thresholds (calibrated on validation)',
                 fontsize=14, fontweight='bold')

    # آستانه‌ها
    ths = [per_class[c] for c in CLASS_NAMES]
    bars = axes[0].bar(DISPLAY_NAMES, ths, color=colors, edgecolor='black', lw=0.7)
    axes[0].axhline(CONFIDENCE_THRESHOLD, color='red', ls='--', lw=1.5,
                    label=f'Global fallback ({CONFIDENCE_THRESHOLD})')
    for b, t in zip(bars, ths):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height()+0.01,
                     f'{t:.2f}', ha='center', va='bottom', fontweight='bold')
    axes[0].set_ylim(0, 1); axes[0].set_ylabel('Threshold')
    axes[0].set_title('Chosen threshold per class'); axes[0].legend()
    axes[0].tick_params(axis='x', rotation=20); axes[0].grid(axis='y', alpha=0.3)

    # precision/recall در آستانهٔ انتخابی
    prec = [metrics[c]['precision'] for c in CLASS_NAMES]
    rec  = [metrics[c]['recall']    for c in CLASS_NAMES]
    x = np.arange(NUM_CLASSES); w = 0.38
    axes[1].bar(x - w/2, np.array(prec)*100, w, label='Precision', color='#2980b9')
    axes[1].bar(x + w/2, np.array(rec)*100,  w, label='Recall',    color='#e67e22')
    axes[1].set_xticks(x); axes[1].set_xticklabels(DISPLAY_NAMES, rotation=20)
    axes[1].set_ylim(0, 105); axes[1].set_ylabel('%')
    axes[1].set_title('Accept-decision Precision/Recall at chosen threshold')
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'thresholds_calibration.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ نمودار کالیبراسیون: {path}")


def main():
    model_path = os.path.join(MODELS_DIR, 'fer_model_best.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, 'fer_model_final.keras')
    print(f"\n  بارگذاری مدل: {model_path}")
    model = load_model(model_path)

    print("  پیش‌بینی روی validation set ...")
    _, val_ds, _ = build_datasets()
    probs, y_true = gather_predictions(model, val_ds)
    y_pred = np.argmax(probs, axis=1)
    conf   = np.max(probs, axis=1)
    print(f"  نمونه‌های validation: {len(y_true)}")

    per_class, metrics = {}, {}
    print("\n  کالیبراسیون آستانهٔ هر کلاس (بیشینه‌سازی F1):")
    print(f"  {'class':10s}  {'thr':>5s}  {'F1':>6s}  {'prec':>6s}  {'recall':>6s}")
    for c in range(NUM_CLASSES):
        t, f1, p, r = best_threshold_for_class(c, y_pred, conf, y_true)
        per_class[CLASS_NAMES[c]] = round(t, 3)
        metrics[CLASS_NAMES[c]] = {'f1': f1, 'precision': p, 'recall': r}
        print(f"  {DISPLAY_NAMES[c]:10s}  {t:5.2f}  {f1*100:5.1f}%  "
              f"{p*100:5.1f}%  {r*100:5.1f}%")

    # مقایسهٔ پوشش/دقت: آستانهٔ کالیبره vs سراسری 0.40 vs 0.50
    th_cal = {c: per_class[CLASS_NAMES[c]] for c in range(NUM_CLASSES)}
    cov_cal,  acc_cal  = coverage_stats(th_cal, y_pred, conf, y_true)
    cov_040, acc_040 = coverage_stats({c: 0.40 for c in range(NUM_CLASSES)},
                                      y_pred, conf, y_true)
    cov_050, acc_050 = coverage_stats({c: 0.50 for c in range(NUM_CLASSES)},
                                      y_pred, conf, y_true)

    print("\n  مقایسهٔ پوشش (٪ پذیرفته‌شده) و دقتِ پذیرفته‌شده‌ها:")
    print(f"    Per-class calibrated : coverage={cov_cal*100:5.1f}%  acc={acc_cal*100:5.1f}%")
    print(f"    Global 0.40          : coverage={cov_040*100:5.1f}%  acc={acc_040*100:5.1f}%")
    print(f"    Global 0.50          : coverage={cov_050*100:5.1f}%  acc={acc_050*100:5.1f}%")

    # ذخیره
    os.makedirs(MODELS_DIR, exist_ok=True)
    out = {
        'per_class': per_class,
        'global_fallback': CONFIDENCE_THRESHOLD,
        'calibrated_on': 'validation',
        'coverage_calibrated': round(cov_cal, 4),
        'accuracy_on_accepted_calibrated': round(acc_cal, 4),
    }
    with open(THRESHOLDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ آستانه‌ها ذخیره شد: {THRESHOLDS_PATH}")

    plot_thresholds(per_class, metrics)
    print("\n  کالیبراسیون کامل شد. webcam_inference حالا خودکار از این آستانه‌ها استفاده می‌کند.\n")


if __name__ == '__main__':
    main()
