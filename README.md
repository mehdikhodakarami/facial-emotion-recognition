<h1 align="center">🎭 Facial Emotion Recognition with SE-ResNet</h1>

<p align="center">
  <b>A complete, end-to-end deep-learning system that recognizes 8 facial emotions in real time.</b><br>
  Custom SE-ResNet · streaming <code>tf.data</code> pipeline · per-class confidence calibration · live webcam inference.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Test%20Accuracy-77.9%25-brightgreen">
  <img src="https://img.shields.io/badge/Top--2%20Accuracy-92.4%25-success">
  <img src="https://img.shields.io/badge/Params-2.9M-informational">
  <img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

<p align="center">
  <b>🇬🇧 English</b> · <a href="docs/README_FA.md">🇮🇷 فارسی</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Course-Data%20Mining-8A2BE2">
  <img src="https://img.shields.io/badge/University-Kharazmi%20University%2C%20Tehran-004080">
  <img src="https://img.shields.io/badge/Supervisor-Prof.%20Keyvan%20Borna-B22222">
</p>

<p align="center">
  🎓 Final project for the <b>Data Mining</b> course at <b>Kharazmi University, Tehran</b><br>
  Developed under the valued supervision and guidance of <b>Professor Keyvan Borna</b>
</p>

---

## ✨ Overview

This project detects **8 emotions** — `Angry · Contempt · Disgust · Fear · Happy · Neutral · Sad · Surprise` —
from a face image, and runs live from a webcam. It was built from scratch as the final project for the
**Data Mining** course at **Kharazmi University, Tehran**, under the supervision of **Professor Keyvan Borna** —
with a strong focus on **doing things the right way**: a memory-safe data pipeline, a modern attention-based
architecture, disciplined train/validation/test separation, and an inference stack tuned for the real world.

Despite being **lightweight (2.9M parameters)** and trained entirely on a **laptop CPU (Apple M2, no GPU)**,
the model reaches **77.9% top-1** and **92.4% top-2** accuracy on the held-out test set — close to the level of
human agreement on this notoriously hard 8-class problem (random guess = 12.5%).

> 🇮🇷 پروژهٔ پایانی هوش مصنوعی: تشخیص ۸ احساس چهره با شبکهٔ عصبی عمیق SE-ResNet،
> خط‌لولهٔ دادهٔ جریانی، و کالیبراسیون آستانهٔ اطمینان به‌ازای هر کلاس.

---

## 🏆 Results (held-out test set)

| Metric | Score |
|--------|:-----:|
| **Top-1 Accuracy** | **77.92%** |
| **Top-2 Accuracy** | **92.44%** |
| Weighted F1 | 78.26% |
| Macro F1 | 65.29% |

<details>
<summary><b>Per-class report (click to expand)</b></summary>

| Emotion | Precision | Recall | F1 | Support |
|---------|:---------:|:------:|:--:|:-------:|
| Happy    | 0.92 | 0.88 | **0.90** | 929 |
| Neutral  | 0.84 | 0.77 | **0.80** | 1274 |
| Surprise | 0.78 | 0.81 | **0.79** | 450 |
| Angry    | 0.67 | 0.79 | **0.73** | 322 |
| Sad      | 0.58 | 0.65 | **0.61** | 449 |
| Fear     | 0.48 | 0.61 | **0.54** | 98 |
| Disgust  | 0.50 | 0.52 | **0.51** | 21 |
| Contempt | 0.34 | 0.33 | **0.34** | 30 |

*Note: the test set is heavily imbalanced (e.g. only 21 Disgust / 30 Contempt samples), which is why
Macro-F1 and per-class metrics are reported alongside overall accuracy.*
</details>

### 📊 Visual results

| Training history | Confusion matrix |
|:---:|:---:|
| ![training](results/training_history.png) | ![confusion](results/confusion_matrix.png) |

| Per-class accuracy | Calibrated thresholds |
|:---:|:---:|
| ![perclass](results/per_class_accuracy.png) | ![thresholds](results/thresholds_calibration.png) |

---

## 🧠 Architecture — SE-ResNet

The model combines two proven, award-winning ideas:

- **Residual blocks** (ResNet, CVPR 2016) — skip connections that let gradients flow through a deep network.
- **Squeeze-and-Excitation attention** (SENet, CVPR 2017) — a lightweight unit that learns *which feature
  channels matter* and re-weights them.

```
Input 64×64×1
  → Augmentation (train only) → Rescaling(1/255)        # preprocessing baked INTO the model
  → Stem: Conv 3×3 + BN + ReLU + MaxPool                 (64 → 32)
  → Stage 1: 2 × SE-Residual (64  ch)                    (32×32)
  → Stage 2: 2 × SE-Residual (128 ch, stride 2)          (32 → 16)
  → Stage 3: 2 × SE-Residual (256 ch, stride 2)          (16 → 8)
  → GlobalAveragePooling → Dropout → Dense(256) → Dropout
  → Dense(8) + Softmax
```

**Why preprocessing lives inside the model:** normalization and augmentation are Keras layers *inside* the
network, so training and inference are guaranteed to be identical — this removes an entire class of subtle
train/serve mismatch bugs.

---

## 🔬 Key techniques

| Technique | Why it's used |
|-----------|---------------|
| **Streaming `tf.data` pipeline** | Streams 66k images from disk — constant, low RAM (runs on 8 GB) |
| **AdamW + Cosine schedule + Warmup** | Stable convergence and strong generalization |
| **Label Smoothing** | Prevents over-confidence, improves calibration |
| **Sample weighting** | Compensates for the (mild) class imbalance |
| **In-model augmentation** | Flip / rotate / zoom / translate / contrast / brightness on-graph |
| **Per-class threshold calibration** ⭐ | See below — the headline contribution |
| **Reproducibility** | Global seed + auto-saved `training_config.json` |

### ⭐ Per-class confidence calibration

A single global "Uncertain" threshold is unfair: easy classes (Happy) are confident, hard classes
(Contempt, Fear) are not — so a fixed threshold wrongly rejects correct predictions of the hard classes.
Instead, [`calibrate_thresholds.py`](calibrate_thresholds.py) finds a **separate threshold per class** by
maximizing each class's F1 **on the validation set** (never the test set — that would leak). The webcam then
uses these calibrated thresholds automatically.

### 🎥 Real-time inference, tuned for the real world

The webcam stack ([`webcam_inference.py`](webcam_inference.py)) adds inference-time robustness — all tunable
in [`config.py`](config.py) **without retraining**:

- **Temporal smoothing** — averages probabilities over recent frames to kill flicker.
- **Neutral-bias correction** — counteracts the model's tendency to over-predict "Neutral".
- **Threshold scaling** — relaxes thresholds for a smoother live experience.
- **Face-crop matching** — tighter crop to match the training distribution.

---

## 📁 Project structure

```
├── config.py               # single source of truth: paths, classes, hyperparameters, inference knobs
├── data_pipeline.py        # tf.data builders for train/val/test + class weights
├── preprocessing.py        # dataset EDA: distribution plots, sample grid, corrupt-image check
├── train_model.py          # SE-ResNet definition + training loop
├── calibrate_thresholds.py # per-class confidence-threshold calibration
├── evaluate_model.py        # full evaluation: F1, confusion matrix, per-class, error analysis
├── webcam_inference.py     # real-time inference (webcam / image / folder)
├── utils.py                # helpers
├── requirements.txt
├── models/                 # saved model, training config, thresholds, summary
└── results/                # all generated plots & reports
```

---

## 🚀 Getting started

### 1. Install

```bash
pip install -r requirements.txt
# Apple Silicon GPU (optional, big speed-up): pip install tensorflow-metal
```

### 2. Dataset

The model is trained on an 8-class facial-emotion dataset (grayscale, 112×112) with `train/validation/test`
splits. Place it under `Faces/` in the project root:

```
Faces/
├── train/       (angry, contempt, disgust, fear, happy, neutral, sad, suprise)
├── validation/
└── test/
```

Then point `config.py` to it (default is `./Faces`) or set an env var:

```bash
export FER_DATA_ROOT="/path/to/dataset"
```

> ⚠️ The "surprise" folder is spelled **`suprise`** on disk; `config.py` matches it exactly so the class
> loads correctly, and displays it as "Surprise" in all plots.

### 3. Run the pipeline

```bash
python preprocessing.py          # (optional) dataset analysis & plots
python train_model.py            # train the model  (~5 h on M2 CPU)
python calibrate_thresholds.py   # calibrate per-class thresholds on validation
python evaluate_model.py         # final evaluation on the test set
```

### 4. Live inference

```bash
python webcam_inference.py --mode webcam
python webcam_inference.py --mode image  --path path/to/image.jpg
python webcam_inference.py --mode folder --path path/to/folder/
```

> 💡 **Pre-trained model:** the trained weights (`fer_model_best.keras`) are available on the
> [**Releases**](../../releases) page. Download it into `models/` to run inference without training,
> or just run `train_model.py` to reproduce it (~5 h on a laptop CPU).

---

## ⚙️ Configuration highlights

Everything is tunable in [`config.py`](config.py):

- **Model size:** `STAGE_FILTERS = (64, 128, 256)` → add `512` for a stronger model on GPU.
- **Speed vs. accuracy:** `IMG_SIZE`, `STEPS_PER_EPOCH_CAP`, `EPOCHS`.
- **Live behavior:** `CLASS_BIAS`, `THRESHOLD_SCALE`, `TEMPORAL_SMOOTHING` — fix webcam issues without retraining.

---

## 📝 Notes

- Trained and evaluated on **Apple M2 (CPU-only, 8 GB RAM)** — ~7 min/epoch, ~5 h total.
- Fully reproducible: fixed global seed, and every run saves its exact config to `models/training_config.json`.

## 🎓 Academic context & acknowledgements

This project was developed as the final project for the **Data Mining** course at
**Kharazmi University, Tehran (Iran)**.

I would like to express my sincere gratitude to **Professor Keyvan Borna** for his
exceptional supervision, insightful guidance, and continuous support throughout this
project. His mentorship was instrumental in shaping both the methodology and the quality
of this work — it is a genuine honor to have completed it under his guidance.

> **Supervisor:** Professor Keyvan Borna — Kharazmi University, Tehran
> **Course:** Data Mining

## 📜 License

Released under the [MIT License](LICENSE).
