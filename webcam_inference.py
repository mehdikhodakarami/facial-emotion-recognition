"""
Real-time Facial Emotion Recognition

ویژگی‌ها:
  ✓ تشخیص چهره با MediaPipe Face Detection
  ✓ Face Alignment با MediaPipe Face Mesh (landmarks)
  ✓ Confidence Threshold → نمایش "Uncertain" + آمار لاگ (امتیاز اضافه)
  ✓ Preprocessing یکسان با train (CLAHE)
  ✓ نوار احتمالات رنگی برای هر 7 احساس
  ✓ نمایش FPS زنده (پایدار)
  ✓ پشتیبانی TFLite و H5
  ✓ سه حالت: webcam | image | folder
  ✓ ذخیره خودکار خروجی‌ها
"""

import os
import cv2
import time
import argparse
import numpy as np

from collections import deque
from config import (
    CLASS_NAMES, DISPLAY_NAMES, IDX_TO_DISPLAY, NUM_CLASSES,
    EMOTION_COLORS, CONFIDENCE_THRESHOLD, THRESHOLDS_PATH,
    load_class_thresholds, MODELS_DIR, RESULTS_DIR, IMG_SIZE,
    TEMPORAL_SMOOTHING, get_class_bias_vector, INFERENCE_TEMPERATURE,
    THRESHOLD_SCALE, FACE_CROP_PADDING,
)


# ══════════════════════════════════════════════════════════════════════════
#  بارگذاری مدل
# ══════════════════════════════════════════════════════════════════════════

def load_inference_model(model_path):
    """
    .tflite → TFLite Interpreter
    .h5     → Keras Model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"مدل پیدا نشد: {model_path}")

    if model_path.endswith('.tflite'):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        print(f"  [OK] مدل TFLite: {model_path}")
        return ('tflite', interp)
    else:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        print(f"  [OK] مدل Keras: {model_path}")
        return ('keras', model)


def predict_with_model(model_tuple, face_input):
    mtype, model = model_tuple
    if mtype == 'tflite':
        inp = model.get_input_details()
        out = model.get_output_details()
        model.set_tensor(inp[0]['index'], face_input.astype(np.float32))
        model.invoke()
        return model.get_tensor(out[0]['index'])[0]
    else:
        return model.predict(face_input, verbose=0)[0]


# ══════════════════════════════════════════════════════════════════════════
#  Face Detector + Aligner با MediaPipe
# ══════════════════════════════════════════════════════════════════════════

class FaceDetectorMP:
    """
    Face Detection با OpenCV Haar Cascade
    + Face Alignment با Eye Detection

    جایگزین MediaPipe به دلیل ناسازگاری نسخه‌های جدید (>= 0.10.14)
    """

    def __init__(self, min_confidence=0.5):
        # Face detector
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_cascade_path  = cv2.data.haarcascades + 'haarcascade_eye.xml'

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("haarcascade_frontalface_default.xml پیدا نشد.")

        print("  [INFO] Using OpenCV Haar Cascade face detector")

    def detect(self, frame):
        """
        Returns: لیست dict با کلیدهای:
            'bbox'    : (x, y, w, h)
            'aligned' : تصویر چهره تراز شده (BGR)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces = []
        if len(detections) == 0:
            return faces

        for (x, y, w, h) in detections:
            # حاشیهٔ کادر (از config؛ تنگ‌تر = هماهنگ‌تر با تصاویر آموزش)
            pad_x = int(w * FACE_CROP_PADDING)
            pad_y = int(h * FACE_CROP_PADDING)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # تلاش برای alignment با چشم‌ها
            aligned = self._align_face(frame, face_crop, x1, y1, x2, y2, gray)

            faces.append({
                'bbox'   : (x1, y1, x2 - x1, y2 - y1),
                'aligned': aligned
            })

        return faces

    def _align_face(self, frame, face_crop, x1, y1, x2, y2, gray_full):
        """
        Alignment با eye detection:
          1. چشم‌ها را در ناحیه چهره پیدا می‌کند
          2. زاویه بین آن‌ها محاسبه می‌شود
          3. تصویر چرخانده می‌شود
        """
        face_gray = gray_full[y1:y2, x1:x2]
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15)
        )

        # اگر دو چشم پیدا نشد، crop خام برگردان
        if len(eyes) < 2:
            return face_crop

        # مراکز دو چشم اول
        eyes = sorted(eyes, key=lambda e: e[0])  # چپ به راست
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[1]

        cx1 = x1 + ex1 + ew1 // 2
        cy1 = y1 + ey1 + eh1 // 2
        cx2 = x1 + ex2 + ew2 // 2
        cy2 = y1 + ey2 + eh2 // 2

        # زاویه
        dy    = cy2 - cy1
        dx    = cx2 - cx1
        angle = np.degrees(np.arctan2(dy, dx))

        # چرخش حول مرکز چهره
        h_frame, w_frame = frame.shape[:2]
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        face_center = (float(face_center[0]), float(face_center[1]))
        M = cv2.getRotationMatrix2D(face_center, angle, scale=1.0)
        rotated = cv2.warpAffine(frame, M, (w_frame, h_frame),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        aligned = rotated[y1:y2, x1:x2]
        if aligned.size == 0:
            return face_crop

        return aligned

    def close(self):
        pass  # OpenCV نیاز به close ندارد


# ══════════════════════════════════════════════════════════════════════════
#  کلاس اصلی EmotionDetector
# ══════════════════════════════════════════════════════════════════════════

class EmotionDetector:

    def __init__(self, model_path=None):
        if model_path is None:
            # ترتیب اولویت: keras بهترین → keras نهایی → tflite → h5
            candidates = [
                os.path.join(MODELS_DIR, 'fer_model_best.keras'),
                os.path.join(MODELS_DIR, 'fer_model_final.keras'),
                os.path.join(MODELS_DIR, 'fer_model_best.tflite'),
                os.path.join(MODELS_DIR, 'fer_model_best.h5'),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)),
                              candidates[0])

        self.model_tuple   = load_inference_model(model_path)
        self.face_detector = FaceDetectorMP()

        # آستانهٔ هر کلاس (کالیبره‌شده روی validation اگر موجود باشد)
        self.thresholds   = load_class_thresholds()
        self.calibrated   = os.path.exists(THRESHOLDS_PATH)
        if self.calibrated:
            print("  [OK] آستانه‌های کالیبره‌شدهٔ هر کلاس بارگذاری شد.")
            for i, c in enumerate(DISPLAY_NAMES):
                print(f"       {c:10s}: {self.thresholds[i]:.2f}")
        else:
            print(f"  [INFO] آستانهٔ سراسری {CONFIDENCE_THRESHOLD:.2f} "
                  f"(برای کالیبراسیون per-class: python calibrate_thresholds.py)")

        # تنظیمات اصلاح استنتاج زنده (بدون آموزش مجدد)
        self.bias_vec   = np.array(get_class_bias_vector(), dtype=np.float32)
        self.temp       = max(float(INFERENCE_TEMPERATURE), 1e-3)
        self.thr_scale  = float(THRESHOLD_SCALE)
        self._n_smooth  = max(int(TEMPORAL_SMOOTHING), 1)
        self._prob_hist = deque(maxlen=self._n_smooth)   # هموارسازی زمانی
        if not np.allclose(self.bias_vec, 1.0):
            print("  [OK] تصحیح سوگیری کلاس‌ها فعال (neutral کم‌وزن).")
        if self._n_smooth > 1:
            print(f"  [OK] هموارسازی زمانی روی {self._n_smooth} فریم فعال.")

        # آمار برای Confidence Threshold (امتیاز اضافه)
        self.stats = {
            'total'    : 0,
            'uncertain': 0,
            'emotions' : {c: 0 for c in CLASS_NAMES}
        }

        # FPS محاسبه پایدار
        self._frame_times = []

    # ──────────────────────────────────────────────────────────────────
    #  پیش‌پردازش - یکسان با utils.py در مرحله آموزش
    # ──────────────────────────────────────────────────────────────────

    def preprocess_face(self, face_bgr):
        """
        دقیقاً یکسان با پایپ‌لاین آموزش:
          BGR → Grayscale → Resize IMG_SIZE → (1,IMG_SIZE,IMG_SIZE,1) خام 0..255

        نکتهٔ مهم: نرمال‌سازی (تقسیم بر 255) دیگر اینجا انجام نمی‌شود، چون
        لایهٔ Rescaling داخل خود مدل این کار را می‌کند. اگر اینجا هم تقسیم کنیم،
        ورودی دوبار نرمال شده و دقت به‌شدت افت می‌کند.
        """
        gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
        x = resized.astype('float32')          # 0..255  (بدون نرمال‌سازی)
        return x.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # ──────────────────────────────────────────────────────────────────
    #  پیش‌بینی با Confidence Threshold
    # ──────────────────────────────────────────────────────────────────

    def _adjust_probs(self, probs):
        """
        اصلاح خروجی خام مدل برای استفادهٔ زنده (بدون آموزش مجدد):
          ۱) دما (temperature): قاطع‌تر/نرم‌ترکردن توزیع
          ۲) تصحیح سوگیری کلاس‌ها (کم‌وزن‌کردن neutral)
          ۳) هموارسازی زمانی روی چند فریم اخیر
        """
        p = np.asarray(probs, dtype=np.float32).clip(1e-8, 1.0)

        # (۱) تصحیح سوگیری (اول): ضرب در بردار bias و نرمال‌سازی
        p = p * self.bias_vec
        p = p / p.sum()

        # (۲) دما (بعد از بایاس): p^(1/T) سپس نرمال‌سازی (T<1 → قاطع‌تر)
        if abs(self.temp - 1.0) > 1e-3:
            p = p ** (1.0 / self.temp)
            p = p / p.sum()

        # (۳) هموارسازی زمانی: میانگین روی N فریم اخیر
        self._prob_hist.append(p)
        smoothed = np.mean(self._prob_hist, axis=0)
        smoothed = smoothed / smoothed.sum()
        return smoothed

    def predict_emotion(self, face_bgr):
        """
        Returns:
            label      : نام احساس یا 'Uncertain'
            confidence : float [0,1]
            all_probs  : np.ndarray shape=(NUM_CLASSES,)  (اصلاح‌شده و هموار)
        """
        inp      = self.preprocess_face(face_bgr)
        raw      = predict_with_model(self.model_tuple, inp)
        probs    = self._adjust_probs(raw)

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])

        self.stats['total'] += 1

        # آستانهٔ مخصوص همان کلاس × ضریب نرم‌سازی برای استفادهٔ زنده
        if conf < self.thresholds[idx] * self.thr_scale:
            # به‌جای تخصیص اشتباه، "Uncertain" برمی‌گرداند
            self.stats['uncertain'] += 1
            return 'Uncertain', conf, probs

        self.stats['emotions'][CLASS_NAMES[idx]] += 1
        return IDX_TO_DISPLAY[idx], conf, probs

    def get_stats_summary(self):
        """خلاصه آمار Confidence Threshold"""
        total = self.stats['total']
        if total == 0:
            return "هیچ چهره‌ای پردازش نشد."

        unc   = self.stats['uncertain']
        mode  = "per-class (calibrated)" if self.calibrated else \
                f"global {CONFIDENCE_THRESHOLD:.2f}"
        lines = [
            f"  کل پیش‌بینی‌ها : {total}",
            f"  Uncertain      : {unc} ({unc/total*100:.1f}%)  ← threshold: {mode}",
            "  توزیع احساسات :"
        ]
        for cls, cnt in self.stats['emotions'].items():
            if cnt > 0:
                disp = DISPLAY_NAMES[CLASS_NAMES.index(cls)]
                lines.append(f"    {disp:<12}: {cnt:4d}  ({cnt/total*100:.1f}%)")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    #  رسم
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _color_for_display(label):
        """رنگ متناظر یک نام نمایشی (مثل 'Surprise') را از روی نام کلاس برمی‌گرداند."""
        if label in DISPLAY_NAMES:
            return EMOTION_COLORS[CLASS_NAMES[DISPLAY_NAMES.index(label)]]
        return (130, 130, 130)

    def draw_face_box(self, frame, x, y, w, h, label, confidence):
        if label == 'Uncertain':
            color = (130, 130, 130)
            text  = f"Uncertain ({confidence*100:.0f}%)"
        else:
            color = self._color_for_display(label)
            text  = f"{label}  {confidence*100:.0f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame,
                      (x, y - th - 12), (x + tw + 8, y),
                      color, -1)
        cv2.putText(frame, text, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def draw_probability_bars(self, frame, probs, x, y, w, h):
        bar_x    = x + w + 10
        bar_maxw = 120
        bar_h    = 17
        pad      = 5

        if bar_x + bar_maxw + 95 > frame.shape[1]:
            bar_x = max(2, x - bar_maxw - 95)

        total_h = len(CLASS_NAMES) * (bar_h + pad)
        oy      = max(0, min(y, frame.shape[0] - total_h - 5))

        overlay  = frame.copy()
        rect_x2  = bar_x + bar_maxw + 92
        rect_y2  = oy + total_h + 5
        if rect_x2 < frame.shape[1] and rect_y2 < frame.shape[0]:
            cv2.rectangle(overlay,
                          (bar_x - 4, oy - 2),
                          (rect_x2, rect_y2),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (cls, disp, prob) in enumerate(zip(CLASS_NAMES, DISPLAY_NAMES, probs)):
            by = oy + i * (bar_h + pad)
            if by + bar_h >= frame.shape[0]:
                break
            fillw = int(prob * bar_maxw)
            color = EMOTION_COLORS[cls]

            cv2.rectangle(frame,
                          (bar_x, by), (bar_x + bar_maxw, by + bar_h),
                          (55, 55, 55), -1)
            if fillw > 0:
                cv2.rectangle(frame,
                              (bar_x, by), (bar_x + fillw, by + bar_h),
                              color, -1)
            cv2.putText(frame,
                        f"{disp}: {prob*100:.0f}%",
                        (bar_x + bar_maxw + 4, by + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (230, 230, 230), 1, cv2.LINE_AA)

    def draw_fps(self, frame):
        """FPS پایدار با میانگین متحرک روی 30 فریم"""
        now = time.time()
        self._frame_times.append(now)

        # فقط 30 فریم اخیر نگه دار
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / (elapsed + 1e-9)
        else:
            fps = 0.0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 80), 2, cv2.LINE_AA)

    def draw_threshold_indicator(self, frame):
        """
        نمایش Confidence Threshold روی تصویر (امتیاز اضافه)
        به کاربر نشان می‌دهد threshold چقدر است
        """
        txt = ("Threshold: per-class" if self.calibrated
               else f"Threshold: {CONFIDENCE_THRESHOLD:.0%} (global)")
        cv2.putText(frame, txt,
                    (frame.shape[1] - 240, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 50), 1, cv2.LINE_AA)

    def process_frame(self, frame):
        faces = self.face_detector.detect(frame)

        if not faces:
            cv2.putText(frame, "No Face Detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 140, 255), 2, cv2.LINE_AA)
        else:
            for face_info in faces:
                x, y, w, h   = face_info['bbox']
                aligned_face  = face_info['aligned']

                if aligned_face is None or aligned_face.size == 0:
                    continue

                label, conf, probs = self.predict_emotion(aligned_face)
                self.draw_face_box(frame, x, y, w, h, label, conf)
                self.draw_probability_bars(frame, probs, x, y, w, h)

        self.draw_fps(frame)
        self.draw_threshold_indicator(frame)
        return frame

    def close(self):
        self.face_detector.close()


# ══════════════════════════════════════════════════════════════════════════
#  حالت‌های اجرا
# ══════════════════════════════════════════════════════════════════════════

def run_webcam(detector):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("  خطا: دوربین باز نشد.")
        return

    print("  دوربین فعال شد.")
    print("  کلید 'q' → خروج  |  's' → ذخیره فریم")
    snap_dir = os.path.join(RESULTS_DIR, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        cv2.imshow('Facial Emotion Recognition  [q=quit  s=save]', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(snap_dir, f'snap_{ts}.jpg')
            cv2.imwrite(path, result)
            print(f"  ذخیره شد: {path}")

    cap.release()
    cv2.destroyAllWindows()

    # نمایش آمار Confidence Threshold بعد از اتمام
    print("\n" + "─"*45)
    print("  آمار Confidence Threshold:")
    print(detector.get_stats_summary())
    print("─"*45)


def run_on_image(detector, image_path, save_output=True):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  خطا: عکس خوانده نشد: {image_path}")
        return

    result = detector.process_frame(frame)

    if save_output:
        out_dir  = os.path.join(RESULTS_DIR, 'test_outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                                f'result_{os.path.basename(image_path)}')
        cv2.imwrite(out_path, result)
        print(f"  ذخیره شد: {out_path}")

    # نمایش آمار
    print("\n" + "─"*45)
    print(detector.get_stats_summary())
    print("─"*45)

    cv2.imshow(f'Result - {os.path.basename(image_path)}', result)
    print("  هر کلیدی را بزنید تا پنجره بسته شود.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_folder(detector, folder_path):
    out_dir = os.path.join(RESULTS_DIR, 'test_outputs')
    os.makedirs(out_dir, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([f for f in os.listdir(folder_path)
                    if f.lower().endswith(valid_exts)])

    if not files:
        print(f"  هیچ عکسی در {folder_path} پیدا نشد.")
        return

    print(f"  {len(files)} عکس پیدا شد ...")

    for i, fname in enumerate(files, 1):
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is None:
            continue
        result   = detector.process_frame(frame)
        out_path = os.path.join(out_dir, f'result_{fname}')
        cv2.imwrite(out_path, result)
        print(f"  [{i:3d}/{len(files)}]  {fname}")

    # آمار نهایی
    print("\n" + "─"*45)
    print("  آمار Confidence Threshold:")
    print(detector.get_stats_summary())
    print("─"*45)
    print(f"\n  خروجی‌ها در: {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Facial Emotion Recognition - Real-time Inference',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='webcam',
                        choices=['webcam', 'image', 'folder'],
                        help=(
                            'webcam : تشخیص زنده از دوربین\n'
                            'image  : پردازش یک عکس (--path لازم)\n'
                            'folder : پردازش پوشه عکس (--path لازم)'
                        ))
    parser.add_argument('--path',  type=str, default=None,
                        help='مسیر عکس یا پوشه')
    parser.add_argument('--model', type=str, default=None,
                        help='مسیر مدل (.h5 یا .tflite) [اختیاری]')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  Facial Emotion Recognition")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print("="*50)

    detector = EmotionDetector(model_path=args.model)

    try:
        if args.mode == 'webcam':
            run_webcam(detector)
        elif args.mode == 'image':
            if not args.path:
                print("  لطفاً --path را مشخص کنید.")
            else:
                run_on_image(detector, args.path)
        elif args.mode == 'folder':
            if not args.path:
                print("  لطفاً --path را مشخص کنید.")
            else:
                run_on_folder(detector, args.path)
    finally:
        detector.close()
        print("\n  برنامه پایان یافت.")
