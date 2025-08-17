import os
import sys
import re
import csv
import math
import argparse
from argparse import BooleanOptionalAction
from typing import List, Tuple, Optional, Dict

import numpy as np

# On many Linux Wayland setups, OpenCV's Qt plugin for Wayland isn't bundled.
# Prefer XCB if running under Wayland unless the user explicitly sets a platform.
if sys.platform.startswith("linux") and "WAYLAND_DISPLAY" in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2

# PaddleOCR is optional (only required when using --model paddle)
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None  # type: ignore


_ocr_cache: Dict[str, "PaddleOCR"] = {}


def get_paddle_ocr(lang: str = "en") -> "PaddleOCR":
    if PaddleOCR is None:
        raise RuntimeError("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
    # Map common tesseract-like tags to PaddleOCR
    if lang == "eng":
        lang = "en"
    if lang not in _ocr_cache:
        # Try modern API first, then fall back for older versions
        try:
            _ocr_cache[lang] = PaddleOCR(lang=lang, use_textline_orientation=True)
        except TypeError:
            try:
                _ocr_cache[lang] = PaddleOCR(lang=lang, use_angle_cls=True)
            except TypeError:
                _ocr_cache[lang] = PaddleOCR(lang=lang)
    return _ocr_cache[lang]


# -----------------------------
# ROI Selection GUI
# -----------------------------
class ROISelector:
    def __init__(self, frame: np.ndarray, window_name: str = "Select ROI"):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.window = window_name
        self.points: List[Tuple[int, int]] = []
        self.done = False
        self.cancelled = False
        # selection modes
        self.mode = 'poly'  # 'poly' or 'rect'
        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        # cache of current display scale to map mouse -> image coords
        self._last_disp_size: Optional[Tuple[int, int]] = None

    def _map_mouse_to_image(self, x: int, y: int) -> Tuple[int, int]:
        # Map mouse coordinates (possibly on a resized window) back to image coordinates
        disp_w, disp_h = None, None
        try:
            if hasattr(cv2, 'getWindowImageRect'):
                _, _, disp_w, disp_h = cv2.getWindowImageRect(self.window)
        except Exception:
            disp_w, disp_h = None, None
        if disp_w is None or disp_h is None or disp_w <= 0 or disp_h <= 0:
            sx = sy = 1.0
        else:
            self._last_disp_size = (disp_w, disp_h)
            sx = float(self.w) / float(disp_w)
            sy = float(self.h) / float(disp_h)
        xi = int(round(x * sx))
        yi = int(round(y * sy))
        xi = max(0, min(self.w - 1, xi))
        yi = max(0, min(self.h - 1, yi))
        return xi, yi

    def _on_mouse(self, event, x, y, flags, param):
        if self.mode == 'poly':
            if event == cv2.EVENT_LBUTTONDOWN:
                xi, yi = self._map_mouse_to_image(x, y)
                self.points.append((xi, yi))
        else:
            # Rectangle mode: two left-clicks (no drag) to avoid Qt pan
            if event == cv2.EVENT_LBUTTONDOWN:
                xi, yi = self._map_mouse_to_image(x, y)
                if not self.dragging:
                    # First click starts preview
                    self.dragging = True
                    self.drag_start = (xi, yi)
                    self.drag_current = (xi, yi)
                else:
                    # Second click finalizes rectangle
                    self.drag_current = (xi, yi)
                    if self.drag_start is not None and self.drag_current is not None:
                        x0, y0 = self.drag_start
                        x1, y1 = self.drag_current
                        p0 = (x0, y0)
                        p1 = (x1, y0)
                        p2 = (x1, y1)
                        p3 = (x0, y1)
                        self.points = [p0, p1, p2, p3]
                    self.dragging = False
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                xi, yi = self._map_mouse_to_image(x, y)
                self.drag_current = (xi, yi)

    def _draw(self, img: np.ndarray) -> np.ndarray:
        overlay = img.copy()
        # Draw live polyline
        if self.mode == 'poly':
            if len(self.points) > 0:
                for i in range(1, len(self.points)):
                    cv2.line(overlay, self.points[i - 1], self.points[i], (0, 255, 255), 2, cv2.LINE_AA)
                # last point marker
                cv2.circle(overlay, self.points[-1], 3, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            if self.dragging and self.drag_start and self.drag_current:
                x0, y0 = self.drag_start
                x1, y1 = self.drag_current
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
            elif len(self.points) == 4:
                pts = np.array(self.points, dtype=np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        # Fill polygon if >=3
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            # blend
            overlay = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Instructions box
        if self.mode == 'poly':
            instructions = [
                "Mode: Polygon (b: rectangle, p: polygon)",
                "Left-click: add point",
                "Backspace/u: undo, r: reset",
                "Enter: confirm",
            ]
        else:
            instructions = [
                "Mode: Rectangle (click corner, move, click opposite)",
                "b: rectangle, p: polygon",
                "Backspace/u: undo, r: reset",
                "Enter: confirm",
            ]
        y0 = 20
        for line in instructions:
            cv2.putText(
                overlay,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y0 += 20
        return overlay

    def select(self) -> Optional[List[Tuple[float, float]]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)
        base = self.frame.copy()

        while True:
            disp = self._draw(base)
            cv2.imshow(self.window, disp)
            key = cv2.waitKey(20) & 0xFF

            # Close window button handling
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                self.cancelled = True
                break

            if key == 13 or key == 10:  # Enter
                if len(self.points) >= 3:
                    self.done = True
                    break
                else:
                    print("[warn] Need at least 3 points to confirm.", file=sys.stderr)
            elif key == 8 or key == ord('u') or key == ord('U'):  # Backspace or u
                if self.points:
                    self.points.pop()
            elif key == ord('r') or key == ord('R'):
                self.points.clear()
                self.dragging = False
                self.drag_start = None
                self.drag_current = None
            elif key == ord('b') or key == ord('B'):
                self.mode = 'rect'
                self.points.clear()
                self.dragging = False
                self.drag_start = None
                self.drag_current = None
            elif key == ord('p') or key == ord('P'):
                self.mode = 'poly'
                self.points.clear()
                self.dragging = False
                self.drag_start = None
                self.drag_current = None

        cv2.destroyWindow(self.window)

        if not self.done:
            return None

        # Normalize
        norm = [(x / self.w, y / self.h) for (x, y) in self.points]
        return norm


def reconstruct_polygon(norm_pts: List[Tuple[float, float]], w: int, h: int) -> np.ndarray:
    pts = np.array([(int(round(x * w)), int(round(y * h))) for (x, y) in norm_pts], dtype=np.int32)
    return pts


# -----------------------------
# Image preprocessing and OCR
# -----------------------------
def preprocess_for_ocr(
    gray: np.ndarray,
    scale: float = 2.0,
    do_bilateral: bool = False,
    use_adaptive: bool = False,
    sweep_refine: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # Resize to aid OCR
    if scale and scale != 1.0:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)

    if do_bilateral:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    if use_adaptive:
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: open then close small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    th, angle = deskew_binary(th)

    # Optional sweep-based refinement for tricky small tilts
    if sweep_refine:
        th, add_ang = refine_deskew_sweep(th)
        angle += add_ang

    # Ensure black text on white background for Tesseract
    if np.mean(th) > 127:
        th = 255 - th

    return th, angle


def deskew_binary(bin_img: np.ndarray, max_abs_angle: float = 15.0) -> Tuple[np.ndarray, float]:
    # Find contours; pick the largest component for angle
    contours, _ = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bin_img, 0.0
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:  # too small
        return bin_img, 0.0
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    # Convert OpenCV angle to human-friendly small tilt
    if angle < -45:
        angle = angle + 90
    # Rotate if small angle
    if abs(angle) <= max_abs_angle and abs(angle) > 0.5:
        rotated = rotate_bound(bin_img, -angle)
        return rotated, angle
    return bin_img, 0.0


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # compute the new bounding dimensions of the image
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def refine_deskew_sweep(bin_img: np.ndarray) -> Tuple[np.ndarray, float]:
    # Ensure foreground is white for projection scoring
    img = bin_img.copy()
    if np.sum(img) < (img.size * 255 / 2):
        img = 255 - img
    best_score = -1.0
    best_angle = 0.0
    best_img = bin_img
    # Try small angles around 0
    for ang in range(-10, 11, 2):
        if ang == 0:
            test = img
        else:
            test = rotate_bound(img, ang)
            # Keep binary
            _, test = cv2.threshold(test, 127, 255, cv2.THRESH_BINARY)
        # Column projection variance (maximize)
        col_proj = np.sum(test // 255, axis=0).astype(np.float32)
        score = float(np.var(col_proj))
        if score > best_score:
            best_score = score
            best_angle = float(ang)
            best_img = test
    # If we chose a rotation, apply to original bin to avoid cumulative artifacts
    if abs(best_angle) > 0.5:
        rotated = rotate_bound(bin_img, best_angle)
        _, rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
        return rotated, best_angle
    return bin_img, 0.0


def ocr_text_paddle(bin_img: np.ndarray, lang: str = "en", digits_only: bool = True) -> str:
    try:
        ocr = get_paddle_ocr(lang)
        # PaddleOCR accepts numpy array; prefers RGB, but works with grayscale too
        img = bin_img
        if len(img.shape) == 2:
            # Convert to 3-channel for consistency
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Paddle expects RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Handle API differences across PaddleOCR versions
        try:
            result = ocr.ocr(img, cls=True)
        except TypeError:
            try:
                result = ocr.ocr(img, use_angle_cls=True)
            except TypeError:
                result = ocr.ocr(img)
    except Exception as e:
        print(f"[warn] OCR failed: {e}", file=sys.stderr)
        return ""

    # Flatten results and pick the highest-confidence text
    candidates: List[Tuple[str, float]] = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, list):
                for det in item:
                    try:
                        txt = det[1][0]
                        conf = float(det[1][1])
                        candidates.append((txt, conf))
                    except Exception:
                        continue
            elif isinstance(item, dict):
                # Newer APIs may return dict with 'transcription'/'score'
                txt = item.get('transcription') or item.get('text')
                conf = item.get('score') or item.get('confidence')
                if txt is not None and conf is not None:
                    try:
                        candidates.append((str(txt), float(conf)))
                    except Exception:
                        pass
            else:
                # Some versions return a single-level list of tuples
                try:
                    txt = item[1][0]
                    conf = float(item[1][1])
                    candidates.append((txt, conf))
                except Exception:
                    pass
    if not candidates:
        return ""
    text = max(candidates, key=lambda x: x[1])[0]
    if digits_only:
        # Keep only digits/.- and spaces
        text = re.sub(r"[^0-9\.-]+", " ", text)
    return text.strip()


def configure_tesseract_from_env() -> None:
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        try:
            import pytesseract  # type: ignore
            pytesseract.pytesseract.tesseract_cmd = env_cmd
        except Exception:
            pass
    try:
        import pytesseract  # type: ignore
        _ = pytesseract.get_tesseract_version()
    except Exception:
        hint = (
            "Tesseract OCR not found. Install it and/or set TESSERACT_CMD. "
            "Windows example: TESSERACT_CMD=\"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\""
        )
        raise RuntimeError(hint)


def ocr_text_tesseract(bin_img: np.ndarray, lang: str = "eng", digits_only: bool = True) -> str:
    try:
        import pytesseract  # type: ignore
    except Exception as e:
        print(f"[warn] pytesseract not available: {e}", file=sys.stderr)
        return ""
    allowed = "0123456789.-"
    if not digits_only:
        allowed += "°CcdC"
    config = f"--psm 7 -c tessedit_char_whitelist={allowed}"
    try:
        text = pytesseract.image_to_string(bin_img, lang=lang, config=config)
    except Exception as e:
        print(f"[warn] OCR failed: {e}", file=sys.stderr)
        return ""
    return text.strip()


def parse_temperature(text: str) -> Optional[float]:
    if not text:
        return None
    # Normalize decimal separator and strip units
    text = text.replace("\\n", " ").replace("°", " ").replace("C", " ").replace("c", " ")
    m = re.search(r"-?\d+(?:[\.,]\d+)?", text)
    if not m:
        return None
    num = m.group(0).replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


# -----------------------------
# Video sampling and extraction
# -----------------------------
def get_video_first_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def estimate_duration_seconds(cap: cv2.VideoCapture) -> Optional[float]:
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frames > 0 and fps and fps > 0:
        return float(frames) / float(fps)
    # Fallback: unknown
    return None


def extract_roi(frame: np.ndarray, poly_pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)
    x, y, w, h = cv2.boundingRect(poly_pts)
    roi_frame = frame[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]
    roi = cv2.bitwise_and(roi_frame, roi_frame, mask=roi_mask)
    return roi


def process_video(
    input_path: str,
    interval_sec: float,
    norm_poly: List[Tuple[float, float]],
    output_csv: str,
    model: str = "paddle",
    preview: bool = False,
    lang: str = "eng",
    digits_only: bool = True,
    assist: bool = False,
    scale: float = 2.0,
    bilateral: bool = False,
    adaptive: bool = False,
    deskew_sweep: bool = False,
    rotate_deg: float = 0.0,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    duration = estimate_duration_seconds(cap)
    if duration is None:
        print("[warn] Unknown video duration; will sample until read fails.", file=sys.stderr)

    # Prepare CSV
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    t = 0.0
    measurement_idx = 1

    preview_enabled = preview
    assist_enabled = assist
    if preview_enabled:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    if assist_enabled:
        cv2.namedWindow("Assist", cv2.WINDOW_NORMAL)

    while True:
        # Stop if known duration exceeded (with a tiny epsilon)
        if duration is not None and t > duration + 1e-3:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            # likely end of video
            break

        # Apply global rotation before ROI if requested, to match selection space
        if rotate_deg and abs(rotate_deg) > 0.1:
            frame = rotate_bound(frame, rotate_deg)

        h, w = frame.shape[:2]
        poly_pts = reconstruct_polygon(norm_poly, w, h)
        roi = extract_roi(frame, poly_pts)

        if roi.size == 0:
            temp_val = None
            print(f"[warn] Empty ROI at t={t:.3f}s", file=sys.stderr)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bin_img, angle = preprocess_for_ocr(
                gray,
                scale=scale,
                do_bilateral=bilateral,
                use_adaptive=adaptive,
                sweep_refine=deskew_sweep,
            )
            # OCR engine selection
            if model == 'paddle':
                text = ocr_text_paddle(bin_img, lang=("en" if lang == "eng" else lang), digits_only=digits_only)
            else:
                t_lang = ("eng" if lang == "en" else lang)
                text = ocr_text_tesseract(bin_img, lang=t_lang, digits_only=digits_only)
            temp_val = parse_temperature(text)

            if preview_enabled:
                disp = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
                overlay_text = text if text else "(no text)"
                if temp_val is not None:
                    overlay_text += f"  => {temp_val:.3f}"
                cv2.putText(disp, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Preview", disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    preview_enabled = False
                    try:
                        cv2.destroyWindow("Preview")
                    except Exception:
                        pass

            if assist_enabled:
                disp = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
                overlay_text = text if text else "(no text)"
                if temp_val is not None:
                    overlay_text += f"  => {temp_val:.3f}"
                # Show timestamp and controls
                cv2.putText(disp, f"t={t:.3f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp, overlay_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, "Enter: accept | e: edit | s: empty | q: quit assist", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Assist", disp)
                while True:
                    k = cv2.waitKey(0) & 0xFF
                    if k in (13, 10):  # Enter -> accept current
                        break
                    elif k == ord('e'):
                        try:
                            cv2.destroyWindow("Assist")
                        except Exception:
                            pass
                        try:
                            user_in = input(f"[assist] t={t:.3f}s enter corrected temperature (blank = keep): ")
                        except EOFError:
                            user_in = ""
                        # restore window
                        cv2.namedWindow("Assist", cv2.WINDOW_NORMAL)
                        cv2.imshow("Assist", disp)
                        if user_in.strip():
                            new_val = parse_temperature(user_in.strip())
                            if new_val is None:
                                print("[warn] Invalid number; keeping previous value.", file=sys.stderr)
                            else:
                                temp_val = new_val
                        continue
                    elif k == ord('s'):
                        temp_val = None
                        break
                    elif k == ord('q'):
                        assist_enabled = False
                        try:
                            cv2.destroyWindow("Assist")
                        except Exception:
                            pass
                        break

        rows.append({
            'Medicion': measurement_idx,
            'Tiempo': f"{t:.3f}",
            'temperatura': ("" if temp_val is None else f"{temp_val:.3f}")
        })

        measurement_idx += 1
        t += interval_sec

    cap.release()
    if preview:
        try:
            cv2.destroyWindow("Preview")
        except Exception:
            pass
    if assist:
        try:
            cv2.destroyWindow("Assist")
        except Exception:
            pass

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Medicion', 'Tiempo', 'temperatura'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[info] Wrote {len(rows)} rows to {output_csv}", file=sys.stderr)


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ocr_temp_extractor", description="Extract temperature OCR from video ROI over time.")
    p.add_argument('-i', '--input', required=True, help='Path to input video (mp4 or OpenCV-readable).')
    p.add_argument('-t', '--interval', type=float, required=True, help='Sampling interval in seconds (float).')
    p.add_argument('-o', '--output', required=True, help='Path to output CSV.')
    p.add_argument('--preview', action='store_true', help='Show processed ROI preview during sampling (non-blocking).')
    p.add_argument('--assist', action='store_true', help='Interactive review of each OCR value with ability to correct/skip.')
    p.add_argument('--model', choices=['paddle', 'tesseract'], default='paddle', help='OCR engine to use (default: paddle).')
    p.add_argument('--lang', default='en', help='OCR language code (paddle: en, tesseract: eng).')
    p.add_argument('--digits-only', action=BooleanOptionalAction, default=True, help='Restrict OCR to digits and \'.-\' (default: true).')
    p.add_argument('--scale', type=float, default=2.0, help='Resize scale before OCR preprocessing (default: 2.0).')
    p.add_argument('--adaptive', action='store_true', help='Use adaptive thresholding instead of Otsu.')
    p.add_argument('--bilateral', action='store_true', help='Use bilateral filter (slower, preserves edges).')
    p.add_argument('--deskew-sweep', action='store_true', help='Try small-angle sweep to refine deskew for tilted screens.')
    p.add_argument('--rotate', type=float, default=0.0, help='Rotate ROI by degrees (CCW) before OCR to compensate video tilt.')
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.input):
        print(f"[error] Input video not found: {args.input}", file=sys.stderr)
        return 3

    if args.interval <= 0:
        print("[error] --interval must be > 0", file=sys.stderr)
        return 3

    # Initialize OCR engine based on selection
    if args.model == 'paddle':
        try:
            _ = get_paddle_ocr(args.lang)
        except Exception as e:
            print(f"[error] Failed to initialize PaddleOCR: {e}", file=sys.stderr)
            return 2
    else:
        try:
            configure_tesseract_from_env()
        except RuntimeError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2

    # Open video and get first frame
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[error] Cannot open video: {args.input}", file=sys.stderr)
        return 4

    first = get_video_first_frame(cap)
    if first is None:
        print("[error] Cannot read first frame for ROI selection.", file=sys.stderr)
        cap.release()
        return 4

    # Apply rotation to first frame for selection if requested
    if abs(args.rotate) > 0.1:
        first = rotate_bound(first, args.rotate)

    print("[info] Select ROI (polygon or rectangle). Press Enter to confirm.", file=sys.stderr)
    selector = ROISelector(first)
    norm_poly = selector.select()
    if norm_poly is None:
        print("[error] ROI selection cancelled or invalid.", file=sys.stderr)
        cap.release()
        return 5
    if len(norm_poly) < 3:
        print("[error] Need a polygon with at least 3 points.", file=sys.stderr)
        cap.release()
        return 5

    cap.release()

    try:
        process_video(
            input_path=args.input,
            interval_sec=args.interval,
            norm_poly=norm_poly,
            output_csv=args.output,
            model=args.model,
            preview=(False if args.assist else args.preview),
            lang=args.lang,
            digits_only=args.digits_only,
            assist=args.assist,
            scale=args.scale,
            bilateral=args.bilateral,
            adaptive=args.adaptive,
            deskew_sweep=args.deskew_sweep,
            rotate_deg=args.rotate,
        )
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 6

    return 0


if __name__ == '__main__':
    sys.exit(main())
