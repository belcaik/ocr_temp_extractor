# ocr_temp_extractor

Extract temperature readings from a fixed-lab video by selecting a polygon ROI over the on-screen thermometer and running OCR at a fixed time interval. Outputs a CSV with exactly three columns: `Medicion,Tiempo,temperatura`.

## Features
- Point-based semi-freehand polygon ROI selection on the first frame.
- Time-based sampling using precise timestamp seeks (no frame-by-frame loop).
- Fast preprocessing pipeline (grayscale → resize → denoise → Otsu/adaptive → morphology → deskew with optional sweep).
- OCR via PaddleOCR or Tesseract (select with `--model`); digits-only post-filtering and parsing.
- Optional preview and interactive assist mode to confirm/correct each value.
- CSV export with header: `Medicion,Tiempo,temperatura`.

## Requirements
- Python 3.10+
- Python packages:
  - Core: `opencv-python`, `numpy`, `pillow`
  - OCR engines (choose one or both):
    - Paddle: `paddleocr`, `paddlepaddle`
    - Tesseract: `pytesseract` (requires system Tesseract binary)
  - CSV uses Python stdlib (`csv`), no `pandas` required.

### Install PaddleOCR
- CPU (Linux/macOS/Windows):
  ```bash
  pip install --upgrade pip
  pip install "paddlepaddle>=2.5" paddleocr
  ```
  If `paddlepaddle` fails, consult: https://www.paddlepaddle.org.cn/en/install/quick

### Install Tesseract OCR
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
- Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki and (optionally) set:
  - `setx TESSERACT_CMD "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"`

## Setup
Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python numpy pillow
# Choose OCR engine(s)
pip install "paddlepaddle>=2.5" paddleocr    # for --model paddle
pip install pytesseract                      # for --model tesseract (requires system Tesseract)
```

## Usage
Run the CLI with input video, interval in seconds (float allowed), and output CSV path.

```bash
# Basic (PaddleOCR)
python ocr_temp_extractor.py -i lab_cooling.mp4 -t 2.0 -o data.csv --model paddle

# Using Tesseract
python ocr_temp_extractor.py -i lab_cooling.mp4 -t 2.0 -o data.csv --model tesseract

# With preview window (non-blocking)
python ocr_temp_extractor.py --input in.avi --interval 0.5 --output /tmp/out.csv --preview

# Interactive assist mode (review/correct each value)
python ocr_temp_extractor.py -i in.mp4 -t 1 -o out.csv --assist

# Handling tilted screens / tricky lighting
# - Try a larger scale, adaptive thresholding and deskew sweep
python ocr_temp_extractor.py -i in.mp4 -t 1 -o out.csv --scale 2.5 --adaptive --deskew-sweep

# Edge-preserving denoise (slower but can help 7-seg)
python ocr_temp_extractor.py -i in.mp4 -t 1 -o out.csv --bilateral

# Specify OCR language and allow non-digit chars
# - Paddle: lang like 'en'
# - Tesseract: lang like 'eng'
python ocr_temp_extractor.py -i in.mp4 -t 1 -o out.csv --model paddle --lang en --no-digits-only
python ocr_temp_extractor.py -i in.mp4 -t 1 -o out.csv --model tesseract --lang eng --no-digits-only
```

### ROI Selection Controls
On launching, a window shows the first frame for selecting the temperature display area:
- Left-click: add points
- Backspace or `u`: remove last point
- `r`: reset all points
- Enter: confirm; closes polygon (auto-connects last → first)
- The current polygon is rendered with anti-aliased lines and a semi-transparent fill.
- Points are stored in normalized coordinates so they scale to any frame size.

### Output CSV
- Header (exact): `Medicion,Tiempo,temperatura`
- `Medicion`: 1-based index of the measurement
- `Tiempo`: timestamp in seconds from the start, with three decimals
- `temperatura`: parsed float (e.g., 23.500) or empty if OCR fails

Example:
```csv
Medicion,Tiempo,temperatura
1,0.000,23.500
2,2.000,23.375
3,4.000,
```

### Preview Window
- Shows the thresholded ROI and overlays the raw OCR text and parsed value.
- Press `q` to close the preview; processing continues in the background.

### Assist Mode
Use `--assist` to review each sampled timestamp interactively:
- A small window shows the processed ROI and the OCR result.
- Keys:
  - Enter: accept the current value
  - `e`: edit value (typed in terminal), e.g., `23.45`
  - `s`: set empty (missing/failed OCR)
  - `q`: quit assist for the remaining samples (continues automatically)
- Note: Assist pauses at each sample; `--preview` is disabled automatically while assisting.

### Advanced Preprocessing Flags
- `--scale <float>`: Resize before OCR (default `2.0`). Larger values can improve digit clarity.
- `--adaptive`: Use adaptive thresholding (better for uneven illumination) instead of Otsu.
- `--bilateral`: Use bilateral filter (slower) instead of Gaussian blur; preserves edges.
- `--deskew-sweep`: Try multiple small angles to refine deskew when the screen is slightly tilted.

## Notes & Tips
- The tool seeks by time using `CAP_PROP_POS_MSEC` to avoid drift.
- Minor tilt is corrected via min-area-rect deskew; `--deskew-sweep` further refines.
- If OCR returns extra characters, the parser extracts only the numeric value.
- If a frame cannot be read near the end of the video, processing stops gracefully and writes what was collected.

## Troubleshooting
- PaddleOCR/PaddlePaddle import error: ensure `paddlepaddle` and `paddleocr` are installed and match your Python/OS. On some systems you may need a specific wheel from the PaddlePaddle site.
- Tesseract not found: install it and/or set `TESSERACT_CMD` (Windows path example above).
- Qt plugin error on Wayland: the script sets `QT_QPA_PLATFORM=xcb` automatically on Linux/Wayland; you can also `export QT_QPA_PLATFORM=xcb`.
- Video won’t open: check the file path and that the codec is supported by OpenCV/FFmpeg.
- Poor OCR accuracy:
  - Ensure the ROI tightly bounds the digits.
  - Try `--scale`, `--adaptive`, `--deskew-sweep`, or `--bilateral`.
  - Reduce glare/contrast; the pipeline uses Otsu/adaptive + morphology to clean 7‑seg digits.

## Development
Run from source:
```bash
. .venv/bin/activate
python ocr_temp_extractor.py -i input.mp4 -t 1.0 -o out.csv --preview
```
