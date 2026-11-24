"""
quality.py - measure how a PDF looks, before any model reads it.

These measurements need no ground truth. They describe the *input*, so that monitor.py
can notice documents (or whole batches) that look unlike the ones the pipeline was
tested on: blank pages, very dark or faded scans, blur, tilt, low resolution, and so on.

All image measurements are taken on the first page rendered in grey-scale at a fixed
resolution (ANALYSIS_DPI), so that values are comparable between documents.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

ANALYSIS_DPI = 100   # fixed, so measurements like sharpness are comparable between files
INK_MARGIN = 40      # a pixel is "ink" if it is this much darker than the page background.
                     # Relative, not absolute, so faint grey print on thermal receipts still counts.


def measure(path: Path) -> dict:
    """Return a dictionary of input-quality measurements for one PDF.

    pages          number of pages
    file_kb        file size
    effective_dpi  resolution of the largest embedded image (None for text-only PDFs)
    ink_ratio      share of dark pixels; ~0 means a blank page
    brightness     average grey value, 0 (black) to 255 (white)
    contrast       spread of grey values; low = faded or uniformly dark
    sharpness      how strong fine edges are; low = blurry
    skew_degrees   estimated tilt of the text lines
    """
    with pymupdf.open(path) as document:
        page = document[0]
        pixmap = page.get_pixmap(dpi=ANALYSIS_DPI, colorspace=pymupdf.csGRAY)
        grey = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width).astype(np.float32)
        return {
            "pages": document.page_count,
            "file_kb": round(path.stat().st_size / 1024, 1),
            "effective_dpi": effective_dpi(document, page),
            "ink_ratio": ink_ratio(grey),
            "brightness": round(float(grey.mean()), 2),
            "contrast": round(float(grey.std()), 2),
            "sharpness": round(sharpness(grey), 2),
            "skew_degrees": skew_degrees(grey),
        }


def effective_dpi(document, page) -> float | None:
    """Resolution of the page's largest image: image pixels / page inches.

    A scan at 50 dpi is much harder to read than one at 200 dpi. Returns None if the
    page has no images (a digital PDF with a text layer).
    """
    best = None
    for image in page.get_images(full=True):
        xref, width_px = image[0], image[2]
        rects = page.get_image_rects(xref)
        if not rects or rects[0].width <= 0:
            continue
        dpi = width_px / (rects[0].width / 72)  # PDF sizes are in points; 72 points = 1 inch
        best = dpi if best is None else max(best, dpi)
    return round(best, 1) if best else None


def ink_ratio(grey: np.ndarray) -> float:
    """Share of pixels that are clearly darker than the page background (~0 = blank page).

    The background is taken as the median grey value, since most of a document page is
    paper. Measuring relative to it means faint print and dark scans are both handled.
    """
    background = float(np.median(grey))
    return round(float((grey < background - INK_MARGIN).mean()), 5)


def sharpness(grey: np.ndarray) -> float:
    """Variance of the Laplacian: a standard blur measure.

    The Laplacian responds to rapid changes in brightness (edges of letters). A sharp
    page has many strong responses (high variance); a blurred page has few.
    """
    laplacian = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
                 - 4 * grey[1:-1, 1:-1])
    return float(laplacian.var())


def skew_degrees(grey: np.ndarray, max_angle: float = 15.0, step: float = 0.5) -> float:
    """Estimate the tilt of the text lines, in degrees.

    Method (projection profile): rotate the page by a range of angles; when the text
    lines are exactly horizontal, the row sums of dark pixels form sharp peaks (text
    rows) and gaps (between rows), so their variance is largest. The best angle,
    reversed, is the page's tilt.
    """
    background = float(np.median(grey))
    ink = Image.fromarray(((grey < background - INK_MARGIN) * 255).astype(np.uint8))  # same mask as ink_ratio
    ink.thumbnail((500, 500))  # smaller image = much faster, still accurate to ~0.5 degree
    if np.asarray(ink).mean() < 0.5:  # blank page: no text to measure
        return 0.0
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step / 2, step):
        rotated = np.asarray(ink.rotate(angle, resample=Image.BILINEAR, fillcolor=0), dtype=np.float32)
        score = rotated.sum(axis=1).var()
        if score > best_score:
            best_angle, best_score = float(angle), score
    return round(-best_angle, 1)
