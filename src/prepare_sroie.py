"""
prepare_sroie.py - turn the SROIE receipt images into PDFs plus a table of correct answers.

Usage:
    python -I src/prepare_sroie.py

Input:  data/raw/sroie_repo/data/img/*.jpg   receipt scans
        data/raw/sroie_repo/data/key/*.json  official labels (company, date, address, total)
        (a copy of github.com/zzzDavid/ICDAR-2019-SROIE, the ICDAR 2019 training set
        with label corrections)

Output: data/sroie/pdf/sroie_000.pdf ...      one PDF per receipt (image only, no text layer)
        data/sroie/ground_truth/receipts.csv  receipt_id, company, address, date, date_iso, total

Two date columns are kept:
    date      exactly as labelled, e.g. "25/12/2018" (what the official benchmark scores)
    date_iso  converted to YYYY-MM-DD, e.g. "2018-12-25" (what our model is asked to output)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image


# Date formats seen in the SROIE labels, tried in this order. These are Malaysian
# receipts, so day-first formats come first. The US month-first format is tried last,
# so it is only used when day-first is impossible (e.g. "12/28/2017": there is no month 28).
DATE_FORMATS = [
    "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%b-%y", "%d %b %y", "%b %d, %Y", "%d%b%Y", "%d/%b/%Y",
    "%Y%m%d", "%d%m%Y",
    "%m/%d/%Y",
]


def to_iso_date(raw: str) -> str:
    """Convert a labelled date like "25/12/2018" or "10 MAR 2018" to "YYYY-MM-DD".

    Returns "" if no format fits. A result outside 2000-2030 is treated as a misread
    format and the next format is tried.
    """
    # Tidy up: drop surrounding brackets, collapse spaces, and use "Mar" capitalisation
    # so month names match strptime's %b / %B.
    cleaned = re.sub(r"\s+", " ", raw.strip().strip("()")).title()
    for date_format in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, date_format)
        except ValueError:
            continue  # this format does not fit; try the next one
        if 2000 <= parsed.year <= 2030:
            return parsed.date().isoformat()
    return ""


def image_to_pdf(image_path: Path, pdf_path: Path, dpi: int = 200) -> None:
    """Wrap one image in a single-page PDF.

    The page size is set from the image's pixel size at the given dpi, so the receipt
    keeps its real (long, narrow) shape. PDF sizes are in points: 72 points = 1 inch.
    """
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    document = pymupdf.open()  # a new, empty PDF
    page = document.new_page(width=width_px * 72 / dpi, height=height_px * 72 / dpi)
    page.insert_image(page.rect, filename=str(image_path))
    document.save(pdf_path, deflate=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("data/raw/sroie_repo/data"))
    parser.add_argument("--out", type=Path, default=Path("data/sroie"))
    args = parser.parse_args()

    pdf_dir = args.out / "pdf"
    truth_dir = args.out / "ground_truth"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_path in sorted((args.src / "key").glob("*.json")):
        receipt_number = label_path.stem  # e.g. "000"

        # Find the matching image (normally .jpg, but accept any extension).
        images = list((args.src / "img").glob(f"{receipt_number}.*"))
        if not images:
            print(f"skip {receipt_number}: no image")
            continue
        image_to_pdf(images[0], pdf_dir / f"sroie_{receipt_number}.pdf")

        label = json.loads(label_path.read_text(encoding="utf-8"))
        # Keep only digits and the decimal point: "RM 1,234.50" or "$8.20" -> "1234.50" / "8.20".
        # One receipt (033) has an empty total in the official labels; it stays "".
        total = re.sub(r"[^\d.]", "", label.get("total", ""))

        rows.append({
            "receipt_id": f"sroie_{receipt_number}",
            "company": label.get("company", ""),
            "address": label.get("address", ""),
            "date": label.get("date", ""),
            "date_iso": to_iso_date(label.get("date", "")),
            "total": total,
        })

    table = pd.DataFrame(rows)
    table.to_csv(truth_dir / "receipts.csv", index=False, encoding="utf-8")

    # Report any date we could not convert, so a new format can be added to DATE_FORMATS.
    unconverted = table[table.date_iso == ""]
    print(f"wrote {len(table)} receipts; {len(unconverted)} dates not normalized")
    if len(unconverted):
        print(unconverted[["receipt_id", "date"]].to_string(index=False))


if __name__ == "__main__":
    main()
