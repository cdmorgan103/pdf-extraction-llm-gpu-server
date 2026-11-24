"""
make_anomalies.py - build a test set of unusual scans, and score how monitoring handles them.

Usage:
    python src/make_anomalies.py generate            # writes data/anomalies/
    python src/extract.py --input data/anomalies/pdf --schema invoice --model qwen2.5vl:7b \
        --text-mode image --num-ctx 16384 --run anomaly_test_vision
    python src/make_anomalies.py report --run anomaly_test_vision

What it generates (all from *new* invoices, seed 2026, so nothing overlaps the data the
baselines were built from, except the deliberate duplicates):

    NORM-xx                 normal scans, to measure how often good documents get flagged
    ANOM-<type>-x           3 of each unusual type:
        blank           an empty scanned page
        upside_down     page rotated 180 degrees
        heavy_tilt      page rotated ~12 degrees
        extreme_blur    badly out of focus
        very_dark       very dark scan
        very_faded      almost white, barely visible print
        heavy_noise     strong noise and stains
        cropped         bottom half missing (totals cut off)
        low_res         scanned at ~45 dpi
        not_invoice     a letter, not an invoice
        merged          two different invoices in one PDF
        duplicate       an exact copy of an invoice already processed earlier

A document is "handled" if it was flagged for review OR extracted fully correctly
(an unusual scan the model still reads perfectly doesn't need a person).
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

import config
from evaluate import INVOICE_FIELDS, LINE_ITEM_FIELDS, compare
from generate_invoices import make_invoice, make_scanned, render_pdf

OUT = Path("data/anomalies")
SCAN_DPI = 150
ANOMALY_TYPES = ["blank", "upside_down", "heavy_tilt", "extreme_blur", "very_dark", "very_faded",
                 "heavy_noise", "cropped", "low_res", "not_invoice", "merged", "duplicate"]
# Types where the invoice's content is still all there, so the correct answers still apply.
CONTENT_INTACT = {"upside_down", "heavy_tilt", "extreme_blur", "very_dark", "very_faded",
                  "heavy_noise", "low_res", "duplicate"}
DUPLICATE_SOURCES = ["INV-0002", "INV-0012", "INV-0027"]  # scanned invoices already in the database


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_bytes: bytes, dpi: int = SCAN_DPI) -> tuple[list[Image.Image], list[tuple[float, float]]]:
    """Render each page to a grey-scale image. Also returns each page's size in points."""
    images, sizes = [], []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            images.append(Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples))
            sizes.append((page.rect.width, page.rect.height))
    return images, sizes


def images_to_pdf(images: list[Image.Image], sizes: list[tuple[float, float]]) -> bytes:
    """Put images back into a PDF, one per page, each filling a page of the given size."""
    output = pymupdf.open()
    for image, (width, height) in zip(images, sizes):
        buffer = io.BytesIO()
        image.convert("L").save(buffer, format="JPEG", quality=70)
        page = output.new_page(width=width, height=height)
        page.insert_image(page.rect, stream=buffer.getvalue())
    return output.tobytes(deflate=True)


def adjust(image: Image.Image, function) -> Image.Image:
    """Apply a function to the pixel values (as floats) and clip back to 0-255."""
    pixels = function(np.asarray(image, dtype=np.float32))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# The anomalies
# ---------------------------------------------------------------------------

def make_anomaly(kind: str, scanned_pdf: bytes, rng: random.Random, np_rng: np.random.Generator,
                 other_scanned_pdf: bytes | None = None) -> bytes:
    """Turn a normal scanned invoice into one unusual document of the given kind."""
    images, sizes = pdf_to_images(scanned_pdf)
    first = images[0]

    if kind == "blank":
        noise = np_rng.normal(245, 6, (first.height, first.width))
        images = [adjust(first, lambda _: noise)]
    elif kind == "upside_down":
        images = [image.rotate(180) for image in images]
    elif kind == "heavy_tilt":
        angle = rng.choice([-1, 1]) * rng.uniform(10, 14)
        images = [image.rotate(angle, resample=Image.BICUBIC, fillcolor=255) for image in images]
    elif kind == "extreme_blur":
        images = [image.filter(ImageFilter.GaussianBlur(rng.uniform(3.5, 5))) for image in images]
    elif kind == "very_dark":
        images = [adjust(image, lambda p: p * 0.35) for image in images]
    elif kind == "very_faded":
        images = [adjust(image, lambda p: 255 - (255 - p) * 0.12) for image in images]
    elif kind == "heavy_noise":
        noisy = []
        for image in images:
            pixels = np.asarray(image, dtype=np.float32) + np_rng.normal(0, 45, (image.height, image.width))
            stained = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(stained)
            for _ in range(6):  # coffee-cup style stains
                x, y, r = rng.randint(0, image.width), rng.randint(0, image.height), rng.randint(60, 180)
                draw.ellipse((x - r, y - r, x + r, y + r), outline=90, width=rng.randint(8, 20))
            noisy.append(stained)
        images = noisy
    elif kind == "cropped":
        # Cut through the printed content, not the page: short invoices leave the bottom of
        # the page empty, so cropping a fixed share of the page might remove only blank paper.
        # Keep the top 70% of the content (header, parties, most items); totals are lost.
        pixels = np.asarray(first)
        ink_rows = np.where((pixels < 160).mean(axis=1) > 0.01)[0]  # >1% dark: text, not dust specks
        top, bottom = int(ink_rows.min()), int(ink_rows.max())
        cut_at = top + int(0.7 * (bottom - top))
        cropped = first.copy()
        cropped.paste(255, (0, cut_at, first.width, first.height))
        images = [cropped]
    elif kind == "low_res":
        # Embed a small image: ~45 dpi when stretched over the page.
        width_in = sizes[0][0] / 72
        small_width = int(45 * width_in)
        images = [image.resize((small_width, int(image.height * small_width / image.width)), Image.BILINEAR)
                  for image in images]
    elif kind == "merged":
        other_images, other_sizes = pdf_to_images(other_scanned_pdf)
        images, sizes = images + other_images, sizes + other_sizes
    else:
        raise ValueError(kind)
    return images_to_pdf(images, sizes)


def make_letter(rng: random.Random) -> bytes:
    """A one-page business letter: a real document, but not an invoice."""
    style = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=15)
    paragraphs = [
        "Dear Ms Carter,",
        "Thank you for meeting with us last week to discuss the renovation of the community library. "
        "We were encouraged by the level of interest from local residents and the support of the council.",
        "As agreed, we will circulate a short questionnaire to members over the coming weeks and share "
        "the results at the next meeting. The working group will also visit two comparable libraries "
        "to learn from their experience with extended opening hours.",
        "Please let me know if you would like to add any items to the agenda.",
        "Yours sincerely,", "Daniel Hughes", "Secretary, Friends of the Library",
    ]
    rng.shuffle(paragraphs[1:4])
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=LETTER)
    flow = []
    for text in paragraphs:
        flow += [Paragraph(text, style), Spacer(1, 10)]
    document.build(flow)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate(n_normal: int, per_type: int, seed: int) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "pdf").mkdir(parents=True)
    (OUT / "ground_truth" / "json").mkdir(parents=True)

    counter = [10000]  # fresh invoice numbers, far from the INV-0001..0200 range

    def fresh_scanned():
        counter[0] += 1
        invoice, country = make_invoice(counter[0], rng)
        pdf = render_pdf(invoice, country, rng)
        return invoice, make_scanned(pdf, rng)

    def save(doc_id: str, pdf_bytes: bytes, kind: str, invoice=None, note: str = ""):
        (OUT / "pdf" / f"{doc_id}.pdf").write_bytes(pdf_bytes)
        has_truth = invoice is not None
        if has_truth:
            truth = asdict(invoice) if not isinstance(invoice, dict) else invoice
            truth["invoice_id"] = doc_id
            (OUT / "ground_truth" / "json" / f"{doc_id}.json").write_text(
                json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append({"doc_id": doc_id, "kind": kind, "has_truth": has_truth, "note": note})

    rows = []
    for i in range(1, n_normal + 1):
        invoice, scanned = fresh_scanned()
        save(f"NORM-{i:02d}", scanned, "normal", invoice)

    for kind in ANOMALY_TYPES:
        for i in range(1, per_type + 1):
            doc_id = f"ANOM-{kind}-{i}"
            if kind == "duplicate":
                source = DUPLICATE_SOURCES[i - 1]
                truth = json.loads(Path(f"data/synthetic/ground_truth/json/{source}.json").read_text(encoding="utf-8"))
                save(doc_id, Path(f"data/synthetic/scanned/{source}.pdf").read_bytes(), kind, truth,
                     f"copy of {source}")
            elif kind == "not_invoice":
                scanned = make_scanned(make_letter(rng), rng)
                save(doc_id, scanned, kind)
            elif kind == "merged":
                invoice, scanned = fresh_scanned()
                _, other = fresh_scanned()
                save(doc_id, make_anomaly(kind, scanned, rng, np_rng, other), kind, note="two invoices")
            else:
                invoice, scanned = fresh_scanned()
                save(doc_id, make_anomaly(kind, scanned, rng, np_rng), kind,
                     invoice if kind in CONTENT_INTACT else None)

    pd.DataFrame(rows).to_csv(OUT / "expected.csv", index=False)
    print(f"wrote {len(rows)} documents ({n_normal} normal, {len(rows) - n_normal} unusual) -> {OUT}")


# ---------------------------------------------------------------------------
# Report: was each unusual document caught?
# ---------------------------------------------------------------------------

def fully_correct(truth: dict, predicted: dict | None) -> bool:
    """Same standard as evaluate.py: every header field and every line item right (fuzzy)."""
    if not predicted:
        return False
    for field, kind in INVOICE_FIELDS.items():
        if not compare(kind, truth.get(field), predicted.get(field))[1]:
            return False
    items, predicted_items = truth["line_items"], predicted.get("line_items") or []
    if len(items) != len(predicted_items):
        return False
    return all(compare(kind, item[field], p.get(field))[1]
               for item, p in zip(items, predicted_items) for field, kind in LINE_ITEM_FIELDS.items())


def report(run: str) -> None:
    expected = pd.read_csv(OUT / "expected.csv")
    rows = []
    for row in expected.itertuples():
        record = json.loads((config.RUNS_DIR / run / "json" / f"{row.doc_id}.json").read_text(encoding="utf-8"))
        review_flags = [f["flag_code"] for f in record.get("flags", []) if f["severity"] == "review"]
        correct = None
        if row.has_truth:
            truth = json.loads((OUT / "ground_truth" / "json" / f"{row.doc_id}.json").read_text(encoding="utf-8"))
            correct = fully_correct(truth, record.get("data"))
        rows.append({"doc_id": row.doc_id, "kind": row.kind, "flagged": bool(review_flags),
                     "correct": correct, "handled": bool(review_flags) or bool(correct),
                     "flags": ", ".join(review_flags)})
    results = pd.DataFrame(rows)
    results.to_csv(config.RUNS_DIR / run / "anomaly_report.csv", index=False)

    summary = (results.groupby("kind", sort=False)
               .agg(n=("doc_id", "size"), flagged=("flagged", "mean"), correct=("correct", "mean"),
                    handled=("handled", "mean"))
               .assign(flagged=lambda t: (t.flagged * 100).round(0), handled=lambda t: (t.handled * 100).round(0),
                       correct=lambda t: (t.correct.astype(float) * 100).round(0)))
    print(summary.to_string())
    anomalies = results[results.kind != "normal"]
    normals = results[results.kind == "normal"]
    print(f"\nUnusual documents handled (flagged or fully correct): {anomalies.handled.mean():.0%} "
          f"({anomalies.handled.sum()} of {len(anomalies)})")
    print(f"Normal documents flagged (false-flag rate): {normals.flagged.mean():.0%} "
          f"({normals.flagged.sum()} of {len(normals)})")
    missed = anomalies[~anomalies.handled]
    if len(missed):
        print("\nMissed:")
        print(missed[["doc_id", "correct", "flags"]].to_string(index=False))
    print(f"\nDetails: runs/{run}/anomaly_report.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--normal", type=int, default=20)
    gen.add_argument("--per-type", type=int, default=3)
    gen.add_argument("--seed", type=int, default=2026)
    rep = sub.add_parser("report")
    rep.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.normal, args.per_type, args.seed)
    else:
        report(args.run)


if __name__ == "__main__":
    main()
