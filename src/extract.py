"""
extract.py - turn PDFs into tables using a local LLM served by Ollama.

Usage:
    python src/extract.py --input data/synthetic/digital --schema invoice --run digital_qwen7b_v2
    python src/extract.py --input data/sroie/pdf --schema receipt --run sroie_text --limit 20

For scanned (image-only) PDFs, add one of:
    --text-mode ocr      read the page images with OCR, then extract from that text
    --text-mode image    send the page images to a vision model (use with --model qwen2.5vl:7b)

For each PDF in the input folder:
    1. Read the PDF: its text layer (PyMuPDF), OCR text, or page images, per --text-mode.
    2. Send the text, our instructions and the table format (from schemas.py) to the model.
       The table format is also passed to Ollama as `format`, which forces the reply to be
       valid JSON with exactly those fields.
    3. Check the reply: Pydantic validates the types, then arithmetic checks verify that
       the numbers agree with each other. If a check fails, tell the model what is wrong
       and ask once more. Finally, any date not in ISO form is converted in code.
    4. Save one JSON file per document in runs/<run name>/json/.

At the end, all JSON files are combined into CSV tables in the run folder.

Running again with the same --run name skips documents that are already done, so an
interrupted run can simply be restarted. Use --overwrite to redo everything.

Nothing is trained here. The model's weights are fixed; every call is a prediction.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import requests
from pydantic import ValidationError

import config
import monitor
import quality
from schemas import SCHEMAS

log = logging.getLogger("ocr.extract")

# Ollama runs as a web server (this PC by default; set OLLAMA_URL to use another machine).
OLLAMA_URL = config.OLLAMA_URL


# ---------------------------------------------------------------------------
# Instructions for the model (the "system prompt")
#
# Prompt versions are kept side by side so earlier results can be reproduced
# with --prompt v1. Do not edit a version once it has been used for a run;
# add a new one instead.
# ---------------------------------------------------------------------------

PROMPT_V1 = """You extract data from business documents into JSON.
Rules:
- Use only information present in the document text. Never invent values.
- Dates: convert to ISO YYYY-MM-DD. Read the day/month order from the document's
  conventions (e.g. EUR/GBP documents usually use DD/MM/YYYY, USD documents MM/DD/YYYY).
- Numbers: plain JSON numbers, no currency symbols or thousands separators.
  Note that some documents use comma as the decimal separator (1.234,56 means 1234.56).
- Copy names and identifiers exactly as printed.
- If an optional field is absent, use null (or 0 for charges that are not present).
"""

# v2 = v1 plus one rule for each error pattern found in the v1 baseline's errors.csv:
#   vendor/customer swapped, addresses cut short, "Shipping crate" read as shipping,
#   "$" read as CAD; then a note explaining how the document is presented (see below).
PROMPT_V2_RULES = PROMPT_V1 + """- Vendor vs customer: the VENDOR is the party issuing the document, i.e. the name in the
  letterhead/header that is NOT under a label. The CUSTOMER is the party under a label such as
  "Bill To", "Billed To", "Customer", "Sold To" or "Invoice To". Either one may be a person's
  surname or a company; decide by position and label, never by which name looks like a company.
- Addresses: include every address line (street, town, region and postal code), joined with ", ".
- Shipping: only a separate charge in the totals section labelled e.g. Shipping, Delivery,
  Freight or Postage counts. A line item whose description mentions shipping is a normal line
  item, not the shipping charge. If the totals section has no such charge, shipping is 0.
- Currency: "$" alone means USD unless the document explicitly names another dollar currency
  (e.g. CAD, AUD). "£" is GBP, "€" is EUR.
"""

# The last rule of v2 tells the model how the document is presented. It depends on
# --text-mode, so there is one version per mode. ("lines" mode, used only with v1,
# has no note; with v2 it gets the blocks note, as in the original v2 runs.)
INPUT_NOTES = {
    # PDF text layer, grouped into blocks by PyMuPDF (the original v2 runs)
    "blocks": """- The text is given as blocks, one per line; " | " separates lines within a block
  (for item tables, it separates the cells of a row).
""",
    # Text read from a scanned image by OCR
    "ocr": """- The text was read from a scanned image by OCR. Each line of output is one line of the
  page; " | " separates pieces of text on the same line, such as table cells or two
  side-by-side columns (so a line may mix, e.g., the seller's and the buyer's address).
  OCR can drop spaces between words and misread characters; where a value is clearly
  garbled, use the surrounding text and the arithmetic of the totals to recover it.
""",
    # The page images themselves, read by a vision model
    "image": """- The document is given as one image per page. Read all text from the image(s).
""",
}

PROMPT_V2 = PROMPT_V2_RULES + INPUT_NOTES["blocks"]

# v3 = v2 plus rules for one document type, added after studying the SROIE errors.
# For invoices there are no extra rules yet, so v3 is identical to v2.
V3_SCHEMA_RULES = {
    "invoice": "",
    # SROIE labels the registered company, but the model often returned the brand or
    # outlet ("DOMINO'S PIZZA" instead of "DOMMAL FOOD SERVICES SDN BHD"), and sometimes
    # included the registration number ("KING'S CONFECTIONERY S/B 273500-U (KSB)").
    "receipt": """- Company: if the receipt prints a registered company name (usually ending in SDN BHD,
  SDN. BHD., S/B, BHD, BERHAD, ENTERPRISE, TRADING or PLT, often next to a registration
  number such as (123456-X)), use that registered name, not the brand, shop or outlet name.
  Otherwise use the shop name printed at the top. Do not include registration numbers or
  GST numbers in the name.
""",
}

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V2}  # v3's extra rules are added in system_prompt


def system_prompt(version: str, text_mode: str, schema: str = "invoice") -> str:
    """The system prompt for a prompt version, document type and --text-mode.

    v1 is fixed. v2 and v3 end with a note on how the document is presented
    (blocks, OCR text or images); v3 also adds the rules for the document type.
    """
    if version == "v1":
        return PROMPT_V1
    if version == "v2" and text_mode not in ("ocr", "image"):
        return PROMPT_V2  # exactly as used for the original v2 text runs
    note = INPUT_NOTES.get(text_mode, INPUT_NOTES["blocks"])
    schema_rules = V3_SCHEMA_RULES[schema] if version == "v3" else ""
    return PROMPT_V2_RULES + schema_rules + note


# ---------------------------------------------------------------------------
# Step 1: get the text out of the PDF
# ---------------------------------------------------------------------------

def pdf_text(path: Path, mode: str = "blocks") -> str:
    """Return all the text in a PDF as one string.

    mode="lines"  (used by v1)
        Lines sorted top-to-bottom across the whole page width. Simple, but when two
        things sit side by side (e.g. the seller's address next to the invoice details)
        their lines get interleaved, which scrambles addresses.

    mode="blocks" (default, used by v2)
        PyMuPDF groups text that belongs together into blocks (a paragraph, an address,
        one table row). Each block becomes one output line, with " | " between the lines
        inside it, e.g. "Bill To: | Acme Ltd | 1 High St | London". Addresses stay whole
        and table rows read like "Office chair | 1 | 426,01 € | 426,01 €".

    Scanned PDFs contain only an image, so this returns (almost) nothing for them.
    """
    page_texts = []
    with pymupdf.open(path) as document:
        for page in document:
            if mode == "lines":
                page_texts.append(page.get_text("text", sort=True))
            else:
                # Each block is a tuple: (x0, y0, x1, y1, text, block_number, block_type).
                # block_type 0 is text; 1 would be an image, which we skip.
                blocks = page.get_text("blocks", sort=True)
                block_lines = [" | ".join(block[4].strip().split("\n"))
                               for block in blocks if block[6] == 0]
                page_texts.append("\n".join(block_lines))

    return join_pages(page_texts)


def join_pages(page_texts: list[str]) -> str:
    """Combine per-page text; multi-page documents get a marker where each page starts."""
    if len(page_texts) == 1:
        return page_texts[0]
    return "\n".join(f"--- page {number} ---\n{text}"
                     for number, text in enumerate(page_texts, start=1))


# --- Scanned PDFs, option A: OCR ---------------------------------------------

OCR_DPI = 200       # resolution the page is rendered at before OCR
_ocr_engine = None  # created on first use, because loading the OCR models takes a moment


def ocr_text(path: Path) -> str:
    """Read a scanned PDF with OCR (RapidOCR, runs on the CPU) and return its text.

    OCR returns a list of text boxes: the four corner points of each box, the text
    inside it, and a confidence score. To give the model something readable, the boxes
    are put back into lines:
      1. Straighten: scans are often slightly tilted, so text on the right of a line
         sits a little higher or lower than text on the left. The typical tilt is
         measured from the boxes' top edges and undone.
      2. Group boxes whose vertical centres are close into one line.
      3. Sort each line left to right and join its pieces with " | ".
    """
    return join_pages(ocr_pages(path))


def ocr_pages(path: Path) -> list[str]:
    """OCR every page of a PDF; returns one text (lines joined by newlines) per page."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR  # imported here so text-only runs don't need it
        _ocr_engine = RapidOCR()

    page_texts = []
    with pymupdf.open(path) as document:
        for page in document:
            # Render the page to an image (height x width x colour channels).
            pixmap = page.get_pixmap(dpi=OCR_DPI)
            image = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            boxes, _ = _ocr_engine(image)  # boxes: [[corners, text, confidence], ...] or None
            page_texts.append(boxes_to_lines(boxes or []))
    return page_texts


def boxes_to_lines(boxes: list) -> str:
    """Arrange OCR boxes into lines of text (see ocr_text for the steps)."""
    if not boxes:
        return ""

    # 1. Measure the tilt: the angle of each box's top edge (top-left -> top-right corner).
    angles = []
    for corners, _, _ in boxes:
        (x0, y0), (x1, y1) = corners[0], corners[1]
        if x1 - x0 > 20:  # very short boxes give unreliable angles
            angles.append(np.arctan2(y1 - y0, x1 - x0))
    tilt = float(np.median(angles)) if angles else 0.0
    cos_t, sin_t = np.cos(-tilt), np.sin(-tilt)

    # Each box -> (left x, vertical centre, height, text), in straightened coordinates.
    pieces = []
    for corners, text, _ in boxes:
        corners = np.asarray(corners, dtype=float)
        x = corners[:, 0] * cos_t - corners[:, 1] * sin_t   # rotate every corner by -tilt
        y = corners[:, 0] * sin_t + corners[:, 1] * cos_t
        pieces.append((x.min(), (y.min() + y.max()) / 2, y.max() - y.min(), text))

    # 2. Group into lines: walk down the page; a piece joins the current line if its
    #    centre is within 60% of a typical text height from the line's centre.
    typical_height = float(np.median([height for _, _, height, _ in pieces]))
    pieces.sort(key=lambda piece: piece[1])
    lines, current = [], [pieces[0]]
    for piece in pieces[1:]:
        line_centre = np.mean([p[1] for p in current])
        if abs(piece[1] - line_centre) < 0.6 * typical_height:
            current.append(piece)
        else:
            lines.append(current)
            current = [piece]
    lines.append(current)

    # 3. Left to right within each line.
    return "\n".join(" | ".join(p[3] for p in sorted(line, key=lambda p: p[0])) for line in lines)


# --- Scanned PDFs, option B: give the page images to a vision model -------------

IMAGE_DPI = 150  # higher = sharper but more image tokens (Qwen2.5-VL: ~1 token per 28x28 pixels)


def page_images(path: Path) -> list[str]:
    """Render each page as a PNG image, base64-encoded (the form Ollama's API expects)."""
    images = []
    with pymupdf.open(path) as document:
        for page in document:
            png = page.get_pixmap(dpi=IMAGE_DPI).tobytes("png")
            images.append(base64.b64encode(png).decode("ascii"))
    return images


def read_document(path: Path, mode: str) -> tuple[str, list[str]]:
    """Get a document's content in the form --text-mode asks for.

    Returns (text, images): text for "lines"/"blocks"/"ocr", page images for "image".
    """
    if mode == "image":
        return "", page_images(path)
    if mode == "ocr":
        return ocr_text(path), []
    return pdf_text(path, mode), []


# ---------------------------------------------------------------------------
# Step 3 (defined before step 2 because step 2 uses them): fixing and checking the reply
# ---------------------------------------------------------------------------

# Which fields hold dates, per schema.
DATE_FIELDS = {"invoice": ["invoice_date", "due_date"], "receipt": ["date"]}

# Date formats with a month name, tried after the numeric ones.
MONTH_NAME_FORMATS = ["%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d-%b-%y",
                      "%d %b %y", "%d%b%Y", "%d/%b/%Y", "%b %d %Y", "%B %d %Y"]


def to_iso_date(value: str | None, day_first: bool) -> str | None:
    """Convert a date string to ISO YYYY-MM-DD. Returns the value unchanged if it can't be parsed.

    Models are told to output ISO dates, but sometimes copy the date as printed
    ("19/10/2018", "18-11-18", "Jul 28, 2025"). Converting those is plain string
    handling, so it is done here in code rather than by asking the model again.

    For an all-number date like 05/04/2024, the order of day and month can't be told
    from the string itself; `day_first` decides. If one part is over 12, that part
    must be the day, whatever `day_first` says.
    """
    if not value:
        return value
    text = value.strip()

    # Already ISO (or year-first with / or .): just tidy it.
    match = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        return _iso_or_none(year, month, day) or value

    # All-number dates: 19/10/2018, 18-11-18, 28.05.2024
    match = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})", text)
    if match:
        first, second, year = (int(part) for part in match.groups())
        if year < 100:
            year += 2000                     # two-digit year: 18 -> 2018
        if first > 12:
            day, month = first, second       # e.g. 28/05: must be day first
        elif second > 12:
            day, month = second, first       # e.g. 05/28: must be month first
        elif day_first:
            day, month = first, second
        else:
            month, day = first, second
        return _iso_or_none(year, month, day) or value

    # Eight digits with no separators: 20180304 (year first) or 25032018 (day first).
    if re.fullmatch(r"\d{8}", text):
        formats = ["%Y%m%d", "%d%m%Y"] if text[:2] in ("19", "20") else ["%d%m%Y", "%Y%m%d"]
    else:
        formats = MONTH_NAME_FORMATS
        text = re.sub(r"\s+", " ", text).title()  # "28 JUL 2025" -> "28 Jul 2025" for %b

    for date_format in formats:
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        iso = _iso_or_none(parsed.year, parsed.month, parsed.day)
        if iso:
            return iso
    return value  # could not parse: leave it for a person (it will show up in errors.csv)


def _iso_or_none(year: int, month: int, day: int) -> str | None:
    """Build an ISO date string, or None if the date is impossible or implausible."""
    if not 2000 <= year <= 2035:  # anything else is almost certainly a misread
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:            # e.g. 31 February
        return None


def normalize_dates(schema: str, data: dict) -> dict:
    """Convert every date field in an extracted record to ISO, in place.

    Returns the changes made, {field: [before, after]}, so they can be recorded.
    Day/month order for all-number dates: US invoices (USD) are month-first;
    other invoices and the Malaysian SROIE receipts are day-first.
    """
    day_first = not (schema == "invoice" and data.get("currency") == "USD")
    changes = {}
    for field in DATE_FIELDS.get(schema, []):
        before = data.get(field)
        after = to_iso_date(before, day_first)
        if after != before:
            data[field] = after
            changes[field] = [before, after]
    return changes

def consistency_checks(schema: str, invoice: dict) -> list[str]:
    """Return a list of arithmetic problems in an extracted invoice (empty list = all good).

    An invoice states the same facts more than once (line amounts, subtotal, total),
    so the numbers can be checked against each other without knowing the right answer.
    Each problem is written as a sentence because it is sent back to the model.

    Receipts have no such checks, so they always pass.
    """
    if schema != "invoice":
        return []

    tolerance = 0.011  # one cent, plus a little room for floating-point error
    problems = []
    items = invoice.get("line_items") or []

    # 1. Each line: quantity x unit price should equal the line amount.
    for item in items:
        line_total = item["quantity"] * item["unit_price"]
        if abs(line_total - item["amount"]) > tolerance:
            problems.append(f"line {item['line_no']}: quantity x unit_price = "
                            f"{line_total:.2f}, but amount is {item['amount']}")

    # 2. The line amounts should add up to the subtotal.
    items_sum = round(sum(item["amount"] for item in items), 2)
    if items and abs(items_sum - invoice["subtotal"]) > tolerance:
        problems.append(f"line item amounts add up to {items_sum}, but subtotal is {invoice['subtotal']}"
                        " (a line item may be missing or duplicated)")

    # 3. Subtotal x tax rate should equal the tax amount.
    expected_tax = invoice["subtotal"] * invoice["tax_rate"]
    if abs(expected_tax - invoice["tax_amount"]) > tolerance:
        problems.append(f"subtotal x tax_rate = {expected_tax:.2f}, but tax_amount is {invoice['tax_amount']}")

    # 4. Subtotal + tax + shipping should equal the total.
    expected_total = round(invoice["subtotal"] + invoice["tax_amount"] + invoice["shipping"], 2)
    if abs(expected_total - invoice["total"]) > tolerance:
        problems.append(f"subtotal + tax_amount + shipping = {expected_total}, but total is {invoice['total']}"
                        " (check whether shipping is really a separate charge in the totals section)")

    return problems


# ---------------------------------------------------------------------------
# Step 2: ask the model
# ---------------------------------------------------------------------------

def call_llm(model: str, json_schema: dict, messages: list[dict], context_size: int) -> tuple[str, float]:
    """Send one chat request to Ollama. Returns (the model's reply text, seconds taken)."""
    response = requests.post(OLLAMA_URL, timeout=600, json={
        "model": model,
        "messages": messages,
        "format": json_schema,  # constrained decoding: the reply must match this JSON Schema
        "stream": False,        # wait for the whole reply instead of receiving it word by word
        "options": {
            "temperature": 0,          # always pick the most likely token -> same input, same output
            "num_ctx": context_size,   # how many tokens (prompt + reply) the model can handle
        },
    })
    response.raise_for_status()  # turn HTTP errors (e.g. unknown model) into an exception
    body = response.json()
    seconds = body.get("total_duration", 0) / 1e9  # Ollama reports nanoseconds
    return body["message"]["content"], seconds


def extract_document(text: str, images: list[str], schema_name: str, args) -> dict:
    """Extract one document's fields, retrying if the arithmetic checks fail.

    Pass either the document's text, or its page images (for a vision model).
    Returns the fields to store in the document's output record.
    May raise ValidationError (reply did not fit the schema) or a network error;
    the caller records those as failures.
    """
    schema_class = SCHEMAS[schema_name]
    json_schema = schema_class.model_json_schema()

    # The conversation sent to the model. The system message holds the general rules;
    # the user message holds the table format and the document itself.
    if images:
        # Vision model: the page images travel alongside the message.
        document_message = {"role": "user", "images": images, "content": (
            "Extract the fields defined by this JSON schema from the attached document image(s).\n\n"
            f"JSON schema:\n{json.dumps(json_schema)}")}
    else:
        document_message = {"role": "user", "content": (
            "Extract the fields defined by this JSON schema from the document below.\n\n"
            f"JSON schema:\n{json.dumps(json_schema)}\n\n"
            f"Document text:\n<<<\n{text}\n>>>")}
    messages = [
        {"role": "system", "content": system_prompt(args.prompt, args.text_mode, schema_name)},
        document_message,
    ]

    total_seconds = 0.0
    problems_per_attempt = []  # the check failures of each attempt, kept for the record
    best_data, best_problems = None, None

    for attempt in range(args.retries + 1):
        reply, seconds = call_llm(args.model, json_schema, messages, args.num_ctx)
        total_seconds += seconds

        # Parse the JSON and validate it (types converted, missing fields rejected).
        data = schema_class.model_validate(json.loads(reply)).model_dump()
        problems = consistency_checks(schema_name, data)
        problems_per_attempt.append(problems)

        # Keep whichever attempt fails the fewest checks. A retry can make things worse:
        # small models sometimes "fix" the arithmetic by inventing numbers. On a tie the
        # earlier attempt is kept.
        if best_data is None or len(problems) < len(best_problems):
            best_data, best_problems = data, problems

        # Stop if everything checks out, or if we have used up our retries.
        if not problems or attempt == args.retries:
            break

        # Otherwise continue the conversation: show the model its own answer and what is
        # wrong with it, and ask for a corrected version.
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": (
                "Your JSON fails these consistency checks:\n- " + "\n- ".join(problems)
                + "\nRe-read the document carefully and return the corrected JSON. Only change values that the"
                  " document supports; if the document itself is inconsistent, keep what is printed.")},
        ]

    # Convert any non-ISO dates in the chosen answer (see to_iso_date).
    date_fixes = normalize_dates(schema_name, best_data) if args.fix_dates else {}

    return {
        "status": "ok",
        "data": best_data,
        "date_fixes": date_fixes,  # {field: [as the model wrote it, after conversion]}
        # Still failing a check after all attempts -> a person should look at it.
        # (main() later widens needs_review to include the monitoring flags.)
        "needs_review": bool(best_problems),
        "final_check_failures": best_problems,
        "attempts": len(problems_per_attempt),
        "check_failures": problems_per_attempt,
        "seconds": round(total_seconds, 2),
    }


# ---------------------------------------------------------------------------
# Step 4: combine the per-document JSON files into CSV tables
# ---------------------------------------------------------------------------

def write_tables(schema: str, run_dir: Path) -> None:
    """Write <schema>s.csv (one row per document) and, for invoices, line_items.csv."""
    header_rows, line_item_rows = [], []

    for path in sorted((run_dir / "json").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        data = record.get("data") or {}            # failed documents have no data
        items = data.pop("line_items", None)       # move line items to their own table
        header_rows.append({"doc_id": path.stem, "status": record["status"], **data})
        for item in items or []:
            line_item_rows.append({"doc_id": path.stem, **item})

    pd.DataFrame(header_rows).to_csv(run_dir / f"{schema}s.csv", index=False, encoding="utf-8")
    if line_item_rows:
        pd.DataFrame(line_item_rows).to_csv(run_dir / "line_items.csv", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line settings. worker.py passes argv itself to reuse the same defaults."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="folder of PDFs")
    parser.add_argument("--schema", choices=SCHEMAS, required=True, help="which table format to extract")
    parser.add_argument("--model", default="qwen2.5:7b", help="any model installed in Ollama (see `ollama list`)")
    parser.add_argument("--run", help="run name, i.e. the output folder under runs/ "
                                      "(default: <input folder>_<model>)")
    parser.add_argument("--limit", type=int, help="only process the first N PDFs")
    parser.add_argument("--num-ctx", type=int, default=8192, help="model context window in tokens")
    parser.add_argument("--overwrite", action="store_true", help="redo documents already extracted")
    parser.add_argument("--prompt", choices=PROMPTS, default="v2", help="system prompt version")
    parser.add_argument("--text-mode", choices=["lines", "blocks", "ocr", "image"], default="blocks",
                        help="how the document is given to the model: lines/blocks = the PDF's text layer "
                             "(see pdf_text); ocr = text read from the page image by OCR (for scans); "
                             "image = the page images themselves (needs a vision model, e.g. qwen2.5vl:7b)")
    parser.add_argument("--retries", type=int, default=1,
                        help="extra attempts when arithmetic checks fail (0 = no retries)")
    parser.add_argument("--no-date-fix", dest="fix_dates", action="store_false",
                        help="keep dates exactly as the model wrote them (to reproduce runs made before "
                             "date conversion was added)")
    parser.add_argument("--no-cross-check", dest="cross_check", action="store_false",
                        help="in image mode, skip reading the page with OCR as an independent check "
                             "(saves ~2 s per document; see monitor.cross_check_flags)")
    parser.add_argument("--no-db", dest="use_db", action="store_false",
                        help="don't write to the database even if OCR_DB_URL is set")
    return parser.parse_args(argv)


def model_digest(model: str) -> str | None:
    """Ollama's content hash for the installed model, so a run records exactly which
    weights it used. (A tag like qwen2.5:7b can point to new weights after `ollama pull`.)"""
    tags_url = OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        models = requests.get(tags_url, timeout=10).json().get("models", [])
    except (requests.RequestException, ValueError):
        return None
    for entry in models:
        if entry.get("name") == model or entry.get("model") == model:
            return entry.get("digest")
    return None


def file_sha256(path: Path) -> str:
    """Fingerprint of the file's bytes: identical files give identical fingerprints."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace, pdfs: list[Path] | None = None) -> dict:
    """Process PDFs with the given settings. Used by the command line and by worker.py.

    pdfs: the files to process (default: every PDF in args.input, up to args.limit).
    Returns {"run_dir", "processed", "errors", "records": {doc_id: record}}.
    """
    # Set up the run folder and record the settings, so every result can be traced
    # back to the exact model, prompt and options that produced it.
    run_name = args.run or f"{args.input.name}_{args.model.replace(':', '-').replace('/', '-')}"
    run_dir = config.RUNS_DIR / run_name
    (run_dir / "json").mkdir(parents=True, exist_ok=True)
    settings = {name: str(value) for name, value in vars(args).items()}
    settings["model_digest"] = model_digest(args.model)
    settings["ollama_url"] = OLLAMA_URL
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))

    if pdfs is None:
        pdfs = sorted(args.input.glob("*.pdf"))[: args.limit]  # [:None] means "all"
    log.info(f"{len(pdfs)} PDFs -> {run_dir}  (model {args.model}, prompt {args.prompt}, "
             f"text {args.text_mode}, retries {args.retries})")

    # Database (optional): results are written there as well as to JSON, and it holds the
    # baseline used for outlier checks and earlier documents used for duplicate checks.
    engine, baseline = None, None
    if args.use_db and config.DB_URL:
        import db
        engine = db.get_engine()
        db.create_schema(engine)
        db.save_run(engine, run_name, settings)
        baseline = db.load_baselines(engine, args.schema, args.text_mode)
        log.info(f"writing to database; baseline metrics available: {len(baseline)}")

    start_time = time.time()
    processed = 0
    errors = 0  # documents that failed with an exception (network, model, etc.)
    records = {}

    for number, pdf in enumerate(pdfs, start=1):
        output_path = run_dir / "json" / f"{pdf.stem}.json"
        if output_path.exists() and not args.overwrite:
            continue  # already done in an earlier run

        read_start = time.time()
        text, images = read_document(pdf, args.text_mode)
        record = {"source": str(pdf), "model": args.model, "text_mode": args.text_mode, "chars": len(text),
                  "read_seconds": round(time.time() - read_start, 2),  # PDF reading / OCR / rendering time
                  "file_sha256": file_sha256(pdf),
                  "quality": quality.measure(pdf)}  # input-quality measurements for monitoring
        if args.text_mode == "ocr":
            record["ocr_text"] = text  # kept so OCR mistakes can be told apart from model mistakes

        if not images and len(text.strip()) < 20:
            # No text found: an image-only (scanned) PDF read in a text mode. Use --text-mode ocr or image.
            record.update(status="no_text_layer", data=None)
        else:
            # Record failures and keep going, so one bad document does not stop a long run.
            try:
                record.update(extract_document(text, images, args.schema, args))
            except ValidationError as error:   # reply did not fit the table format
                record.update(status="invalid", data=None, error=str(error))
            except Exception as error:         # network problem, bad JSON, etc.
                record.update(status="error", data=None, error=repr(error))

        # Independent second reader: in image mode the model reads the picture, so OCR the
        # page too and keep its text, to check the model's answer against (see monitor.py).
        if args.text_mode == "image" and args.cross_check and record.get("data"):
            check_start = time.time()
            record["check_text"] = ocr_pages(pdf)
            record["check_seconds"] = round(time.time() - check_start, 2)

        # Monitoring: every reason this document might be wrong (see monitor.py).
        # Any flag with severity "review" sends the document to a person.
        record["flags"] = monitor.document_flags(args.schema, pdf.stem, record, engine, baseline, run_name)
        record["needs_review"] = any(f["severity"] == monitor.REVIEW for f in record["flags"])

        output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if engine is not None:
            db.save_document(engine, run_name, pdf.stem, args.schema, record)
        processed += 1
        errors += record["status"] == "error"
        records[pdf.stem] = record

        # Progress line, with the reasons if the document was flagged for review.
        review_codes = [f["flag_code"] for f in record["flags"] if f["severity"] == monitor.REVIEW]
        review_note = f"  REVIEW: {', '.join(review_codes)}" if review_codes else ""
        log.info(f"[{number}/{len(pdfs)}] {pdf.name}: {record['status']}  {record.get('seconds', '')}s{review_note}")

    write_tables(args.schema, run_dir)

    if processed:
        elapsed = time.time() - start_time
        log.info(f"processed {processed} docs in {elapsed:.0f}s ({elapsed / processed:.1f}s/doc)")
    log.info(f"tables written to {run_dir}")
    return {"run_dir": run_dir, "processed": processed, "errors": errors, "records": records}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    result = run(args)
    # If every document failed with an error, something systemic is wrong (Ollama down,
    # model missing, network). Exit with a non-zero code so a scheduler notices.
    if result["processed"] and result["errors"] == result["processed"]:
        raise SystemExit(f"all {result['processed']} documents failed; "
                         f"first check that Ollama is reachable at {OLLAMA_URL}")


if __name__ == "__main__":
    main()
