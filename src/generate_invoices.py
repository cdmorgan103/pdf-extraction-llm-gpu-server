"""
generate_invoices.py - create fake invoice PDFs whose correct answers we know exactly.

Usage:
    python src/generate_invoices.py                      # 200 invoices, 50 scanned, seed 42
    python src/generate_invoices.py --n 500 --seed 7     # a different, larger set

Output (under --out, default data/synthetic):
    digital/INV-0001.pdf ...        PDFs with a text layer (text can be read directly)
    scanned/INV-0002.pdf ...        the same invoices as images that look scanned
                                    (no text layer, so they need OCR or a vision model)
    ground_truth/invoices.csv       correct answers, one row per invoice
    ground_truth/line_items.csv     correct answers, one row per line item
    ground_truth/json/INV-0001.json correct answers per invoice (used by evaluate.py)

Why synthetic data: because we create each invoice ourselves, the correct answer
for every field is known exactly, which is what makes accurate scoring possible.

To keep extraction from being trivial, invoices vary in:
    - layout (4 templates) and font (3)
    - wording of labels ("Invoice No." / "Inv #" / "Bill Number" ...)
    - country conventions: US, UK or German dates, and 1,234.56 vs 1.234,56 numbers
    - an optional shipping charge, and 1 to 30 line items (long ones span two pages)

The correct answers are always stored in one standard form: ISO dates (YYYY-MM-DD),
plain numbers, and ISO currency codes (USD, GBP, EUR).

Reproducibility: everything random comes from one generator seeded with --seed, so the
same seed always produces the same invoices. Note for anyone editing this file: adding,
removing or reordering a random call changes every invoice that follows it.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from faker import Faker
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# ---------------------------------------------------------------------------
# What an invoice contains (this is also the shape of the correct answers)
# ---------------------------------------------------------------------------

@dataclass
class LineItem:
    line_no: int
    description: str
    quantity: float
    unit_price: float
    amount: float


@dataclass
class Invoice:
    invoice_id: str          # file name, e.g. "INV-0001" (not printed on the invoice)
    invoice_number: str      # the number printed on the invoice
    invoice_date: str        # ISO yyyy-mm-dd
    due_date: str            # ISO yyyy-mm-dd
    vendor_name: str
    vendor_address: str
    customer_name: str
    customer_address: str
    currency: str            # ISO 4217 code
    subtotal: float
    tax_rate: float          # a fraction, e.g. 0.2 for 20%
    tax_amount: float
    shipping: float          # 0.0 when there is no shipping charge
    total: float
    line_items: list[LineItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The choices invoices are randomly made from
# ---------------------------------------------------------------------------

# Country conventions. Each entry is:
#   (Faker locale for names/addresses, currency code, currency symbol,
#    possible date formats, decimal separator, thousands separator, page size)
LOCALES = {
    "US": ("en_US", "USD", "$", ["%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"], ".", ",", LETTER),
    "GB": ("en_GB", "GBP", "£", ["%d/%m/%Y", "%d %B %Y", "%d %b %Y"], ".", ",", A4),
    "DE": ("de_DE", "EUR", "€", ["%d.%m.%Y", "%Y-%m-%d"], ",", ".", A4),
}

# Realistic tax rates per country (US 0% = no sales tax; DE 7% = reduced VAT).
TAX_RATES = {"US": [0.0, 0.0625, 0.07, 0.0825], "GB": [0.2], "DE": [0.19, 0.07]}

# Products: (description, minimum unit price, maximum unit price).
# "Shipping crate" is a deliberate trap: it is a line item, not a shipping charge.
PRODUCTS = [
    ("Consulting services", 80, 250), ("Software license (annual)", 200, 2500),
    ("Cloud hosting - monthly", 40, 900), ("Laptop stand", 15, 80),
    ("Wireless keyboard", 20, 120), ("Office chair", 90, 600), ("Printer toner", 30, 150),
    ("Network switch 24-port", 150, 900), ("Technical support hours", 60, 180),
    ("Training workshop", 300, 3000), ("USB-C docking station", 80, 300),
    ("Website maintenance", 100, 1200), ("Data migration", 500, 5000),
    ("Copy paper (box)", 20, 60), ("Monitor 27in", 180, 700), ("Shipping crate", 25, 90),
    ("Security audit", 1000, 8000), ("Ergonomic mouse", 15, 90), ("Desk lamp", 12, 70),
    ("Project management", 70, 200),
]

# Different wordings for the same label. One is picked at random per invoice.
# (The order of keys matters for reproducibility: one random pick per key, in order.)
LABELS = {
    "invoice_number": ["Invoice No.", "Invoice #", "Inv #", "Bill Number", "Invoice Number", "Reference"],
    "invoice_date": ["Invoice Date", "Date", "Issued", "Date of Issue", "Billing Date"],
    "due_date": ["Due Date", "Payment Due", "Pay By", "Due"],
    "bill_to": ["Bill To", "Billed To", "Customer", "Sold To", "Invoice To"],
    "subtotal": ["Subtotal", "Sub-total", "Net Amount", "Total before tax"],
    "tax": ["Tax", "VAT", "Sales Tax", "GST"],
    "shipping": ["Shipping", "Delivery", "Freight", "Postage & Handling"],
    "total": ["Total", "Amount Due", "Grand Total", "Balance Due", "Total Due"],
    "columns": [  # headings of the item table: description, quantity, unit price, amount
        ("Description", "Qty", "Unit Price", "Amount"),
        ("Item", "Quantity", "Rate", "Total"),
        ("Product / Service", "Units", "Price", "Line Total"),
        ("Details", "Hrs/Qty", "Unit Cost", "Ext. Price"),
    ],
}

# (regular font, bold font) pairs built into every PDF reader.
FONTS = [
    ("Helvetica", "Helvetica-Bold"),
    ("Times-Roman", "Times-Bold"),
    ("Courier", "Courier-Bold"),
]

ACCENT_COLOURS = ["#1f4e79", "#2e7d32", "#6a1b9a", "#b71c1c", "#37474f"]


# ---------------------------------------------------------------------------
# Step 1: invent the invoice's data
# ---------------------------------------------------------------------------

def format_money(amount: float, decimal_sep: str, thousands_sep: str, symbol: str, symbol_after: bool) -> str:
    """Format a number the way a given country prints money.

    format_money(1234.5, ".", ",", "$", False) -> "$1,234.50"
    format_money(1234.5, ",", ".", "€", True)  -> "1.234,50 €"
    """
    text = f"{amount:,.2f}"  # always starts as US style: "1,234.50"
    # Swap the separators via a temporary placeholder ("\0"), so that replacing ","
    # and "." does not mix them up: "1,234.50" -> "1\0234.50" -> "1\0234,50" -> "1.234,50"
    text = text.replace(",", "\0").replace(".", decimal_sep).replace("\0", thousands_sep)
    return f"{text} {symbol}" if symbol_after else f"{symbol}{text}"


def make_invoice(index: int, rng: random.Random) -> tuple[Invoice, str]:
    """Invent one invoice. Returns the invoice and its country code ("US", "GB" or "DE")."""
    country = rng.choice(list(LOCALES))
    faker_locale, currency, *_ = LOCALES[country]

    # Faker makes realistic names and addresses for the chosen country. It gets its
    # own seed from our generator, so it is reproducible too.
    fake = Faker(faker_locale)
    fake.seed_instance(rng.randint(0, 10**9))

    # Dates: issued some time in 2024-2025, due 14 to 60 days later.
    invoice_date = date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))
    due_date = invoice_date + timedelta(days=rng.choice([14, 15, 30, 45, 60]))

    # Line items. Mostly short invoices, but now and then 12 or 30 items (two pages).
    item_count = rng.choice([1, 2, 2, 3, 3, 4, 5, 6, 8, 12, 30])
    items = []
    for line_no, (description, min_price, max_price) in enumerate(rng.sample(PRODUCTS * 2, item_count), start=1):
        if "hours" in description.lower():
            quantity = rng.choice([1.5, 4, 7.5, 12, 20])          # hours can be fractional
        else:
            quantity = rng.choice([1, 1, 1, 2, 3, 5, 10, 12, 25])
        unit_price = round(rng.uniform(min_price, max_price), 2)
        items.append(LineItem(line_no, description, float(quantity), unit_price, round(quantity * unit_price, 2)))

    # Totals are calculated, so they are always arithmetically consistent.
    subtotal = round(sum(item.amount for item in items), 2)
    tax_rate = rng.choice(TAX_RATES[country])
    tax_amount = round(subtotal * tax_rate, 2)
    has_shipping = rng.random() < 0.35  # about a third of invoices charge shipping
    shipping = round(rng.uniform(5, 120), 2) if has_shipping else 0.0
    total = round(subtotal + tax_amount + shipping, 2)

    # The printed invoice number comes in a few different styles.
    number_style = rng.choice(["INV-{y}-{n:05d}", "{n:06d}", "A{n:04d}/{y}", "{y}{n:04d}"])
    invoice_number = number_style.format(y=invoice_date.year, n=rng.randint(1, 99999))

    # The parties. Most customers are companies; 1 in 5 is a private person.
    vendor_name = fake.company()
    vendor_address = fake.address().replace("\n", ", ")
    customer_is_company = rng.random() < 0.8
    customer_name = fake.company() if customer_is_company else fake.name()
    customer_address = fake.address().replace("\n", ", ")

    invoice = Invoice(
        invoice_id=f"INV-{index:04d}",
        invoice_number=invoice_number,
        invoice_date=invoice_date.isoformat(),
        due_date=due_date.isoformat(),
        vendor_name=vendor_name,
        vendor_address=vendor_address,
        customer_name=customer_name,
        customer_address=customer_address,
        currency=currency,
        subtotal=subtotal,
        tax_rate=tax_rate,
        tax_amount=tax_amount,
        shipping=shipping,
        total=total,
        line_items=items,
    )
    return invoice, country


# ---------------------------------------------------------------------------
# Step 2: draw the invoice as a PDF
# ---------------------------------------------------------------------------

def render_pdf(invoice: Invoice, country: str, rng: random.Random) -> bytes:
    """Lay the invoice out as a PDF with a randomly chosen look. Returns the PDF's bytes.

    The PDF is built with reportlab's "platypus" system: we make a list of elements
    ("flowables": paragraphs, tables, spacers) and reportlab places them on pages,
    starting a new page when one is full.
    """
    _, _, symbol, date_formats, decimal_sep, thousands_sep, page_size = LOCALES[country]

    # --- Random choices for this invoice's appearance ---
    font, bold_font = rng.choice(FONTS)
    template = rng.choice(["classic", "banner", "minimal", "boxed"])
    # German invoices usually put the symbol after the number (12,50 €).
    # (Only German invoices make this random pick.)
    symbol_after = country == "DE" and rng.random() < 0.7
    date_format = rng.choice(date_formats)
    accent = rng.choice([colors.HexColor(c) for c in ACCENT_COLOURS])
    labels = {key: rng.choice(options) for key, options in LABELS.items()}

    # --- Small helpers that print values in this invoice's style ---
    def money(amount):
        return format_money(amount, decimal_sep, thousands_sep, symbol, symbol_after)

    def printed_date(iso_date):
        return date.fromisoformat(iso_date).strftime(date_format)

    def printed_quantity(quantity):
        return f"{quantity:g}".replace(".", decimal_sep)  # 7.5 -> "7,5" in Germany; 2.0 -> "2"

    def address_lines(address):
        return "<br/>".join(part.strip() for part in address.split(","))  # one part per line

    # --- Text styles ---
    body = ParagraphStyle("body", fontName=font, fontSize=9.5, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=10.5, textColor=colors.HexColor("#444444"))
    title_style = ParagraphStyle("title", fontName=bold_font, fontSize=rng.choice([20, 24, 28]), leading=30,
                                 textColor=accent if template != "minimal" else colors.black)
    vendor_style = ParagraphStyle("vendor", fontName=bold_font, fontSize=13, leading=16)
    bold_body = ParagraphStyle("bold_body", parent=body, fontName=bold_font)

    # --- Building blocks shared by the templates ---
    vendor_block = [Paragraph(invoice.vendor_name, vendor_style),
                    Paragraph(address_lines(invoice.vendor_address), small)]

    title_text = rng.choice(["INVOICE", "Invoice", "TAX INVOICE",
                             "Rechnung / Invoice" if country == "DE" else "INVOICE"])
    title = Paragraph(title_text, title_style)

    # Invoice number and dates, as label/value pairs.
    details = [
        [labels["invoice_number"], invoice.invoice_number],
        [labels["invoice_date"], printed_date(invoice.invoice_date)],
        [labels["due_date"], printed_date(invoice.due_date)],
    ]
    details_table = Table([[Paragraph(f"<b>{label}</b>", body), Paragraph(value, body)] for label, value in details],
                          colWidths=[32 * mm, 38 * mm])
    details_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                                       ("TOPPADDING", (0, 0), (-1, -1), 1)]))

    customer_block = [Paragraph(f"<b>{labels['bill_to']}:</b>", body),
                      Paragraph(invoice.customer_name, bold_body),
                      Paragraph(address_lines(invoice.customer_address), body)]

    # --- Top of the page: arranged differently by each template ---
    elements = []
    if template == "classic":
        # Vendor on the left; title and details on the right; customer below.
        top = Table([[vendor_block, [title, Spacer(1, 4), details_table]]], colWidths=[95 * mm, 80 * mm])
        top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        elements += [top, Spacer(1, 10 * mm), *customer_block]

    elif template == "banner":
        # Coloured band with vendor name and "INVOICE"; address and customer left, details right.
        banner = Table([[Paragraph(f'<font color="white">{invoice.vendor_name}</font>', vendor_style),
                         Paragraph('<font color="white">INVOICE</font>',
                                   ParagraphStyle("banner_title", parent=title_style, alignment=2))]],
                       colWidths=[110 * mm, 65 * mm])
        banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), accent),
                                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        middle = Table([[[Paragraph(address_lines(invoice.vendor_address), small), Spacer(1, 6), *customer_block],
                         details_table]],
                       colWidths=[105 * mm, 70 * mm])
        middle.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        elements += [banner, Spacer(1, 6 * mm), middle]

    elif template == "minimal":
        # Everything in one column, details as plain "Label: value" lines.
        elements += [title, Spacer(1, 3), *vendor_block, Spacer(1, 6 * mm)]
        for label, value in details:
            elements.append(Paragraph(f"{label}: {value}", body))
        elements += [Spacer(1, 5 * mm), *customer_block]

    else:  # "boxed"
        # Title and details first; vendor and customer side by side in two boxes.
        boxes = Table([[vendor_block, customer_block]], colWidths=[87 * mm, 88 * mm])
        boxes.setStyle(TableStyle([("BOX", (0, 0), (0, 0), 0.8, colors.grey),
                                   ("BOX", (1, 0), (1, 0), 0.8, colors.grey),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
        elements += [title, Spacer(1, 4), details_table, Spacer(1, 6 * mm), boxes]

    elements.append(Spacer(1, 8 * mm))

    # --- Item table ---
    rows = [list(labels["columns"])]  # heading row
    for item in invoice.line_items:
        rows.append([Paragraph(item.description, body), printed_quantity(item.quantity),
                     money(item.unit_price), money(item.amount)])
    item_table = Table(rows, colWidths=[85 * mm, 20 * mm, 35 * mm, 35 * mm],
                       repeatRows=1)  # repeat the heading row if the table continues on page 2

    table_style = [
        ("FONTNAME", (0, 0), (-1, 0), bold_font),   # heading row bold
        ("FONTNAME", (0, 1), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),        # numbers right-aligned
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if template in ("classic", "boxed"):    # full grid, grey heading
        table_style += [("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee"))]
    elif template == "banner":              # coloured heading, striped rows
        table_style += [("BACKGROUND", (0, 0), (-1, 0), accent),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f3f3")])]
    else:                                   # minimal: just a rule under the heading and at the end
        table_style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black)]
    item_table.setStyle(TableStyle(table_style))
    elements += [item_table, Spacer(1, 4 * mm)]

    # --- Totals (right-aligned) ---
    tax_label = f"{labels['tax']} ({invoice.tax_rate * 100:g}%)".replace(".", decimal_sep)  # e.g. "VAT (20%)"
    total_rows = [[labels["subtotal"], money(invoice.subtotal)],
                  [tax_label, money(invoice.tax_amount)]]
    if invoice.shipping:
        total_rows.append([labels["shipping"], money(invoice.shipping)])
    total_rows.append([labels["total"], money(invoice.total)])

    totals_table = Table(total_rows, colWidths=[40 * mm, 35 * mm], hAlign="RIGHT")
    totals_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTNAME", (0, -1), (-1, -1), bold_font),            # final total in bold
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),   # rule above the final total
    ]))
    elements += [totals_table, Spacer(1, 10 * mm)]

    # --- Optional footer ---
    payment_days = (date.fromisoformat(invoice.due_date) - date.fromisoformat(invoice.invoice_date)).days
    footer = rng.choice([
        "Thank you for your business!",
        f"Payment terms: net {payment_days} days.",
        "Please include the invoice number with your payment.",
        "",  # no footer
    ])
    if footer:
        elements.append(Paragraph(footer, small))

    # --- Build the PDF in memory ---
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=page_size,
                                 leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                                 title=f"Invoice {invoice.invoice_number}")
    document.build(elements)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Step 3: make a "scanned" copy
# ---------------------------------------------------------------------------

def make_scanned(pdf_bytes: bytes, rng: random.Random) -> bytes:
    """Turn a digital PDF into an image-only PDF that looks like a cheap scan.

    Each page is converted to a grey-scale image, then degraded the way real scans are:
    slightly rotated, blurred, uneven contrast, sensor noise, dust specks and JPEG
    compression. The result has no text layer, so text extraction returns nothing.
    """
    source = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    output = pymupdf.open()
    noise_rng = np.random.default_rng(rng.randint(0, 10**9))  # numpy generator for per-pixel noise
    dpi = rng.choice([110, 150, 200])                          # scan resolution

    for page in source:
        # Render the page to a grey-scale image.
        pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)

        # Paper fed in slightly crooked, and a slightly out-of-focus scanner.
        image = image.rotate(rng.uniform(-2.0, 2.0), resample=Image.BICUBIC, expand=True, fillcolor=255)
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))

        # Work on the pixels as numbers (0 = black, 255 = white).
        pixels = np.asarray(image, dtype=np.float32)
        pixels = pixels * rng.uniform(0.85, 1.0) + rng.uniform(0, 25)        # lower contrast, greyish paper
        pixels += noise_rng.normal(0, rng.uniform(4, 14), pixels.shape)      # sensor noise
        dust = noise_rng.random(pixels.shape) < rng.uniform(0.0005, 0.003)   # a few random dark specks
        pixels[dust] = 30
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

        # Save as a low-quality JPEG (compression artefacts) and place it on a new page
        # of the same size as the original.
        jpeg = io.BytesIO()
        image.save(jpeg, format="JPEG", quality=rng.randint(45, 80))
        new_page = output.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=jpeg.getvalue())

    return output.tobytes(deflate=True)


# ---------------------------------------------------------------------------
# Main: generate everything and write the files
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="number of digital invoices")
    parser.add_argument("--n-scanned", type=int, default=50, help="how many of them also get a scanned copy")
    parser.add_argument("--seed", type=int, default=42, help="same seed = same invoices")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    rng = random.Random(args.seed)  # the single source of all randomness

    digital_dir = args.out / "digital"
    scanned_dir = args.out / "scanned"
    truth_dir = args.out / "ground_truth"
    truth_json_dir = truth_dir / "json"
    for folder in (digital_dir, scanned_dir, truth_json_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # Decide up front which invoices also get a scanned copy.
    scanned_numbers = set(rng.sample(range(1, args.n + 1), min(args.n_scanned, args.n)))

    invoice_rows, line_item_rows = [], []
    for number in range(1, args.n + 1):
        invoice, country = make_invoice(number, rng)

        pdf = render_pdf(invoice, country, rng)
        (digital_dir / f"{invoice.invoice_id}.pdf").write_bytes(pdf)
        if number in scanned_numbers:
            (scanned_dir / f"{invoice.invoice_id}.pdf").write_bytes(make_scanned(pdf, rng))

        # Correct answers: the full invoice as JSON, plus rows for the two CSV tables.
        record = asdict(invoice)
        (truth_json_dir / f"{invoice.invoice_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        items = record.pop("line_items")
        invoice_rows.append(record)
        line_item_rows += [{"invoice_id": invoice.invoice_id, **item} for item in items]

    pd.DataFrame(invoice_rows).to_csv(truth_dir / "invoices.csv", index=False, encoding="utf-8")
    pd.DataFrame(line_item_rows).to_csv(truth_dir / "line_items.csv", index=False, encoding="utf-8")
    print(f"wrote {args.n} digital + {len(scanned_numbers)} scanned invoices, "
          f"{len(line_item_rows)} line items -> {args.out}")


if __name__ == "__main__":
    main()
