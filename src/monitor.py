"""
monitor.py - decide which documents need a person, without knowing the right answers.

Three kinds of signal, all label-free:

  1. Rules        Things that must be true of any correct extraction: required fields
                  present, total > 0, due date not before issue date, known currency,
                  arithmetic checks passed, not a duplicate of an earlier document.
  2. Guardrails   Fixed limits on input quality: blank page, very low resolution,
                  heavy tilt, too many pages.
  3. Outliers     Values far from what is normal for this kind of document, measured
                  against a baseline built from runs we validated with ground truth.
                  Uses a robust z-score: (value - median) / spread, where the spread is
                  based on the median absolute deviation, so a few odd documents in the
                  baseline don't distort it.

Each signal becomes a flag {flag_code, severity, detail}. Severity "review" sends the
document to a person; "info" is recorded for monitoring only.

At batch level, drift() compares a whole run with the baseline using the Population
Stability Index (PSI): how much the distribution of a metric has shifted.

Command line:
    python src/monitor.py build-baseline --runs digital_qwen7b_v2 scanned_vision_qwen7bvl_datefix ...
    python src/monitor.py drift --run <run name>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date

import numpy as np
import sqlalchemy as sa

import config
import db

REVIEW, INFO = "review", "info"

# Fixed guardrails (absolute limits that hold for any document)
MIN_INK_RATIO = 0.001       # below this the page is essentially blank
MIN_EFFECTIVE_DPI = 90      # scans below this are too coarse to read reliably
MAX_SKEW_DEGREES = 5.0      # our scans tilt by at most ~2 degrees
MAX_PAGES = 3
MIN_SHARPNESS = 30          # real documents measured >= 110; badly blurred or faded test pages < 5
EARLIEST_DATE, LATEST_DATE = date(2000, 1, 1), date(2035, 12, 31)

# Metrics checked for outliers: name -> (scale, severity).
#   scale "log": compared on a log scale because the metric varies by orders of magnitude
#                (a total of 50 vs 50,000 is normal; its spread is multiplicative).
#   severity: tested on the SROIE receipts, documents with unusual *input statistics*
#             (brightness, contrast, ink, sharpness) were no more often wrong than others
#             (8% vs 10%), and processing time only weakly (14%). So those are "info":
#             recorded and used for drift, but not sent to a person. Extreme inputs are
#             caught by the fixed guardrails above instead. Output outliers stay "review".
OUTLIER_METRICS = {
    "ink_ratio": ("log", INFO),
    "brightness": ("plain", INFO),
    "contrast": ("plain", INFO),
    "sharpness": ("log", INFO),
    "llm_seconds": ("log", INFO),
    "total": ("log", REVIEW),
    "item_count": ("log", REVIEW),
}

REQUIRED_FIELDS = {
    "invoice": ["invoice_number", "invoice_date", "vendor_name", "customer_name", "currency", "total"],
    "receipt": ["company", "date", "total"],
}


# ---------------------------------------------------------------------------
# Per-document metrics (inputs and outputs) used for outliers and drift
# ---------------------------------------------------------------------------

def document_metrics(record: dict) -> dict:
    """Numbers describing one processed document, for comparison with the baseline."""
    quality = record.get("quality") or {}
    data = record.get("data") or {}
    metrics = {name: quality.get(name) for name in ("ink_ratio", "brightness", "contrast", "sharpness")}
    metrics["total"] = data.get("total")
    metrics["item_count"] = len(data["line_items"]) if data.get("line_items") is not None else None
    metrics["llm_seconds"] = record.get("seconds")
    return metrics


def transform(value, kind: str):
    """Put a metric on the scale used for comparisons (log1p for wide-ranging metrics)."""
    if value is None:
        return None
    value = float(value)
    if kind == "log":
        return math.log1p(max(value, 0.0))
    return value


# ---------------------------------------------------------------------------
# Flags for one document
# ---------------------------------------------------------------------------

def flag(code: str, severity: str, detail: str = "") -> dict:
    return {"flag_code": code, "severity": severity, "detail": detail[:1000]}


def document_flags(schema: str, doc_id: str, record: dict, engine=None, baseline: dict | None = None,
                   run_id: str | None = None) -> list[dict]:
    """All flags for one processed document. engine/baseline are optional: without a
    database, duplicate checks and outlier scores are skipped (rules still apply).
    run_id lets a re-processed document be compared only with documents that came in
    before it, so the original of a duplicate pair is never flagged."""
    flags = []
    quality = record.get("quality") or {}
    data = record.get("data")

    # --- Did extraction work at all? ---
    if record.get("status") != "ok" or not data:
        flags.append(flag("extraction_failed", REVIEW, f"status {record.get('status')}: {record.get('error', '')}"))

    # --- Input guardrails ---
    if quality.get("ink_ratio") is not None and quality["ink_ratio"] < MIN_INK_RATIO:
        flags.append(flag("blank_page", REVIEW, f"ink ratio {quality['ink_ratio']}"))
    if quality.get("effective_dpi") is not None and quality["effective_dpi"] < MIN_EFFECTIVE_DPI:
        flags.append(flag("low_resolution", REVIEW, f"{quality['effective_dpi']} dpi"))
    if quality.get("skew_degrees") is not None and abs(quality["skew_degrees"]) > MAX_SKEW_DEGREES:
        flags.append(flag("heavy_skew", REVIEW, f"{quality['skew_degrees']} degrees"))
    if quality.get("pages", 1) > MAX_PAGES:
        flags.append(flag("too_many_pages", REVIEW, f"{quality['pages']} pages"))
    if quality.get("sharpness") is not None and quality["sharpness"] < MIN_SHARPNESS:
        flags.append(flag("blurred_or_faded", REVIEW, f"sharpness {quality['sharpness']}"))

    # --- Cross-check against an independent OCR reading of the page (image mode) ---
    if data and record.get("check_text"):
        flags += cross_check_flags(data, record["check_text"])

    if data:
        # --- Required fields present ---
        for field in REQUIRED_FIELDS[schema]:
            value = data.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                flags.append(flag(f"missing:{field}", REVIEW))

        # --- Values make sense ---
        total = data.get("total")
        if total is not None and total <= 0:
            flags.append(flag("nonpositive_total", REVIEW, f"total {total}"))

        dates = {field: data.get(field) for field in ("invoice_date", "due_date", "date") if data.get(field)}
        parsed = {}
        for field, value in dates.items():
            try:
                parsed[field] = date.fromisoformat(value)
            except (TypeError, ValueError):
                flags.append(flag(f"bad_date:{field}", REVIEW, str(value)))
                continue
            if not EARLIEST_DATE <= parsed[field] <= LATEST_DATE:
                flags.append(flag(f"implausible_date:{field}", REVIEW, value))
        if "invoice_date" in parsed and "due_date" in parsed and parsed["due_date"] < parsed["invoice_date"]:
            flags.append(flag("due_before_issue", REVIEW, f"{dates['invoice_date']} -> {dates['due_date']}"))

        if schema == "invoice":
            if data.get("currency") and data["currency"].upper() not in config.ALLOWED_CURRENCIES:
                flags.append(flag("unknown_currency", REVIEW, data["currency"]))
            if not data.get("line_items"):
                flags.append(flag("no_line_items", REVIEW))

    # --- Results of the arithmetic checks and fixes done in extract.py ---
    # final_check_failures = problems left in the answer that was kept. Records written
    # before that field existed only have needs_review, which then meant the same thing.
    failures = record.get("check_failures") or []
    final = record.get("final_check_failures")
    if final is None and record.get("needs_review"):
        final = failures[-1] if failures else ["arithmetic check failed"]
    if final:
        flags.append(flag("check_failed", REVIEW, "; ".join(final)))
    elif record.get("attempts", 1) > 1:
        flags.append(flag("retried", INFO, "; ".join(failures[0]) if failures else ""))
    if record.get("date_fixes"):
        flags.append(flag("date_converted", INFO, json.dumps(record["date_fixes"])))

    # --- Duplicates (needs the database) ---
    if engine is not None:
        cutoff = db.first_processed_at(engine, run_id, doc_id) if run_id else None
        if record.get("file_sha256"):
            same_file = db.other_documents_with_hash(engine, record["file_sha256"], doc_id, cutoff)
            if same_file:
                flags.append(flag("duplicate_file", REVIEW, "same file as " + ", ".join(same_file[:5])))
        if data and schema == "invoice":
            same_number = db.other_invoices_with_number(engine, data.get("vendor_name"),
                                                        data.get("invoice_number"), doc_id, cutoff)
            if same_number:
                flags.append(flag("duplicate_invoice", REVIEW,
                                  f"{data.get('vendor_name')} #{data.get('invoice_number')} also in "
                                  + ", ".join(same_number[:5])))

    # --- Statistical outliers against the baseline ---
    if baseline:
        for name, value in document_metrics(record).items():
            stats = baseline.get(name)
            kind, severity = OUTLIER_METRICS[name]
            value = transform(value, kind)
            if stats is None or value is None or not stats["scale"]:
                continue
            z = (value - stats["median"]) / stats["scale"]
            if abs(z) > config.OUTLIER_Z:
                flags.append(flag(f"outlier:{name}", severity, f"robust z = {z:+.1f}"))

    return flags


TITLE_WORDS = re.compile(r"\b(invoice|rechnung|receipt)\b", re.IGNORECASE)


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def cross_check_flags(data: dict, page_texts: list[str]) -> list[dict]:
    """Compare the model's answer with what a separate OCR engine read on the page.

    1. Grounding: the extracted total must actually be printed somewhere. A vision model
       can produce a perfectly consistent total that is not on the page, e.g. when the
       totals section is cut off it may add up the line items itself. Every check based
       only on the model's own answer passes in that case; an independent reader catches it.
       Compared as digits only (1,188.24 / 1.188,24 / $1188.24 -> 118824), token by token,
       so formatting and currency symbols don't matter.
    2. More than one document: a later page that starts with an invoice/receipt title
       suggests several documents were scanned into one file. Normal continuation pages
       start with the item table, not a title.
    """
    flags = []
    total = data.get("total")
    if total:
        wanted = digits(f"{total:.2f}")
        tokens = {digits(token) for page in page_texts for token in re.split(r"[\s|]+", page)}
        if not any(wanted in token for token in tokens if token):
            flags.append(flag("total_not_on_page", REVIEW, f"total {total} not found by OCR"))
    for number, page in enumerate(page_texts[1:], start=2):
        top_lines = page.splitlines()[:6]
        if any(TITLE_WORDS.search(line) for line in top_lines):
            flags.append(flag("multiple_documents", REVIEW, f"page {number} starts like a new document"))
            break
    return flags


# ---------------------------------------------------------------------------
# Baselines: what is "normal"
# ---------------------------------------------------------------------------

def robust_stats(values: list[float]) -> tuple[float, float]:
    """Median and a robust spread. 1.4826 x MAD equals the standard deviation for normal
    data, so z-scores read like ordinary ones; a small floor avoids dividing by ~0."""
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - median)))
    floor = 0.05 * abs(median) + 1e-6
    return median, max(scale, floor)


def metrics_from_db(engine, run_names: list[str]) -> list[dict]:
    """The same metrics as document_metrics(), read from the database for whole runs.

    Returns one dict per successfully processed document, with its run's schema and
    text_mode, so documents can be grouped by kind.
    """
    d, r = db.documents, db.runs
    total = sa.func.coalesce(db.invoice_headers.c.total, db.receipts.c.total)
    item_count = (sa.select(sa.func.count()).where(db.invoice_line_items.c.document_id == d.c.document_id)
                  .scalar_subquery())
    query = (sa.select(d.c.run_id, r.c.schema, r.c.text_mode, d.c.ink_ratio, d.c.brightness, d.c.contrast,
                       d.c.sharpness, d.c.llm_seconds, total.label("total"), item_count.label("item_count"))
             .select_from(d.join(r, d.c.run_id == r.c.run_id)
                          .outerjoin(db.invoice_headers, db.invoice_headers.c.document_id == d.c.document_id)
                          .outerjoin(db.receipts, db.receipts.c.document_id == d.c.document_id))
             .where(d.c.run_id.in_(run_names) & (d.c.status == "ok")))
    with engine.connect() as connection:
        rows = [dict(row._mapping) for row in connection.execute(query)]
    for row in rows:
        if row["schema"] != "invoice":
            row["item_count"] = None  # receipts have no line items
    return rows


def build_baseline(engine, run_names: list[str]) -> None:
    """Compute and store baseline statistics from runs whose results we have validated.

    Documents are grouped by (schema, text_mode), because a clean digital PDF and a
    scanned page naturally look very different.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in metrics_from_db(engine, run_names):
        groups.setdefault((row["schema"], row["text_mode"]), []).append(row)

    with engine.begin() as connection:
        for (schema, text_mode), rows in groups.items():
            connection.execute(db.baselines.delete().where(
                (db.baselines.c.schema == schema) & (db.baselines.c.text_mode == text_mode)))
            for name, (kind, _) in OUTLIER_METRICS.items():
                values = [transform(row[name], kind) for row in rows]
                values = [v for v in values if v is not None]
                if len(values) < 20:
                    continue  # too few to say what is normal
                median, scale = robust_stats(values)
                # Decile boundaries, de-duplicated: metrics with many tied values (item counts)
                # give repeated quantiles. Store each bin's real baseline share alongside, so
                # drift is measured against what the baseline actually looked like.
                edges = sorted(set(np.quantile(values, np.linspace(0, 1, 11))[1:-1].tolist()))
                expected = bin_shares(values, edges).tolist()
                connection.execute(db.baselines.insert().values(
                    schema=schema, text_mode=text_mode, metric=name, median=median, scale=scale,
                    count=len(values), bins_json=json.dumps({"edges": edges, "expected": expected}),
                    built_from=", ".join(run_names), built_at=db.now()))
            print(f"baseline {schema}/{text_mode}: {len(rows)} documents")


# ---------------------------------------------------------------------------
# Drift: has a whole batch shifted?
# ---------------------------------------------------------------------------

def bin_shares(values: list[float], edges: list[float]) -> np.ndarray:
    """Share of values falling in each bin defined by the edges (len(edges) + 1 bins)."""
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / counts.sum()


def psi(values: list[float], edges: list[float], expected: list[float]) -> float:
    """Population Stability Index of values against the baseline's bins.

    The bins are the baseline's deciles; `expected` is the share of baseline documents in
    each (about 10% each, unless values are tied). PSI adds up, over the bins,
    (actual share - expected share) x ln(actual / expected). Rules of thumb: < 0.1
    stable, 0.1-0.25 some shift, > 0.25 significant shift.
    """
    actual = np.clip(bin_shares(values, edges), 1e-4, None)
    expected = np.clip(np.asarray(expected), 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift(engine, run_name: str) -> list[dict]:
    """PSI of every baseline metric for one run. Returns rows: metric, psi, status, n."""
    documents = metrics_from_db(engine, [run_name])
    if not documents:
        return []
    baseline = db.load_baselines(engine, documents[0]["schema"], documents[0]["text_mode"])
    rows = []
    for name, (kind, _) in OUTLIER_METRICS.items():
        if name not in baseline or not baseline[name]["bins"]:
            continue
        values = [transform(row[name], kind) for row in documents]
        values = [v for v in values if v is not None]
        if len(values) < 10:
            continue
        bins = baseline[name]["bins"]
        if "expected" not in bins:
            continue  # baseline built by an older version; rebuild it
        value = psi(values, bins["edges"], bins["expected"])
        status = "drift" if value > config.DRIFT_PSI_ALERT else "watch" if value > 0.1 else "stable"
        rows.append({"metric": name, "psi": round(value, 3), "status": status, "n": len(values)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-baseline", help="compute 'normal' from validated runs")
    build.add_argument("--runs", nargs="+", required=True)
    check = sub.add_parser("drift", help="compare a run with the baseline")
    check.add_argument("--run", required=True)
    args = parser.parse_args()

    engine = db.get_engine()
    db.create_schema(engine)
    if args.command == "build-baseline":
        build_baseline(engine, args.runs)
    else:
        for row in drift(engine, args.run):
            print(f"{row['metric']:12} PSI {row['psi']:6.3f}  {row['status']}  (n={row['n']})")


if __name__ == "__main__":
    main()
