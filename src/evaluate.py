"""
evaluate.py - score an extraction run against the correct answers.

Usage:
    python src/evaluate.py --run runs/digital_qwen7b_v2 --schema invoice
    python src/evaluate.py --run runs/sroie_text --schema receipt

What it does:
    1. Loads the model's answers   (runs/<name>/json/*.json, written by extract.py)
    2. Loads the correct answers   (data/.../ground_truth/, written by the data scripts)
    3. Compares them field by field and prints accuracy tables
    4. Writes runs/<name>/errors.csv listing every mismatch side by side.
       That file is the one to read to understand *why* the model gets things wrong.

How a field counts as right depends on its type:
    text     "exact": equal after lower-casing and removing punctuation and spaces
             "fuzzy": at least 90% character similarity after the same clean-up
    number   within 0.01 of the true value (tax rate: within 0.0001)
    exact    dates, currency codes, invoice numbers: the strings must match

A document the model failed on (no usable output) counts as wrong on every field.

This script never calls the model; it only compares files, so it runs in a second.
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Which comparison rule applies to which field
# ---------------------------------------------------------------------------

TEXT = "text"    # names, addresses, descriptions: allow formatting differences
NUMBER = "num"   # money and quantities: allow a 0.01 rounding difference
RATE = "rate"    # tax rate: a fraction like 0.07, so needs a tighter tolerance
EXACT = "exact"  # dates, codes, identifiers: must match exactly

INVOICE_FIELDS = {
    "invoice_number": EXACT,
    "invoice_date": EXACT,
    "due_date": EXACT,
    "vendor_name": TEXT,
    "vendor_address": TEXT,
    "customer_name": TEXT,
    "customer_address": TEXT,
    "currency": EXACT,
    "subtotal": NUMBER,
    "tax_rate": RATE,
    "tax_amount": NUMBER,
    "shipping": NUMBER,
    "total": NUMBER,
}

LINE_ITEM_FIELDS = {
    "description": TEXT,
    "quantity": NUMBER,
    "unit_price": NUMBER,
    "amount": NUMBER,
}

RECEIPT_FIELDS = {
    "company": TEXT,
    "address": TEXT,
    "date": EXACT,
    "total": NUMBER,
}

FUZZY_THRESHOLD = 0.9      # text similarity needed for a "fuzzy" match
NUMBER_TOLERANCE = 0.011   # a cent, plus a little room for floating-point error
RATE_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Comparing one value
# ---------------------------------------------------------------------------

def normalize_text(value) -> str:
    """Lower-case and keep only letters and digits.

    "Smith, Luna & Ward" -> "smithlunaward". This makes the comparison ignore
    punctuation, spacing and capitalisation, which are not real extraction errors.
    """
    if value is None:
        return ""
    return re.sub(r"[\W_]+", "", str(value).casefold())


def compare(kind: str, expected, got) -> tuple[bool, bool]:
    """Compare one predicted value with the correct one.

    Returns (exact_match, fuzzy_match). For numbers and exact fields both values
    are the same; only text fields have a separate, more lenient fuzzy result.
    """
    if kind == TEXT:
        expected_clean = normalize_text(expected)
        got_clean = normalize_text(got)
        exact = expected_clean == got_clean
        similarity = SequenceMatcher(None, expected_clean, got_clean).ratio()  # 0.0 to 1.0
        return exact, exact or similarity >= FUZZY_THRESHOLD

    if kind in (NUMBER, RATE):
        tolerance = RATE_TOLERANCE if kind == RATE else NUMBER_TOLERANCE
        try:
            match = abs(float(expected) - float(got)) <= tolerance
        except (TypeError, ValueError):  # missing value or not a number
            match = False
        return match, match

    # EXACT: compare as strings, ignoring case and surrounding spaces.
    # A missing value (None) is treated as an empty string.
    match = str(expected or "").strip().upper() == str(got or "").strip().upper()
    return match, match


# ---------------------------------------------------------------------------
# Loading the two sides of the comparison
# ---------------------------------------------------------------------------

def load_predictions(run_dir: Path) -> tuple[dict, dict]:
    """Read the model's output for every document in a run.

    Returns two dictionaries keyed by document id:
      predictions[doc_id]  -> the extracted fields, or None if extraction failed
      needs_review[doc_id] -> True if the document was flagged for a human
    """
    predictions, needs_review = {}, {}
    for path in (run_dir / "json").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        succeeded = record["status"] == "ok"
        predictions[path.stem] = record["data"] if succeeded else None
        # A failed extraction always needs a human. Older runs have no
        # needs_review field, so default to False for them.
        needs_review[path.stem] = record.get("needs_review", False) or not succeeded
    return predictions, needs_review


def load_truth(schema: str, truth_dir: Path | None = None) -> dict[str, dict]:
    """Read the correct answers, keyed by document id (the PDF's file name).

    truth_dir: a folder with one JSON of answers per document (e.g. golden/receipt/truth).
    Without it, the development data folders are used.
    """
    if truth_dir is not None:
        return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in truth_dir.glob("*.json")}
    if schema == "invoice":
        # One JSON file per synthetic invoice, including its line items.
        truth_dir = Path("data/synthetic/ground_truth/json")
        return {path.stem: json.loads(path.read_text(encoding="utf-8"))
                for path in truth_dir.glob("*.json")}

    # Receipts: one CSV row per receipt. Read everything as text so values
    # like "9.00" are kept as printed; an empty cell stays "" (not NaN).
    table = pd.read_csv("data/sroie/ground_truth/receipts.csv", dtype=str, keep_default_na=False)
    return {
        row.receipt_id: {
            "company": row.company,
            "address": row.address,
            "date": row.date_iso,   # the model is asked for ISO dates, so compare with date_iso
            "total": row.total,
        }
        for row in table.itertuples()
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run folder, e.g. runs/digital_qwen7b_v2")
    parser.add_argument("--schema", choices=["invoice", "receipt"], required=True)
    parser.add_argument("--db", action="store_true",
                        help="also store the field-by-field results in the database (table evaluations)")
    parser.add_argument("--subset", type=Path,
                        help="only score documents that also have a PDF in this folder, e.g. "
                             "data/synthetic/scanned, to compare runs on exactly the same documents")
    args = parser.parse_args()

    predictions, needs_review = load_predictions(args.run)
    truth = load_truth(args.schema)
    header_fields = INVOICE_FIELDS if args.schema == "invoice" else RECEIPT_FIELDS

    # Only score the documents this run actually processed (a run may use --limit).
    doc_ids = sorted(predictions)
    if args.subset:
        subset_ids = {path.stem for path in args.subset.glob("*.pdf")}
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in subset_ids]
    if not doc_ids:
        raise SystemExit(f"no predictions in {args.run}")

    field_results = []  # one row per field comparison -> per-field accuracy
    mismatches = []     # one row per wrong field      -> errors.csv
    doc_results = []    # one row per document          -> document-level accuracy

    for doc_id in doc_ids:
        expected = truth[doc_id]
        predicted = predictions[doc_id] or {}  # failed extraction: every field is missing

        # --- Header fields (one value per document) ---
        header_all_right = True
        for field, kind in header_fields.items():
            # One SROIE receipt has no total in its official label; skip it rather
            # than count the model wrong for something we cannot check.
            if args.schema == "receipt" and field == "total" and expected["total"] == "":
                continue

            exact, fuzzy = compare(kind, expected.get(field), predicted.get(field))
            field_results.append({"doc_id": doc_id, "field": field, "exact": exact, "fuzzy": fuzzy})
            header_all_right = header_all_right and fuzzy
            if not exact:
                mismatches.append({"doc_id": doc_id, "field": field, "expected": expected.get(field),
                                   "got": predicted.get(field), "fuzzy_ok": fuzzy})

        # --- Line items (invoices only) ---
        # Items are matched by position: the model's 1st item against the true 1st item,
        # and so on. If the model skips an item, every later item is compared against
        # the wrong one, so a single omission can show up as many errors.
        items_all_right = None
        if args.schema == "invoice":
            true_items = expected["line_items"]
            predicted_items = predicted.get("line_items") or []

            items_all_right = len(true_items) == len(predicted_items)
            if not items_all_right:
                mismatches.append({"doc_id": doc_id, "field": "line_items.count", "expected": len(true_items),
                                   "got": len(predicted_items), "fuzzy_ok": False})

            for position, true_item in enumerate(true_items):
                # A missing predicted item is an empty dict, so all its fields count as wrong.
                predicted_item = predicted_items[position] if position < len(predicted_items) else {}
                for field, kind in LINE_ITEM_FIELDS.items():
                    exact, fuzzy = compare(kind, true_item[field], predicted_item.get(field))
                    field_results.append({"doc_id": doc_id, "field": f"item.{field}", "exact": exact,
                                          "fuzzy": fuzzy})
                    items_all_right = items_all_right and fuzzy
                    if not exact:
                        mismatches.append({"doc_id": doc_id, "field": f"line {position + 1}.{field}",
                                           "expected": true_item[field], "got": predicted_item.get(field),
                                           "fuzzy_ok": fuzzy})

        doc_results.append({
            "doc_id": doc_id,
            "failed": predictions[doc_id] is None,
            "header_ok": header_all_right,
            "items_ok": items_all_right,
            "needs_review": needs_review[doc_id],
        })

    # --- Summarise ---
    # Accuracy per field = share of comparisons that were right, as a percentage.
    # sort=False keeps the fields in the order they were defined above.
    field_accuracy = (pd.DataFrame(field_results)
                      .groupby("field", sort=False)[["exact", "fuzzy"]]
                      .mean().mul(100).round(1))
    docs = pd.DataFrame(doc_results)
    # A subset gets its own file, e.g. errors_scanned.csv, so the full errors.csv is kept.
    errors_path = args.run / (f"errors_{args.subset.name}.csv" if args.subset else "errors.csv")
    pd.DataFrame(mismatches).to_csv(errors_path, index=False, encoding="utf-8")

    if args.db:
        # Store every comparison (run, document, field, right/wrong) for the dashboard.
        import db
        engine = db.get_engine()
        db.create_schema(engine)
        db.save_evaluations(engine, args.run.name, [
            {"doc_id": row["doc_id"], "field": row["field"], "exact": bool(row["exact"]),
             "fuzzy": bool(row["fuzzy"])} for row in field_results])

    # --- Report ---
    print(f"\nRun: {args.run}   documents: {len(docs)}   failed/no output: {int(docs.failed.sum())}\n")
    print("Field accuracy (%)")
    print(field_accuracy.to_string())
    print(f"\nDocuments with every header field right (fuzzy): {docs.header_ok.mean() * 100:.1f}%")

    if args.schema == "invoice":
        docs["all_ok"] = docs.header_ok & docs["items_ok"].astype(bool)
        print(f"Documents with every line item right (fuzzy):    {docs['items_ok'].mean() * 100:.1f}%")
        print(f"Documents fully right:                            {docs.all_ok.mean() * 100:.1f}%")

        # The number that matters if the pipeline runs unattended: of the documents
        # that were NOT sent for review, how many are actually fully right?
        auto_accepted = docs[~docs.needs_review]
        print(f"\nFlagged for human review (failed a check):        {docs.needs_review.sum()} of {len(docs)}")
        if len(auto_accepted):
            print(f"Auto-accepted documents that are fully right:     {auto_accepted.all_ok.mean() * 100:.1f}%"
                  f"  ({auto_accepted.all_ok.sum()} of {len(auto_accepted)})")

    print(f"\nMismatches written to {errors_path}")


if __name__ == "__main__":
    main()
