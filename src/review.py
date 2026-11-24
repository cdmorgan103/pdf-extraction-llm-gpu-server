"""
review.py - the human review loop: how live accuracy gets measured without ground truth.

    python src/review.py export --run <run>            -> reviews/queue_<run>_<date>.csv
    (a person opens the CSV in Excel, checks each value against the PDF, fixes wrong ones)
    python src/review.py import reviews/queue_....csv  -> reviews + evaluations tables

Which documents go in the queue:
    flagged  every document monitoring flagged for review
    audit    a random sample (default 5%) of documents that were NOT flagged

The audit sample is what makes the accuracy estimate honest. Flagged documents are
chosen *because* they look wrong, so their error rate says nothing about the rest. A
random sample of unflagged documents estimates how many errors slip through unnoticed.

How to fill in the CSV: one row per field. `corrected_value` starts as a copy of the
extracted value. Change it where the value is wrong, and put your name in `reviewer`
on every row you have checked (rows without a reviewer are ignored on import).

For testing without a person:
    python src/review.py simulate <csv> --truth-dir data/anomalies/ground_truth/json
fills the CSV from known correct answers, as a perfect reviewer would.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

import config
import db
from evaluate import INVOICE_FIELDS, LINE_ITEM_FIELDS, RECEIPT_FIELDS, compare

QUEUE_DIR = config.REVIEWS_DIR


def field_kind(field: str) -> str:
    """The comparison rule for a field name like 'total' or 'line 3.amount'."""
    if field.startswith("line "):
        return LINE_ITEM_FIELDS[field.split(".", 1)[1]]
    return {**INVOICE_FIELDS, **RECEIPT_FIELDS}[field]


def as_text(value) -> str:
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def recent_live_runs(engine, days: int) -> list[str]:
    """The worker's daily runs (live_<type>_<mode>_<date>) from the last `days` days."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with engine.connect() as connection:
        names = [row.run_id for row in connection.execute(sa.select(db.runs.c.run_id))]
    return sorted(n for n in names if n.startswith("live_") and n[-10:] >= cutoff)


def export(engine, run_ids: list[str], audit_rate: float, seed: int) -> Path:
    d = db.documents
    with engine.connect() as connection:
        docs = pd.read_sql(sa.select(d.c.document_id, d.c.doc_id, d.c.schema, d.c.needs_review,
                                     d.c.source_path).where(d.c.run_id.in_(run_ids)), connection)
        flags = pd.read_sql(sa.select(db.document_flags.c.document_id, db.document_flags.c.flag_code)
                            .where(db.document_flags.c.document_id.in_(docs.document_id.tolist())
                                   & (db.document_flags.c.severity == "review")), connection)
        headers = pd.read_sql(sa.select(db.invoice_headers), connection).set_index("document_id")
        receipts = pd.read_sql(sa.select(db.receipts), connection).set_index("document_id")
        items = pd.read_sql(sa.select(db.invoice_line_items), connection)
    if docs.empty:
        raise SystemExit(f"no documents for runs {run_ids}")
    # Documents already reviewed are not sent again.
    with engine.connect() as connection:
        reviewed = {row.document_id for row in connection.execute(sa.select(db.reviews.c.document_id).distinct())}
    docs = docs[~docs.document_id.isin(reviewed)]

    # Which documents: all flagged + a seeded random sample of the rest.
    flagged = docs[docs.needs_review]
    unflagged = docs[~docs.needs_review].sort_values("doc_id")
    audit_count = max(1, round(len(unflagged) * audit_rate)) if len(unflagged) else 0
    audit = unflagged.loc[random.Random(seed).sample(list(unflagged.index), audit_count)]
    flag_text = flags.groupby("document_id").flag_code.apply(", ".join)

    rows = []
    for sample_type, selection in (("flagged", flagged), ("audit", audit)):
        for doc in selection.itertuples():
            fields = {}
            if doc.schema == "invoice" and doc.document_id in headers.index:
                fields = headers.loc[doc.document_id].to_dict()
                for item in items[items.document_id == doc.document_id].sort_values("line_no").itertuples():
                    for name in LINE_ITEM_FIELDS:
                        fields[f"line {item.line_no}.{name}"] = getattr(item, name)
            elif doc.schema == "receipt" and doc.document_id in receipts.index:
                fields = receipts.loc[doc.document_id].to_dict()
            if not fields:  # extraction failed: still list the main fields so values can be entered
                fields = {name: None for name in (INVOICE_FIELDS if doc.schema == "invoice" else RECEIPT_FIELDS)}
            for field, value in fields.items():
                rows.append({
                    "document_id": doc.document_id, "doc_id": doc.doc_id, "sample_type": sample_type,
                    "flags": flag_text.get(doc.document_id, ""), "source_path": doc.source_path,
                    "field": field, "extracted_value": as_text(value), "corrected_value": as_text(value),
                    "reviewer": "",
                })

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    label = run_ids[0] if len(run_ids) == 1 else f"{len(run_ids)}runs"
    path = QUEUE_DIR / f"queue_{label}_{date.today().isoformat()}.csv"
    # utf-8-sig so Excel shows characters like ü and £ correctly
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(flagged)} flagged + {len(audit)} audit documents ({len(rows)} fields) -> {path}")
    return path


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_queue(engine, path: Path) -> None:
    queue = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    done = queue[queue.reviewer.str.strip() != ""]
    if done.empty:
        raise SystemExit("no reviewed rows (fill in the reviewer column)")

    stamp = db.now()
    review_rows, evaluation_rows = [], []
    with engine.connect() as connection:
        doc_info = pd.read_sql(sa.select(db.documents.c.document_id, db.documents.c.run_id, db.documents.c.doc_id)
                               .where(db.documents.c.document_id.in_([int(x) for x in done.document_id.unique()])),
                               connection).set_index("document_id")
    for row in done.itertuples():
        document_id = int(row.document_id)
        review_rows.append({"document_id": document_id, "field": row.field, "extracted_value": row.extracted_value,
                            "corrected_value": row.corrected_value, "sample_type": row.sample_type,
                            "reviewer": row.reviewer, "reviewed_at": stamp})
        expected = row.corrected_value or None
        exact, fuzzy = compare(field_kind(row.field), expected, row.extracted_value or None)
        info = doc_info.loc[document_id]
        evaluation_rows.append({"run_id": info.run_id, "doc_id": info.doc_id, "field": row.field,
                                "exact": exact, "fuzzy": fuzzy, "source": "review",
                                "sample_type": row.sample_type, "evaluated_at": stamp})

    reviewed_ids = sorted({r["document_id"] for r in review_rows})
    with engine.begin() as connection:
        # Re-importing the same documents replaces their earlier review.
        connection.execute(db.reviews.delete().where(db.reviews.c.document_id.in_(reviewed_ids)))
        for run_id, doc_ids in pd.DataFrame(evaluation_rows).groupby("run_id").doc_id:
            connection.execute(db.evaluations.delete().where(
                (db.evaluations.c.source == "review") & (db.evaluations.c.run_id == run_id)
                & db.evaluations.c.doc_id.in_(list(set(doc_ids)))))
        connection.execute(db.reviews.insert(), review_rows)
        connection.execute(db.evaluations.insert(), evaluation_rows)

    corrected = sum(r["corrected_value"] != r["extracted_value"] for r in review_rows)
    print(f"imported {len(review_rows)} reviewed fields from {len(reviewed_ids)} documents; "
          f"{corrected} values corrected")


# ---------------------------------------------------------------------------
# Simulate a perfect reviewer (for testing the loop)
# ---------------------------------------------------------------------------

def simulate(path: Path, truth_dirs: list[Path]) -> None:
    """Fill a queue from known answers. Documents without known answers are left unreviewed."""
    queue = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    truth_cache = {}

    def truth_for(doc_id):
        if doc_id not in truth_cache:
            truth_cache[doc_id] = None
            for folder in truth_dirs:
                if (folder / f"{doc_id}.json").exists():
                    truth_cache[doc_id] = json.loads((folder / f"{doc_id}.json").read_text(encoding="utf-8"))
        return truth_cache[doc_id]

    for index, row in queue.iterrows():
        truth = truth_for(row.doc_id)
        if truth is None:
            continue
        if row.field.startswith("line "):
            number, name = row.field[5:].split(".", 1)
            items = truth["line_items"]
            value = items[int(number) - 1][name] if int(number) <= len(items) else ""
        else:
            value = truth.get(row.field)
        queue.at[index, "corrected_value"] = as_text(value)
        queue.at[index, "reviewer"] = "simulated"
    queue.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"simulated review of {queue.reviewer.ne('').sum()} fields in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    which = ex.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", nargs="+", help="one or more run names")
    which.add_argument("--recent-live", type=int, metavar="DAYS",
                       help="all of the worker's daily runs from the last N days (e.g. 7 for a weekly review)")
    ex.add_argument("--audit-rate", type=float, default=config.AUDIT_SAMPLE_RATE)
    ex.add_argument("--seed", type=int, default=0)
    im = sub.add_parser("import")
    im.add_argument("csv", type=Path)
    sim = sub.add_parser("simulate")
    sim.add_argument("csv", type=Path)
    sim.add_argument("--truth-dir", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    if args.command == "simulate":
        simulate(args.csv, args.truth_dir)
        return
    engine = db.get_engine()
    db.create_schema(engine)
    if args.command == "export":
        runs = args.run or recent_live_runs(engine, args.recent_live)
        if not runs:
            print("no live runs in that period")
            return
        export(engine, runs, args.audit_rate, args.seed)
    else:
        import_queue(engine, args.csv)


if __name__ == "__main__":
    main()
