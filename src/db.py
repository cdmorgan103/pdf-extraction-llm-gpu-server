"""
db.py - store pipeline results in a SQL database (SQL Server, or SQLite for tests).

The tables (created by create_schema(), safe to run repeatedly):

    runs                one row per extraction run (model, prompt, settings)
    documents           one row per document per run: status, timings, review flag,
                        and the input-quality measurements used for outlier detection
    invoice_headers     extracted invoice fields          (one row per invoice document)
    invoice_line_items  extracted line items              (many rows per invoice document)
    receipts            extracted receipt fields          (one row per receipt document)
    document_flags      every reason a document was flagged (one row per reason)
    reviews             values checked by a person (the review loop's ground truth)
    evaluations         field-by-field right/wrong, from labelled data or reviews
    baselines           what "normal" looks like for each metric (for outlier/drift checks)

Uses SQLAlchemy Core: tables are defined in Python and SQLAlchemy writes the right
SQL for whichever database the URL points to.

Usage from other scripts:
    import db
    engine = db.get_engine()          # reads OCR_DB_URL (see config.py)
    db.create_schema(engine)
    db.save_run(engine, run_name, settings)
    db.save_document(engine, run_name, record)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import sqlalchemy as sa

import config

metadata = sa.MetaData()

# Short-hands for column types. Unicode = NVARCHAR on SQL Server (needed for ü, é, £ ...).
Text = sa.UnicodeText
def Name(length=400):
    return sa.Unicode(length)


# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

runs = sa.Table(
    "runs", metadata,
    sa.Column("run_id", Name(200), primary_key=True),       # the --run name, e.g. sroie_vision_qwen7bvl_v3
    sa.Column("schema", Name(50)),                          # invoice / receipt
    sa.Column("model", Name(200)),
    sa.Column("prompt", Name(20)),
    sa.Column("text_mode", Name(20)),
    sa.Column("settings_json", Text),                       # every command-line setting
    sa.Column("started_at", sa.DateTime),
)

documents = sa.Table(
    "documents", metadata,
    sa.Column("document_id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", Name(200), sa.ForeignKey("runs.run_id"), nullable=False),
    sa.Column("doc_id", Name(200), nullable=False),         # file name without .pdf
    sa.Column("source_path", Name(1000)),
    sa.Column("file_sha256", sa.String(64), index=True),    # fingerprint of the file, finds duplicates
    sa.Column("schema", Name(50)),
    sa.Column("status", Name(30)),                          # ok / no_text_layer / invalid / error
    sa.Column("needs_review", sa.Boolean),
    sa.Column("attempts", sa.Integer),
    sa.Column("llm_seconds", sa.Float),
    sa.Column("read_seconds", sa.Float),
    sa.Column("chars", sa.Integer),                         # length of text sent (0 in image mode)
    sa.Column("error", Text),
    sa.Column("processed_at", sa.DateTime),
    # Input-quality measurements (see quality.py)
    sa.Column("pages", sa.Integer),
    sa.Column("file_kb", sa.Float),
    sa.Column("effective_dpi", sa.Float),                   # resolution of the scanned image, if any
    sa.Column("ink_ratio", sa.Float),                       # share of dark pixels (~0 = blank page)
    sa.Column("brightness", sa.Float),
    sa.Column("contrast", sa.Float),
    sa.Column("sharpness", sa.Float),                       # low = blurry
    sa.Column("skew_degrees", sa.Float),
    sa.UniqueConstraint("run_id", "doc_id", name="uq_documents_run_doc"),
)

invoice_headers = sa.Table(
    "invoice_headers", metadata,
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.document_id"), primary_key=True),
    sa.Column("invoice_number", Name(200)),
    sa.Column("invoice_date", Name(30)),
    sa.Column("due_date", Name(30)),
    sa.Column("vendor_name", Name()),
    sa.Column("vendor_address", Name(1000)),
    sa.Column("customer_name", Name()),
    sa.Column("customer_address", Name(1000)),
    sa.Column("currency", Name(10)),
    sa.Column("subtotal", sa.Float),
    sa.Column("tax_rate", sa.Float),
    sa.Column("tax_amount", sa.Float),
    sa.Column("shipping", sa.Float),
    sa.Column("total", sa.Float),
)

invoice_line_items = sa.Table(
    "invoice_line_items", metadata,
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.document_id"), primary_key=True),
    sa.Column("line_no", sa.Integer, primary_key=True),
    sa.Column("description", Name(1000)),
    sa.Column("quantity", sa.Float),
    sa.Column("unit_price", sa.Float),
    sa.Column("amount", sa.Float),
)

receipts = sa.Table(
    "receipts", metadata,
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.document_id"), primary_key=True),
    sa.Column("company", Name()),
    sa.Column("address", Name(1000)),
    sa.Column("date", Name(30)),
    sa.Column("total", sa.Float),
)

document_flags = sa.Table(
    "document_flags", metadata,
    sa.Column("flag_id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.document_id"), nullable=False, index=True),
    sa.Column("flag_code", Name(100)),      # e.g. check_failed, outlier:sharpness, missing:total
    sa.Column("severity", Name(20)),        # review = send to a person; info = just recorded
    sa.Column("detail", Name(1000)),
)

reviews = sa.Table(
    "reviews", metadata,
    sa.Column("review_id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.document_id"), nullable=False, index=True),
    sa.Column("field", Name(100)),
    sa.Column("extracted_value", Name(1000)),
    sa.Column("corrected_value", Name(1000)),   # what the reviewer says is right
    sa.Column("sample_type", Name(20)),         # flagged / audit
    sa.Column("reviewer", Name(200)),
    sa.Column("reviewed_at", sa.DateTime),
)

evaluations = sa.Table(
    "evaluations", metadata,
    sa.Column("evaluation_id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", Name(200), index=True),
    sa.Column("doc_id", Name(200)),
    sa.Column("field", Name(100)),
    sa.Column("exact", sa.Boolean),
    sa.Column("fuzzy", sa.Boolean),
    sa.Column("source", Name(30)),              # labelled (test data with answers) / review (checked by a person)
    sa.Column("sample_type", Name(20)),         # for reviews: flagged / audit
    sa.Column("evaluated_at", sa.DateTime),
)

baselines = sa.Table(
    "baselines", metadata,
    sa.Column("schema", Name(50), primary_key=True),
    sa.Column("text_mode", Name(20), primary_key=True),
    sa.Column("metric", Name(100), primary_key=True),
    sa.Column("median", sa.Float),
    sa.Column("scale", sa.Float),               # robust spread (MAD-based), for z-scores
    sa.Column("count", sa.Integer),
    sa.Column("bins_json", Text),               # decile edges + shares, for drift (PSI)
    sa.Column("built_from", Name(1000)),        # which runs this baseline came from
    sa.Column("built_at", sa.DateTime),
)


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------

def get_engine(url: str | None = None) -> sa.Engine:
    """Connect to the results database (OCR_DB_URL unless a URL is given)."""
    url = url or config.DB_URL
    if not url:
        raise SystemExit("No database configured: set OCR_DB_URL (see src/config.py).")
    if url.startswith("mssql"):
        ensure_sqlserver_database(url)
        # fast_executemany makes bulk inserts (e.g. line items) much quicker on SQL Server.
        return sa.create_engine(url, fast_executemany=True)
    return sa.create_engine(url)


def ensure_sqlserver_database(url: str) -> None:
    """Create the SQL Server database itself if it doesn't exist yet."""
    parsed = sa.engine.make_url(url)
    database = parsed.database
    master = sa.create_engine(parsed.set(database="master"), isolation_level="AUTOCOMMIT")
    with master.connect() as connection:
        connection.execute(sa.text(f"IF DB_ID(N'{database}') IS NULL CREATE DATABASE [{database}]"))
    master.dispose()


def create_schema(engine: sa.Engine) -> None:
    """Create any tables that don't exist yet. Existing tables and data are left alone."""
    metadata.create_all(engine)


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # stored as UTC


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------

def save_run(engine: sa.Engine, run_id: str, settings: dict) -> None:
    """Insert or update the row describing a run."""
    row = {
        "run_id": run_id,
        "schema": settings.get("schema"),
        "model": settings.get("model"),
        "prompt": settings.get("prompt"),
        "text_mode": settings.get("text_mode"),
        "settings_json": json.dumps(settings),
    }
    with engine.begin() as connection:
        exists = connection.execute(sa.select(runs.c.run_id).where(runs.c.run_id == run_id)).first()
        if exists:
            connection.execute(runs.update().where(runs.c.run_id == run_id).values(**row))
        else:
            connection.execute(runs.insert().values(**row, started_at=now()))


def save_document(engine: sa.Engine, run_id: str, doc_id: str, schema: str, record: dict,
                  processed_at: datetime | None = None) -> int:
    """Write one document's result (the record extract.py also saves as JSON).

    Replaces any earlier result for the same run and document, keeping its document_id
    so that human reviews attached to it survive a re-run or backfill. Everything is
    written in one transaction, so a crash never leaves half a document in the database.
    Returns the document_id. processed_at: when it was really processed (backfills pass
    the time the result file was written); default now for new, unchanged for re-saves.
    """
    quality = record.get("quality") or {}
    data = record.get("data") or {}
    values = dict(
        source_path=record.get("source"),
        file_sha256=record.get("file_sha256"),
        schema=schema,
        status=record.get("status"),
        needs_review=bool(record.get("needs_review")),
        attempts=record.get("attempts"),
        llm_seconds=record.get("seconds"),
        read_seconds=record.get("read_seconds"),
        chars=record.get("chars"),
        error=record.get("error"),
        **({"processed_at": processed_at} if processed_at else {}),
        **{name: quality.get(name) for name in
           ("pages", "file_kb", "effective_dpi", "ink_ratio", "brightness", "contrast", "sharpness",
            "skew_degrees")},
    )

    with engine.begin() as connection:
        old = connection.execute(sa.select(documents.c.document_id).where(
            (documents.c.run_id == run_id) & (documents.c.doc_id == doc_id))).first()
        if old:
            # Same document again: replace its extracted values and flags, but keep its
            # reviews and its processed_at (the first time it came in, used for duplicates).
            document_id = old.document_id
            for table in (invoice_line_items, invoice_headers, receipts, document_flags):
                connection.execute(table.delete().where(table.c.document_id == document_id))
            connection.execute(documents.update().where(documents.c.document_id == document_id).values(**values))
        else:
            values.setdefault("processed_at", now())
            document_id = connection.execute(documents.insert().values(
                run_id=run_id, doc_id=doc_id, **values)).inserted_primary_key[0]

        if data and schema == "invoice":
            header = {key: value for key, value in data.items() if key != "line_items"}
            connection.execute(invoice_headers.insert().values(document_id=document_id, **header))
            items = [{"document_id": document_id, **item} for item in data.get("line_items") or []]
            if items:
                connection.execute(invoice_line_items.insert(), items)
        elif data and schema == "receipt":
            connection.execute(receipts.insert().values(document_id=document_id, **data))

        flags = [{"document_id": document_id, **flag} for flag in record.get("flags") or []]
        if flags:
            connection.execute(document_flags.insert(), flags)

    return document_id


def save_evaluations(engine: sa.Engine, run_id: str, rows: list[dict], source: str = "labelled") -> None:
    """Replace the labelled-data evaluation rows for a run (rows: doc_id, field, exact, fuzzy)."""
    with engine.begin() as connection:
        connection.execute(evaluations.delete().where(
            (evaluations.c.run_id == run_id) & (evaluations.c.source == source)))
        stamp = now()
        if rows:
            connection.execute(evaluations.insert(), [
                {"run_id": run_id, "source": source, "evaluated_at": stamp, **row} for row in rows])


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def first_processed_at(engine: sa.Engine, run_id: str, doc_id: str) -> datetime | None:
    """When this run first processed this document (None if it hasn't yet)."""
    with engine.connect() as connection:
        return connection.execute(sa.select(documents.c.processed_at).where(
            (documents.c.run_id == run_id) & (documents.c.doc_id == doc_id))).scalar()


def _earlier(cutoff: datetime | None):
    """Only documents processed before `cutoff`: the *later* copy is the duplicate."""
    return documents.c.processed_at < cutoff if cutoff else sa.true()


def other_documents_with_hash(engine: sa.Engine, file_sha256: str, doc_id: str,
                              cutoff: datetime | None = None) -> list[str]:
    """doc_ids of *different*, earlier files with identical content (a resubmitted document)."""
    with engine.connect() as connection:
        result = connection.execute(sa.select(documents.c.doc_id).distinct().where(
            (documents.c.file_sha256 == file_sha256) & (documents.c.doc_id != doc_id) & _earlier(cutoff)))
        return [row.doc_id for row in result]


def other_invoices_with_number(engine: sa.Engine, vendor_name: str, invoice_number: str, doc_id: str,
                               cutoff: datetime | None = None) -> list[str]:
    """doc_ids of *different*, earlier documents with the same vendor and invoice number."""
    if not vendor_name or not invoice_number:
        return []
    with engine.connect() as connection:
        result = connection.execute(
            sa.select(documents.c.doc_id).distinct()
            .select_from(invoice_headers.join(documents))
            .where((invoice_headers.c.vendor_name == vendor_name)
                   & (invoice_headers.c.invoice_number == invoice_number)
                   & (documents.c.doc_id != doc_id) & _earlier(cutoff)))
        return [row.doc_id for row in result]


def load_baselines(engine: sa.Engine, schema: str, text_mode: str) -> dict[str, dict]:
    """The stored baseline for each metric of this document type and reading mode."""
    with engine.connect() as connection:
        result = connection.execute(sa.select(baselines).where(
            (baselines.c.schema == schema) & (baselines.c.text_mode == text_mode)))
        return {row.metric: {"median": row.median, "scale": row.scale, "bins": json.loads(row.bins_json or "null")}
                for row in result}
