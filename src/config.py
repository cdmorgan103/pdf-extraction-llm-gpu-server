"""
config.py - every setting that differs between machines, read from environment variables.

On the GPU server these are set once by deploy/setup_server.ps1 (machine-wide), so the
scripts themselves never change between the development PC and the server.

    OCR_HOME            root folder for working data            (default: this project folder)
    OLLAMA_URL          Ollama's chat endpoint                    (default: this machine)
    OCR_DB_URL          SQLAlchemy URL of the results database    (default: none = JSON files only)
    OCR_TEXT_MODEL      model for PDFs with a text layer          (default: qwen2.5:7b)
    OCR_VISION_MODEL    model for scanned / image-only PDFs       (default: qwen2.5vl:7b)

Folders under OCR_HOME (each can be overridden separately, e.g. OCR_INCOMING):
    incoming/<invoice|receipt>/   drop PDFs here; the worker picks them up
    archive/<date>/               every processed PDF, renamed <name>_<time>.pdf (kept as evidence)
    runs/                         per-document JSON results (one folder per daily run)
    reports/  reviews/  logs/     dashboards, review queues, worker logs
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("OCR_HOME", PROJECT))


def folder(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, HOME / default))


INCOMING = folder("OCR_INCOMING", "incoming")
ARCHIVE = folder("OCR_ARCHIVE", "archive")
RUNS_DIR = folder("OCR_RUNS_DIR", "runs")
REPORTS_DIR = folder("OCR_REPORTS_DIR", "reports")
REVIEWS_DIR = folder("OCR_REVIEWS_DIR", "reviews")
LOGS_DIR = folder("OCR_LOGS_DIR", "logs")
GOLDEN_DIR = PROJECT / "golden"   # shipped with the code, not working data

# Ollama's chat endpoint. Normally Ollama runs on the same server as the pipeline.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")

# Results database. Empty = results are only written as JSON/CSV files.
# Network SQL Server with Windows login (service account):
#   mssql+pyodbc://@sqlserver01/ocr_pipeline?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes
# With a SQL login (keep the password in the environment, never in code):
#   mssql+pyodbc://ocr_service:<password>@sqlserver01/ocr_pipeline?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
DB_URL = os.environ.get("OCR_DB_URL", "")

# Models: the worker picks the text model when a PDF has a text layer, otherwise the
# vision model (validated: text 92% on digital invoices; vision 90% on scans, 90% on receipts).
TEXT_MODEL = os.environ.get("OCR_TEXT_MODEL", "qwen2.5:7b")
VISION_MODEL = os.environ.get("OCR_VISION_MODEL", "qwen2.5vl:7b")
TEXT_CONTEXT = int(os.environ.get("OCR_TEXT_CONTEXT", 8192))
VISION_CONTEXT = int(os.environ.get("OCR_VISION_CONTEXT", 16384))
PROMPT_VERSION = os.environ.get("OCR_PROMPT", "v2")

# --- Monitoring thresholds (see monitor.py) ---
OUTLIER_Z = float(os.environ.get("OCR_OUTLIER_Z", 4.0))          # robust z-score above which a value is an outlier
DRIFT_PSI_ALERT = float(os.environ.get("OCR_DRIFT_PSI", 0.25))     # PSI above which a batch has drifted
AUDIT_SAMPLE_RATE = float(os.environ.get("OCR_AUDIT_RATE", 0.05))  # share of unflagged docs sent for review
ALLOWED_CURRENCIES = os.environ.get("OCR_CURRENCIES", "USD,GBP,EUR").split(",")
