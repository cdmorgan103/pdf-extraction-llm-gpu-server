"""
worker.py - the production loop: pick up new PDFs, extract, monitor, store.

    python src/worker.py --once            # process everything waiting, then exit
                                            # (run by Task Scheduler every few minutes)
    python src/worker.py --watch 60        # keep running, checking every 60 seconds

Drop PDFs into   <OCR_HOME>/incoming/invoice/   or   <OCR_HOME>/incoming/receipt/
(the folder says which table format to extract). For each cycle:

  1. Health check: Ollama reachable and both models installed. If not, nothing is
     touched and the worker exits with an error code, so the files are simply picked
     up again on the next run once the problem is fixed.
  2. Each PDF that has finished copying (not modified for SETTLE_SECONDS) is moved to
     archive/<date>/<type>/<name>_<time>.pdf. The time stamp makes every submission a
     separate document, so a file sent twice is caught as a duplicate, not overwritten.
     Files that are not readable PDFs go to archive/<date>/rejected/.
  3. Routing: PDFs with a text layer go to the text model (text mode "blocks"); scans
     go to the vision model (text mode "image"). This follows what testing showed works
     best for each.
  4. Extraction, checks, flags, database writes: exactly as extract.py, into a daily run
     named live_<type>_<mode>_<date>, so drift and accuracy can be followed day by day.

Only one worker runs at a time (a lock file in the logs folder).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

import config
import extract

SETTLE_SECONDS = 15           # a file must be unchanged this long before it is picked up
STALE_LOCK_SECONDS = 6 * 3600  # a lock older than this is assumed to be left over from a crash
SCHEMAS = ("invoice", "receipt")

log = logging.getLogger("ocr.worker")


# ---------------------------------------------------------------------------
# Set-up: logging, single-instance lock, health check
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Log to the console and to logs/worker_<date>.log (extract.py's messages included)."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(config.LOGS_DIR / f"worker_{datetime.now():%Y-%m-%d}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s %(levelname)s %(message)s")


class Lock:
    """Make sure two workers never process the same files at the same time."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < STALE_LOCK_SECONDS:
                raise SystemExit(0)  # another worker is busy; this cycle has nothing to do
            log.warning(f"removing stale lock ({age / 3600:.1f} h old)")
        self.path.write_text(f"{os.getpid()} {datetime.now().isoformat()}")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def ollama_ready() -> bool:
    """Ollama answers and has both models installed."""
    tags_url = config.OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        installed = {m.get("name") for m in requests.get(tags_url, timeout=10).json().get("models", [])}
    except (requests.RequestException, ValueError) as error:
        log.error(f"Ollama not reachable at {tags_url}: {error}")
        return False
    missing = [m for m in (config.TEXT_MODEL, config.VISION_MODEL) if m not in installed]
    if missing:
        log.error(f"Ollama is missing models {missing}; run: ollama pull <model>")
        return False
    return True


# ---------------------------------------------------------------------------
# One cycle
# ---------------------------------------------------------------------------

def waiting_files(schema: str) -> list[Path]:
    """PDFs in incoming/<schema>/ that have finished copying."""
    folder = config.INCOMING / schema
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf" and now - p.stat().st_mtime >= SETTLE_SECONDS)


def archive(path: Path, schema: str, received: datetime) -> Path:
    """Move a file to archive/<date>/<schema>/<name>_<time>.pdf and return the new path."""
    target_dir = config.ARCHIVE / f"{received:%Y-%m-%d}" / schema
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{path.stem}_{received:%H%M%S}.pdf"
    counter = 1
    while target.exists():  # several files with the same name in the same second
        counter += 1
        target = target_dir / f"{path.stem}_{received:%H%M%S}_{counter}.pdf"
    shutil.move(str(path), target)
    return target


def reading_mode(path: Path) -> str | None:
    """'blocks' if the PDF has a text layer, 'image' if it is a scan, None if unreadable."""
    try:
        text = extract.pdf_text(path, "blocks")
    except Exception:  # damaged or not a PDF at all
        return None
    return "blocks" if len(text.strip()) >= 20 else "image"


def run_cycle() -> int:
    """Process everything waiting. Returns the number of documents processed."""
    batches: dict[tuple[str, str], list[Path]] = {}
    for schema in SCHEMAS:
        for path in waiting_files(schema):
            received = datetime.now()
            archived = archive(path, schema, received)
            mode = reading_mode(archived)
            if mode is None:
                rejected = config.ARCHIVE / f"{received:%Y-%m-%d}" / "rejected"
                rejected.mkdir(parents=True, exist_ok=True)
                shutil.move(str(archived), rejected / archived.name)
                log.error(f"{path.name}: not a readable PDF, moved to {rejected}")
                continue
            batches.setdefault((schema, mode), []).append(archived)

    processed = 0
    for (schema, mode), pdfs in batches.items():
        model = config.TEXT_MODEL if mode == "blocks" else config.VISION_MODEL
        context = config.TEXT_CONTEXT if mode == "blocks" else config.VISION_CONTEXT
        run_name = f"live_{schema}_{mode}_{datetime.now():%Y-%m-%d}"
        args = extract.parse_args(["--input", str(pdfs[0].parent), "--schema", schema, "--model", model,
                                   "--text-mode", mode, "--num-ctx", str(context),
                                   "--prompt", config.PROMPT_VERSION, "--run", run_name])
        result = extract.run(args, pdfs)
        flagged = sum(record["needs_review"] for record in result["records"].values())
        log.info(f"{run_name}: {result['processed']} processed, {flagged} flagged for review, "
                 f"{result['errors']} errors")
        processed += result["processed"]
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="process what is waiting, then exit")
    group.add_argument("--watch", type=int, metavar="SECONDS", help="keep running, checking every N seconds")
    args = parser.parse_args()

    setup_logging()
    with Lock(config.LOGS_DIR / "worker.lock"):
        while True:
            if any(waiting_files(schema) for schema in SCHEMAS):
                if ollama_ready():
                    log.info(f"cycle done: {run_cycle()} documents")
                elif args.once:
                    raise SystemExit(1)  # files stay in incoming/ and are retried on the next run
            if args.once:
                return
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
