"""
load_runs.py - copy existing run results (runs/<name>/json) into the database.

Use it once to give the database and dashboard the history of runs made before the
database existed, and again after building a baseline so outlier flags are added.
Safe to run repeatedly: each document's earlier rows are replaced.

Usage:
    python src/load_runs.py                     # every run folder under runs/
    python src/load_runs.py --runs digital_qwen7b_v2 sroie_vision_qwen7bvl_v3

For runs made before monitoring existed, the input-quality measurements and file
fingerprints are computed now from the source PDFs, and flags are worked out with the
current rules. The JSON files themselves are not changed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import config
import db
import monitor
import quality
from extract import file_sha256

# The first run predates config.json; these are the settings it was made with.
KNOWN_SETTINGS = {
    "digital_qwen7b": {"input": "data\\synthetic\\digital", "schema": "invoice", "model": "qwen2.5:7b",
                       "prompt": "v1", "text_mode": "lines", "retries": "0"},
}


def run_settings(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    if run_dir.name in KNOWN_SETTINGS:
        return KNOWN_SETTINGS[run_dir.name]
    raise SystemExit(f"{run_dir} has no config.json; add its settings to KNOWN_SETTINGS")


def written_times(run_dir: Path) -> dict[str, datetime]:
    """When each document's result file was written = when it was really processed (UTC)."""
    return {path.stem: datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
            for path in (run_dir / "json").glob("*.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", help="run names (default: all folders in runs/)")
    args = parser.parse_args()

    engine = db.get_engine()
    db.create_schema(engine)

    run_dirs = [config.RUNS_DIR / name for name in args.runs] if args.runs else \
        sorted(p for p in config.RUNS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_"))

    # Pass 1: give documents already in the database their real processing times, for ALL
    # runs, before any flags are worked out, so duplicate checks know which copy came first.
    with engine.begin() as connection:
        for run_dir in run_dirs:
            for doc_id, stamp in written_times(run_dir).items():
                connection.execute(db.documents.update().where(
                    (db.documents.c.run_id == run_dir.name) & (db.documents.c.doc_id == doc_id))
                    .values(processed_at=stamp))

    # Pass 2: load every document with its measurements and flags.
    measured: dict[str, tuple[dict, str]] = {}  # source path -> (quality, sha256); many runs share files
    for run_dir in run_dirs:
        settings = run_settings(run_dir)
        schema, text_mode = settings["schema"], settings["text_mode"]
        db.save_run(engine, run_dir.name, settings)
        baseline = db.load_baselines(engine, schema, text_mode)

        flagged = 0
        paths = sorted((run_dir / "json").glob("*.json"))
        written = written_times(run_dir)
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            source = record["source"]
            if source not in measured:
                pdf = Path(source)
                measured[source] = (quality.measure(pdf), file_sha256(pdf))
            record.setdefault("quality", measured[source][0])
            record.setdefault("file_sha256", measured[source][1])

            record["flags"] = monitor.document_flags(schema, path.stem, record, engine, baseline, run_dir.name)
            record["needs_review"] = any(f["severity"] == monitor.REVIEW for f in record["flags"])
            flagged += record["needs_review"]
            db.save_document(engine, run_dir.name, path.stem, schema, record, written[path.stem])

        print(f"{run_dir.name}: {len(paths)} documents loaded, {flagged} flagged for review "
              f"(baseline metrics: {len(baseline)})")


if __name__ == "__main__":
    main()
