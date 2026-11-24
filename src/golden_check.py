"""
golden_check.py - prove the server extracts as well as the development PC did.

    python src/golden_check.py                     # all three golden sets (~15 min on a 3090)
    python src/golden_check.py --limit 5           # quick smoke test: 5 documents per set
    python src/golden_check.py --set-baseline      # if it passes, build the monitoring baselines

Run it:
  - once after setting up the server (acceptance test),
  - before and after any change to the model, prompt, code or Ollama version,
  - on a schedule (e.g. monthly), to catch silent changes.

Each set in golden/expected.json is re-extracted with its recorded settings and scored
against the known answers. A set PASSES if its share of fully-right documents is no more
than TOLERANCE below what the development PC achieved on the same documents. The script
exits with code 1 if any set fails, so a scheduler or script can react.

With --set-baseline, the golden runs (validated by passing) become the baseline that
outlier and drift checks compare live documents with, until enough reviewed live data
exists to rebuild it from that (python src/monitor.py build-baseline --runs ...).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import config
import extract
from build_golden import score

TOLERANCE = 0.03   # allowed drop in "fully right" share before a set fails

log = logging.getLogger("ocr.golden")


def check_set(name: str, spec: dict, stamp: str, limit: int | None) -> dict:
    folder = config.GOLDEN_DIR / name
    pdfs = sorted((folder / "pdf").glob("*.pdf"))[:limit]
    run_name = f"golden_{name}_{stamp}"
    args = extract.parse_args(["--input", str(folder / "pdf"), "--schema", spec["schema"], "--model", spec["model"],
                               "--text-mode", spec["text_mode"], "--num-ctx", str(spec["num_ctx"]),
                               "--prompt", config.PROMPT_VERSION, "--run", run_name])
    result = extract.run(args, pdfs)

    fully_right, field_scores = [], {}
    for pdf in pdfs:
        truth = json.loads((folder / "truth" / f"{pdf.stem}.json").read_text(encoding="utf-8"))
        record = result["records"].get(pdf.stem) or json.loads(
            (result["run_dir"] / "json" / f"{pdf.stem}.json").read_text(encoding="utf-8"))
        fields, all_right = score(spec["schema"], truth, record.get("data"))
        fully_right.append(all_right)
        for field, right in fields.items():
            field_scores.setdefault(field, []).append(right)

    achieved = sum(fully_right) / len(fully_right)
    # With --limit, compare against the reference on the same documents, not the whole set.
    reference = spec["fully_right"] if limit is None else None
    passed = reference is None or achieved >= reference - TOLERANCE
    return {"set": name, "run": run_name, "documents": len(pdfs), "fully_right": achieved,
            "reference": reference, "passed": passed,
            "field_accuracy": {f: sum(v) / len(v) for f, v in field_scores.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="*", help="which golden sets (default: all)")
    parser.add_argument("--limit", type=int, help="only the first N documents per set (smoke test; no pass/fail)")
    parser.add_argument("--set-baseline", action="store_true", help="if all sets pass, build baselines from them")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    expected = json.loads((config.GOLDEN_DIR / "expected.json").read_text(encoding="utf-8"))
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    results = [check_set(name, spec, stamp, args.limit)
               for name, spec in expected.items() if not args.sets or name in args.sets]

    print("\nGolden set check")
    print(f"{'set':18} {'docs':>5} {'fully right':>12} {'reference':>10}  result")
    for r in results:
        reference = f"{r['reference']:.1%}" if r["reference"] is not None else "-"
        verdict = "PASS" if r["passed"] else "FAIL"
        if args.limit:
            verdict = "(smoke test)"
        print(f"{r['set']:18} {r['documents']:>5} {r['fully_right']:>12.1%} {reference:>10}  {verdict}")

    if not all(r["passed"] for r in results):
        raise SystemExit("golden check FAILED: do not use this server / change until explained")

    if args.set_baseline and not args.limit:
        import db
        import monitor
        engine = db.get_engine()
        monitor.build_baseline(engine, [r["run"] for r in results])
        print("baselines built from the golden runs")


if __name__ == "__main__":
    main()
