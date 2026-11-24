"""
build_golden.py - assemble the golden set from the development folder (run once, on the dev PC).

The golden set is a fixed sample of documents with known answers, plus the scores the
development PC achieved on them. golden_check.py re-runs it on the server and compares:
if the server's scores match, the server is set up correctly; after any change to the
model, prompt or code, a drop in score shows the change made things worse.

    python src/build_golden.py --dev ".."     # dev = the original OCR folder

Writes golden/<set>/pdf/*.pdf, golden/<set>/truth/*.json and golden/expected.json.

Sets (chosen deterministically, so rebuilding gives the same documents):
    invoice_digital   25 digital invoices  (every 8th of 200)      text mode, qwen2.5:7b
    invoice_scanned   50 scanned invoices  (all)                    image mode, qwen2.5vl:7b
    receipt           105 SROIE receipts   (every 6th of 626)       image mode, qwen2.5vl:7b
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from evaluate import INVOICE_FIELDS, LINE_ITEM_FIELDS, RECEIPT_FIELDS, compare

GOLDEN = Path("golden")

SETS = {
    "invoice_digital": {"schema": "invoice", "model": "qwen2.5:7b", "text_mode": "blocks", "num_ctx": 8192,
                        "source": "data/synthetic/digital", "every": 8, "reference_run": "digital_qwen7b_v2"},
    "invoice_scanned": {"schema": "invoice", "model": "qwen2.5vl:7b", "text_mode": "image", "num_ctx": 16384,
                        "source": "data/synthetic/scanned", "every": 1,
                        "reference_run": "scanned_vision_qwen7bvl_datefix"},
    "receipt": {"schema": "receipt", "model": "qwen2.5vl:7b", "text_mode": "image", "num_ctx": 16384,
                "source": "data/sroie/pdf", "every": 6, "reference_run": "sroie_vision_qwen7bvl_datefix"},
}


def score(schema: str, truth: dict, predicted: dict | None) -> tuple[dict[str, bool], bool]:
    """Per-field right/wrong (fuzzy) and whether the whole document is right."""
    predicted = predicted or {}
    fields = INVOICE_FIELDS if schema == "invoice" else RECEIPT_FIELDS
    results = {}
    for field, kind in fields.items():
        if schema == "receipt" and field == "total" and truth.get("total") in ("", None):
            continue
        results[field] = compare(kind, truth.get(field), predicted.get(field))[1]
    all_right = all(results.values())
    if schema == "invoice":
        items, predicted_items = truth["line_items"], predicted.get("line_items") or []
        items_right = len(items) == len(predicted_items) and all(
            compare(kind, item[f], p.get(f))[1]
            for item, p in zip(items, predicted_items) for f, kind in LINE_ITEM_FIELDS.items())
        results["line_items"] = items_right
        all_right = all_right and items_right
    return results, all_right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=Path, required=True, help="the development OCR folder")
    args = parser.parse_args()

    receipts = pd.read_csv(args.dev / "data/sroie/ground_truth/receipts.csv", dtype=str, keep_default_na=False)
    receipt_truth = {r.receipt_id: {"company": r.company, "address": r.address, "date": r.date_iso, "total": r.total}
                     for r in receipts.itertuples()}

    expected = {}
    for name, spec in SETS.items():
        folder = GOLDEN / name
        if folder.exists():
            shutil.rmtree(folder)
        (folder / "pdf").mkdir(parents=True)
        (folder / "truth").mkdir()

        pdfs = sorted((args.dev / spec["source"]).glob("*.pdf"))[::spec["every"]]
        field_scores, doc_scores = {}, []
        for pdf in pdfs:
            shutil.copy2(pdf, folder / "pdf" / pdf.name)
            if spec["schema"] == "invoice":
                truth = json.loads((args.dev / "data/synthetic/ground_truth/json" / f"{pdf.stem}.json")
                                   .read_text(encoding="utf-8"))
            else:
                truth = receipt_truth[pdf.stem]
            (folder / "truth" / f"{pdf.stem}.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False),
                                                                encoding="utf-8")
            # What the development PC extracted for this document, scored the same way.
            record = json.loads((args.dev / "runs" / spec["reference_run"] / "json" / f"{pdf.stem}.json")
                                .read_text(encoding="utf-8"))
            fields, all_right = score(spec["schema"], truth, record.get("data"))
            for field, right in fields.items():
                field_scores.setdefault(field, []).append(right)
            doc_scores.append(all_right)

        expected[name] = {
            **{k: v for k, v in spec.items() if k not in ("source", "every")},
            "documents": len(pdfs),
            "fully_right": round(sum(doc_scores) / len(doc_scores), 4),
            "field_accuracy": {f: round(sum(v) / len(v), 4) for f, v in field_scores.items()},
        }
        print(f"{name}: {len(pdfs)} documents, reference fully right {expected[name]['fully_right']:.1%}")

    (GOLDEN / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
