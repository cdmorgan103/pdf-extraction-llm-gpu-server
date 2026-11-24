"""
report.py - build the monitoring dashboard: one self-contained HTML file from the database.

    python src/report.py                  -> reports/dashboard_<date>.html

The file has no external dependencies (the charting library is embedded), so it opens
offline in any browser and can be e-mailed or scheduled. Run again to refresh.

Sections
    Overview          headline numbers for the latest run and overall
    Runs              one row per run: volume, flags, failures, speed, accuracy
    Flags             why documents were sent to review
    Accuracy          by field (heatmap: field x run) and by document (worst first)
    Review loop       live accuracy measured by people: flagged vs random audit sample
    Drift             has each run's input/output shifted from the baseline? (PSI)
    Outliers          documents furthest from normal, with reasons
    Monitoring test   how the unusual-scan test set was handled (if it has been run)
"""

from __future__ import annotations

import html
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.offline
import sqlalchemy as sa

import config
import db
import monitor

OUT_DIR = config.REPORTS_DIR

# Colours (reference data-viz palette): one series = slot 1 blue; sequential = blue ramp;
# status colours only for good/warning/critical, always next to a text label.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]]
STATUS = {"stable": "#0ca30c", "watch": "#fab219", "drift": "#d03b3b"}
# Charts are drawn once with fixed colours, so their text uses the mid-grey that reads on
# both the light and the dark page; gridlines use a translucent grey for the same reason.
INK, MUTED, GRID = "#898781", "#898781", "rgba(137,135,129,0.25)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------

def load(engine) -> dict[str, pd.DataFrame]:
    with engine.connect() as c:
        data = {
            "runs": pd.read_sql(sa.select(db.runs), c),
            "docs": pd.read_sql(sa.select(db.documents), c),
            "flags": pd.read_sql(sa.select(db.document_flags), c),
            "evals": pd.read_sql(sa.select(db.evaluations), c),
        }
    data["flags"] = data["flags"].merge(data["docs"][["document_id", "run_id", "doc_id"]], on="document_id")
    return data


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score; behaves well for small n)."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def doc_accuracy(evals: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, document, source): fields checked, fields right, fully right?"""
    if evals.empty:
        return pd.DataFrame(columns=["run_id", "doc_id", "source", "sample_type", "fields", "right", "fully_right"])
    grouped = evals.assign(sample_type=evals.sample_type.fillna("")).groupby(
        ["run_id", "doc_id", "source", "sample_type"])
    out = grouped.fuzzy.agg(fields="size", right="sum").reset_index()
    out["fully_right"] = out.right == out.fields
    return out


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def table(df: pd.DataFrame, table_id: str, numeric: set[str] = frozenset(), run_column: str | None = None) -> str:
    """An HTML table; click a header to sort. Rows carry data-run for the run filter."""
    head = "".join(f'<th class="{"num" if c in numeric else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        run_attr = f' data-run="{html.escape(str(row[run_column]))}"' if run_column else ""
        cells = "".join(
            f'<td class="{"num" if c in numeric else ""}">{row[c] if str(row[c]).startswith("<span") else html.escape(str(row[c]))}</td>'
            for c in df.columns)
        rows.append(f"<tr{run_attr}>{cells}</tr>")
    return (f'<div class="table-wrap"><table id="{table_id}" class="sortable"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def status_badge(status: str) -> str:
    symbol = {"stable": "●", "watch": "▲", "drift": "■"}.get(status, "")
    # Colour never carries the meaning alone: a shape symbol and the word go with it.
    return f'<span class="badge"><span style="color:{STATUS.get(status, MUTED)}">{symbol}</span> {status}</span>'


def figure_html(fig: go.Figure, height: int) -> str:
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=10, b=10), font=dict(family=FONT, color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_family=FONT),
        showlegend=len(fig.data) > 1, legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def section(title: str, intro: str, body: str, anchor: str) -> str:
    return f'<section id="{anchor}"><h2>{title}</h2><p class="intro">{intro}</p>{body}</section>'


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def overview(data) -> str:
    docs, runs = data["docs"], data["runs"]
    latest = runs.sort_values("started_at").iloc[-1]
    latest_docs = docs[docs.run_id == latest.run_id]
    acc = doc_accuracy(data["evals"])
    audit = acc[(acc.source == "review") & (acc.sample_type == "audit")]
    low, high = wilson(int(audit.fully_right.sum()), len(audit))
    labelled_latest = acc[(acc.source == "labelled") & (acc.run_id == latest.run_id)]

    tiles = [
        ("Documents processed", f"{len(docs):,}", f"{len(runs)} runs"),
        ("Latest run", html.escape(latest.run_id), f"{len(latest_docs)} documents"),
        ("Auto-accepted (latest run)", pct(1 - latest_docs.needs_review.mean()), "not flagged for review"),
        ("Flagged for review (latest run)", f"{int(latest_docs.needs_review.sum())}", pct(latest_docs.needs_review.mean())),
        ("Failed extractions (latest run)", f"{int((latest_docs.status != 'ok').sum())}", "no usable output"),
        ("Median time per document", f"{latest_docs.llm_seconds.median():.1f} s", "model time, latest run"),
        ("Fully right, labelled (latest run)",
         pct(labelled_latest.fully_right.mean()) if len(labelled_latest) else "–",
         f"{len(labelled_latest)} documents with known answers"),
        ("Live accuracy, audit sample",
         pct(audit.fully_right.mean()) if len(audit) else "–",
         f"95% CI {pct(low)}–{pct(high)}, n={len(audit)}" if len(audit) else "no reviewed audit sample yet"),
    ]
    body = '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value{" long" if len(value) > 14 else ""}">{value}</div>'
        f'<div class="tile-sub">{sub}</div></div>' for label, value, sub in tiles) + "</div>"
    return section("Overview", "Headline numbers. Accuracy needs known answers: from labelled test data, "
                   "or from people reviewing a random audit sample of live documents.", body, "overview")


def runs_section(data) -> str:
    docs, runs = data["docs"], data["runs"]
    acc = doc_accuracy(data["evals"])
    labelled = acc[acc.source == "labelled"].groupby("run_id").fully_right.mean()
    reviewed = acc[acc.source == "review"].groupby("run_id").fully_right.agg(["mean", "size"])
    per_run = docs.groupby("run_id").agg(
        documents=("document_id", "size"), flagged=("needs_review", "sum"),
        failed=("status", lambda s: int((s != "ok").sum())), median_seconds=("llm_seconds", "median"))
    per_run = runs.set_index("run_id")[["started_at", "schema", "model", "prompt", "text_mode"]].join(per_run)
    per_run = per_run.sort_values("started_at")
    rows = pd.DataFrame({
        "run": per_run.index,
        "started": per_run.started_at.dt.strftime("%Y-%m-%d %H:%M"),
        "type": per_run.schema, "model": per_run.model, "prompt": per_run.prompt, "reading": per_run.text_mode,
        "documents": per_run.documents.astype(int),
        "flagged": [f"{int(f)} ({f / d:.0%})" for f, d in zip(per_run.flagged, per_run.documents)],
        "failed": per_run.failed.astype(int),
        "median s/doc": per_run.median_seconds.round(1),
        "fully right (labelled)": [pct(labelled.get(r)) for r in per_run.index],
        "fully right (reviewed)": [f"{pct(reviewed.loc[r, 'mean'])} (n={int(reviewed.loc[r, 'size'])})"
                                   if r in reviewed.index else "–" for r in per_run.index],
    })
    fig = go.Figure(go.Bar(x=rows.run, y=per_run.flagged / per_run.documents * 100, marker_color=BLUE,
                           marker_cornerradius=4, hovertemplate="%{x}<br>%{y:.1f}% flagged<extra></extra>"))
    fig.update_yaxes(title="% flagged for review", ticksuffix="%")
    body = figure_html(fig, 280) + table(rows, "runs-table", {"documents", "failed", "median s/doc"}, "run")
    return section("Runs", "Each run is one batch through the pipeline. The share flagged for review is the "
                   "workload sent to people.", body, "runs")


def flags_section(data) -> str:
    flags = data["flags"].copy()
    flags["reason"] = flags.flag_code.str.split(":").str[0] + flags.flag_code.str.contains(":").map(
        {True: ":…", False: ""})
    review = flags[flags.severity == "review"]
    counts = review.groupby("flag_code").document_id.nunique().sort_values()
    fig = go.Figure(go.Bar(y=counts.index, x=counts.values, orientation="h", marker_color=BLUE,
                           marker_cornerradius=4, hovertemplate="%{y}: %{x} documents<extra></extra>"))
    fig.update_xaxes(title="documents (all runs)")
    by_run = (flags.groupby(["run_id", "severity", "flag_code"]).document_id.nunique()
              .reset_index(name="documents").sort_values(["run_id", "documents"], ascending=[True, False]))
    by_run.columns = ["run", "severity", "flag", "documents"]
    body = (figure_html(fig, max(220, 24 * len(counts) + 60))
            + "<h3>By run (review = sent to a person, info = recorded only)</h3>"
            + table(by_run, "flags-table", {"documents"}, "run"))
    return section("Flags", "Why documents were sent for review. Every flag is label-free: rules, input-quality "
                   "guardrails, duplicates, and outliers against the baseline.", body, "flags")


def accuracy_section(data) -> str:
    evals = data["evals"]
    if evals.empty:
        return section("Accuracy", "No evaluations yet.", "", "accuracy")
    labelled = evals[evals.source == "labelled"].copy()
    labelled["field_group"] = labelled.field.where(~labelled.field.str.startswith("item."), labelled.field)
    pivot = labelled.groupby(["field_group", "run_id"]).fuzzy.mean().unstack() * 100
    field_order = list(dict.fromkeys(labelled.field_group))
    pivot = pivot.reindex(field_order)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=SEQUENTIAL, zmin=50, zmax=100,
        text=[[f"{v:.0f}" if not math.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2,
        colorbar=dict(title="% right", ticksuffix="%"),
        hovertemplate="%{y}<br>%{x}<br>%{z:.1f}% right<extra></extra>"))
    fig.update_yaxes(autorange="reversed")

    acc = doc_accuracy(evals)
    worst = acc[~acc.fully_right].copy()
    worst["accuracy"] = worst.right / worst.fields
    flags = data["flags"][data["flags"].severity == "review"].groupby(["run_id", "doc_id"]).flag_code.apply(", ".join)
    worst["flags"] = [flags.get((r, d), "") for r, d in zip(worst.run_id, worst.doc_id)]
    worst = worst.sort_values(["accuracy", "run_id"]).head(300)
    worst_rows = pd.DataFrame({"run": worst.run_id, "document": worst.doc_id, "source": worst.source,
                               "fields right": [f"{int(r)} / {int(f)}" for r, f in zip(worst.right, worst.fields)],
                               "accuracy": worst.accuracy.map(lambda x: f"{x:.0%}"),
                               "flagged for review": worst["flags"]})  # not worst.flags: that's a pandas attribute
    summary = acc.groupby(["run_id", "source"]).fully_right.agg(["mean", "size"]).reset_index()
    summary = pd.DataFrame({"run": summary.run_id, "source": summary.source,
                            "documents": summary["size"], "fully right": summary["mean"].map(pct)})
    body = ("<h3>Accuracy by field (% right, labelled test data)</h3>"
            + figure_html(fig, 40 + 26 * len(pivot))
            + "<h3>Accuracy by document</h3>" + table(summary, "doc-summary", {"documents"}, "run")
            + "<h3>Documents with errors (worst first; flagged = monitoring caught it)</h3>"
            + table(worst_rows, "worst-docs", set(), "run"))
    return section("Accuracy", "Field-level and document-level accuracy wherever correct answers exist. "
                   "A wrong document that was not flagged is an error monitoring missed.", body, "accuracy")


def review_section(data) -> str:
    acc = doc_accuracy(data["evals"])
    reviewed = acc[acc.source == "review"]
    if reviewed.empty:
        body = ("<p>No reviews imported yet. Export a queue with <code>python src/review.py export --run …</code>, "
                "fill it in, and import it.</p>")
        return section("Review loop", "Live accuracy, measured by people.", body, "review")
    rows = []
    for (run, sample), group in reviewed.groupby(["run_id", "sample_type"]):
        low, high = wilson(int(group.fully_right.sum()), len(group))
        rows.append({"run": run, "sample": sample, "documents reviewed": len(group),
                     "fully right": pct(group.fully_right.mean()), "95% CI": f"{pct(low)} – {pct(high)}",
                     "fields corrected": int((group.fields - group.right).sum())})
    body = table(pd.DataFrame(rows), "review-table", {"documents reviewed", "fields corrected"}, "run")
    return section("Review loop", "The <b>audit</b> row estimates the accuracy of documents that were "
                   "auto-accepted (a random sample, so it is unbiased). The <b>flagged</b> row shows how often "
                   "flagged documents really were wrong (how precise the flags are).", body, "review")


def drift_section(engine, data) -> str:
    rows = []
    for run in data["runs"].sort_values("started_at").run_id:
        for row in monitor.drift(engine, run):
            rows.append({"run": run, "metric": row["metric"], "PSI": row["psi"],
                         "status": status_badge(row["status"]), "documents": row["n"]})
    if not rows:
        return section("Drift", "No baselines yet (python src/monitor.py build-baseline …).", "", "drift")
    body = table(pd.DataFrame(rows), "drift-table", {"PSI", "documents"}, "run")
    return section("Drift", "Population Stability Index of each metric against the baseline for that document type "
                   "and reading mode. Below 0.1 stable, 0.1–0.25 watch, above 0.25 the batch has shifted: "
                   "check whether documents changed (new supplier, new scanner) before trusting accuracy.",
                   body, "drift")


def outliers_section(data) -> str:
    flags = data["flags"]
    out = flags[flags.flag_code.str.startswith("outlier:")].copy()
    if out.empty:
        return section("Outliers", "No outliers flagged.", "", "outliers")
    out["z"] = out.detail.str.extract(r"z = ([+-][\d.]+)").astype(float)
    out = out.reindex(out.z.abs().sort_values(ascending=False).index).head(200)
    rows = pd.DataFrame({"run": out.run_id, "document": out.doc_id, "metric": out.flag_code.str[8:],
                         "robust z": out.z, "severity": out.severity})
    return section("Outliers", "Documents furthest from normal (robust z-score above the threshold). "
                   "Input-statistic outliers are 'info': on the receipts they were not more often wrong.",
                   table(rows, "outliers-table", {"robust z"}, "run"), "outliers")


def anomaly_section() -> str:
    reports = sorted(config.RUNS_DIR.glob("*/anomaly_report.csv"))
    if not reports:
        return ""
    parts = []
    for path in reports:
        results = pd.read_csv(path)
        summary = results.groupby("kind", sort=False).agg(
            documents=("doc_id", "size"), flagged=("flagged", "mean"), correct=("correct", "mean"),
            handled=("handled", "mean")).reset_index()
        for column in ("flagged", "correct", "handled"):
            summary[column] = summary[column].map(lambda x: "–" if pd.isna(x) else f"{x:.0%}")
        parts.append(f"<h3>{html.escape(path.parent.name)}</h3>"
                     + table(summary, f"anomaly-{path.parent.name}", {"documents"}))
    return section("Monitoring test", "Unusual scans generated on purpose (src/make_anomalies.py). "
                   "<b>handled</b> = flagged for review, or extracted fully correctly anyway. For the "
                   "<b>normal</b> row, flagged = false-flag rate.", "".join(parts), "test")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; }
@media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#898781; --line:#2c2c2a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
header { padding:24px 16px 8px; max-width:1200px; margin:auto; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:19px; margin:0 0 4px; } h3 { font-size:15px; margin:20px 0 8px; }
.sub, .intro { color:var(--ink2); margin:0 0 12px; font-size:14px; line-height:1.45; }
nav { position:sticky; top:0; background:var(--page); padding:8px 16px; border-bottom:1px solid var(--line); z-index:2; }
nav .inner { max-width:1200px; margin:auto; display:flex; flex-wrap:wrap; gap:6px 16px; align-items:center; font-size:14px; }
nav a { color:var(--ink2); text-decoration:none; } nav a:hover { color:var(--ink); }
nav select { margin-left:auto; font:inherit; padding:4px 6px; border:1px solid var(--line); border-radius:6px;
  background:var(--surface); color:var(--ink); }
main { max-width:1200px; margin:auto; padding:8px 16px 48px; }
section { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px; margin:16px 0; }
.tiles { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px; }
.tile { border:1px solid var(--line); border-radius:8px; padding:12px; }
.tile-label { font-size:12px; color:var(--ink2); } .tile-value { font-size:24px; font-weight:600; margin:4px 0;
  overflow-wrap:anywhere; } .tile-value.long { font-size:16px; line-height:1.5; } .tile-sub { font-size:12px; color:var(--muted); }
.table-wrap { overflow-x:auto; max-height:520px; overflow-y:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
th { position:sticky; top:0; background:var(--surface); cursor:pointer; color:var(--ink2); font-weight:600; }
th:hover { color:var(--ink); } td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.hidden { display:none; }
.badge { white-space:nowrap; }
code { font-size:12px; }
footer { color:var(--muted); font-size:12px; max-width:1200px; margin:auto; padding:0 16px 32px; }
"""

JS = """
document.querySelectorAll('table.sortable th').forEach((th, col) => th.addEventListener('click', () => {
  const body = th.closest('table').tBodies[0];
  const asc = th.dataset.dir !== 'asc'; th.closest('tr').querySelectorAll('th').forEach(h => delete h.dataset.dir);
  th.dataset.dir = asc ? 'asc' : 'desc';
  const key = r => { const t = r.cells[col].innerText.replace(/[%,]/g, ''); const n = parseFloat(t);
    return isNaN(n) ? t.toLowerCase() : n; };
  [...body.rows].sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * (asc ? 1 : -1))
    .forEach(r => body.appendChild(r));
}));
const pick = document.getElementById('run-filter');
pick.addEventListener('change', () => document.querySelectorAll('tr[data-run]').forEach(r =>
  r.classList.toggle('hidden', pick.value !== '' && r.dataset.run !== pick.value)));
"""


def build(engine) -> Path:
    data = load(engine)
    if data["docs"].empty:
        raise SystemExit("the database has no documents yet")
    run_options = "".join(f'<option value="{html.escape(r)}">{html.escape(r)}</option>'
                          for r in data["runs"].sort_values("started_at").run_id)
    sections = [overview(data), runs_section(data), flags_section(data), accuracy_section(data),
                review_section(data), drift_section(engine, data), outliers_section(data), anomaly_section()]
    nav_links = [("overview", "Overview"), ("runs", "Runs"), ("flags", "Flags"), ("accuracy", "Accuracy"),
                 ("review", "Review loop"), ("drift", "Drift"), ("outliers", "Outliers"), ("test", "Monitoring test")]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Extraction Monitoring</title>
<style>{CSS}</style><script>{plotly.offline.get_plotlyjs()}</script></head><body>
<header><h1>Extraction monitoring</h1>
<p class="sub">Generated {date.today().isoformat()} from the results database. Tables: click a heading to sort.</p></header>
<nav><div class="inner">{''.join(f'<a href="#{a}">{t}</a>' for a, t in nav_links)}
<select id="run-filter" aria-label="Filter tables by run"><option value="">All runs</option>{run_options}</select></div></nav>
<main>{''.join(sections)}</main>
<footer>Accuracy figures come only from documents with known answers (labelled test data or human review).
Flags, drift and outliers need no answers.</footer>
<script>{JS}</script></body></html>"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"dashboard_{date.today().isoformat()}.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    engine = db.get_engine()
    db.create_schema(engine)
    path = build(engine)
    print(f"dashboard written to {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
