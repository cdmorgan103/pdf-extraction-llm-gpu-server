# PDF extraction pipeline: GPU server version

Production version of the pipeline built in the parent `OCR` folder, prepared for a dedicated Windows GPU server. See **[docs/SERVER_PLAN.md](docs/SERVER_PLAN.md)** for the architecture, security, rollout steps and operations.

## What's different from the development folder

| | Development folder | This folder |
|---|---|---|
| Running | Manual commands per experiment | `src/worker.py` picks up PDFs from `incoming\invoice` and `incoming\receipt` automatically |
| Model choice | Chosen per run | Automatic: PDFs with a text layer go to `qwen2.5:7b`; scans go to `qwen2.5vl:7b` |
| Runs | One per experiment | One per day per type: `live_<type>_<mode>_<date>` |
| Files | Read in place | Every PDF archived as `archive\<date>\<type>\<name>_<time>.pdf`; non-PDFs go to `rejected\` |
| Settings | Some fixed paths | Everything from environment variables (`src/config.py`), set once by `deploy\server_settings.ps1` |
| Output | `print` | `logging` to `logs\worker_<date>.log` and task logs |
| Validation | Full test data (700 MB) | `golden\`: 180 labelled documents plus the development PC's scores (`golden/expected.json`); `src/golden_check.py` compares |
| Review export | One run | `--recent-live 7` covers a week of daily runs and skips documents already reviewed |
| Scheduling | None | `deploy\register_tasks.ps1`: worker every 5 min, daily dashboard, weekly review queue, weekly golden check |

The extraction logic, prompts, checks, flags and database layout are unchanged from development. `extract.py` was only restructured so the worker can call it (`run()`).

## Layout

```
src/            pipeline code (same modules as development, plus worker.py, golden_check.py, build_golden.py)
golden/         golden set: <set>/pdf, <set>/truth, expected.json
deploy/         setup_server.ps1, server_settings.example.ps1, register_tasks.ps1, sql/create_database.sql
docs/           SERVER_PLAN.md
requirements.txt
```

Working data (`incoming`, `archive`, `runs`, `reports`, `reviews`, `logs`) lives under `OCR_HOME`, outside the app folder.

## Quick start on the server

```powershell
Copy-Item deploy\server_settings.example.ps1 deploy\server_settings.ps1   # then edit it
.\deploy\setup_server.ps1                                                  # as administrator
.\.venv\Scripts\python.exe src\golden_check.py --set-baseline             # must PASS
.\deploy\register_tasks.ps1 -ServiceAccount "DOMAIN\svc_ocr"
Start-ScheduledTask "OCR - Ollama"
```

Then drop a PDF into `<OCR_HOME>\incoming\invoice` and check the log in `<OCR_HOME>\logs`.

## Day-to-day commands

```powershell
.\.venv\Scripts\python.exe src\worker.py --once                       # process what's waiting now
.\.venv\Scripts\python.exe src\report.py                              # dashboard -> reports\
.\.venv\Scripts\python.exe src\review.py export --recent-live 7       # review queue -> reviews\
.\.venv\Scripts\python.exe src\review.py import reviews\queue_....csv
.\.venv\Scripts\python.exe src\monitor.py drift --run live_invoice_image_2026-11-24
.\.venv\Scripts\python.exe src\golden_check.py                        # before/after any change
```

The development-only tools (`generate_invoices.py`, `prepare_sroie.py`, `make_anomalies.py`, `load_runs.py`, `build_golden.py`) are included for completeness. The server doesn't need them.
