# GPU server plan: PDF extraction pipeline

## 1. Goal

Move the extraction pipeline from the development PC to a dedicated Windows GPU server on the local network:
- documents dropped into a shared folder are processed automatically
- results land in the network SQL Server
- monitoring flags documents for review
- a dashboard and a weekly review queue are produced on a schedule

Nothing leaves the local network: the models run on the server's GPU.

## 2. Architecture

```
 Scanners / users                       GPU SERVER (Windows)                              SQL SERVER (network)
 ────────────────                       ─────────────────────────────────────────         ───────────────────
 \\gpu-server\ocr-incoming  ──────►  incoming\invoice\  incoming\receipt\
   (SMB share, write-only)                    │  every 5 min: worker.py --once
                                              ▼
                                      archive\<date>\...   (every PDF kept, time-stamped)
                                              │
                                      route: text layer? ──yes──► qwen2.5:7b   (text)  ┐
                                                         └─no───► qwen2.5vl:7b (image) ├─ Ollama on 127.0.0.1:11434 (GPU)
                                              │                                        ┘
                                      checks · OCR cross-check · flags · outliers
                                              │
                                      runs\<run>\json  (audit trail) ───────────────►  ocr_pipeline database
                                                                                      (documents, extracted tables,
 Reviewers  ◄── reviews\queue_*.csv (Mon)                                              flags, reviews, evaluations,
 Managers   ◄── reports\dashboard_*.html (daily)  ◄── report.py ◄────────────────────  baselines)
```

The pipeline runs on the GPU server next to Ollama, so Ollama never needs to be reachable from the network. SQL Server stays on its own machine, where it is already backed up and managed.

## 3. Hardware and software

| Item | Requirement | Why |
|---|---|---|
| GPU | NVIDIA with **≥ 16 GB** VRAM; 24 GB (RTX 3090/4090, A5000) recommended | Both 7B models loaded together use ~15 GB (measured 6.6 GB text + ~9 GB vision). 24 GB leaves room for a larger model later. |
| CPU / RAM | 8+ cores, 32 GB RAM | The OCR cross-check and PDF rendering run on the CPU (~2 s per page). |
| Disk | SSD, 500 GB+ for data | Archived scans are ~0.2–0.9 MB each: about 0.3 TB per year at 1,000 scans a day. |
| OS | Windows Server 2022/2025 or Windows 11 Pro, current NVIDIA driver | Same tooling as the development PC. |
| Software | Installed by `deploy\setup_server.ps1`: Ollama, Python 3.11, ODBC Driver 18, sqlcmd. Python packages pinned in `requirements.txt`. | Same versions as development, so results match. |
| Internet | Only during setup (winget, pip, model download) | After that the server can run fully offline. |

## 4. Security

- **Ollama stays local** (`OLLAMA_HOST=127.0.0.1`). Ollama has **no authentication**: anyone who can reach port 11434 can use the models and see what is sent to them. If developers need the server's GPU, run `setup_server.ps1 -ExposeOllamaToLan -LanSubnet <subnet>`. That opens the port **only** to that subnet through Windows Firewall.
- **Service account** (e.g. `DOMAIN\svc_ocr`): runs all scheduled tasks. Rights:
  - read/write on `OCR_HOME`
  - in SQL: `db_datareader`, `db_datawriter` and `db_ddladmin` on `ocr_pipeline` only (`deploy\sql\create_database.sql`)
  - no administrator rights
- **Database login**: Windows authentication, so no password is stored anywhere. If a SQL login is unavoidable, its password lives only in the machine environment variable, not in files.
- **Connection encryption**: the development URL uses `TrustServerCertificate=yes`. In production, give SQL Server a certificate the server trusts and remove that option, so the connection is encrypted *and* verified.
- **Incoming share**: users get write access to `incoming\` only. `archive\`, `runs\` and `reviews\` hold document contents, which may include personal and financial data. Limit them to the service account and reviewers.
- **Retention**: decide how long archived PDFs and JSON results are kept (e.g. 7 years for invoices, or whatever your finance policy says). Purge on a schedule.

## 5. Rollout

| Phase | What | Done when |
|---|---|---|
| 0. Prepare | Server built, GPU driver installed, service account created, SMB share for `incoming\` planned | `nvidia-smi` shows the GPU |
| 1. Database | DBA runs `deploy\sql\create_database.sql -v ServiceAccount="DOMAIN\svc_ocr"` on the SQL Server | Login works |
| 2. Install | Copy this folder to the server (e.g. `D:\ocr-app`). Fill in `deploy\server_settings.ps1`, run `deploy\setup_server.ps1` as administrator | Script ends with "database OK" |
| 3. Acceptance | `.\.venv\Scripts\python.exe src\golden_check.py --set-baseline` | All three sets **PASS**; baselines built |
| 4. Schedule | `deploy\register_tasks.ps1 -ServiceAccount DOMAIN\svc_ocr`, then `Start-ScheduledTask 'OCR - Ollama'` | Test PDFs dropped into `incoming\` appear in SQL within 5 minutes |
| 5. Pilot (2–4 weeks) | Run alongside the current manual process. Set `OCR_AUDIT_RATE=0.2` (review 20% of unflagged documents) | Audit-sample accuracy is known with a tight confidence interval; flag reasons understood |
| 6. Re-baseline | Rebuild baselines from reviewed pilot runs: `src\monitor.py build-baseline --runs live_...` | Drift on new days is "stable" |
| 7. Go live | Set `OCR_AUDIT_RATE=0.05`. The manual process becomes "review flagged + audit sample" | Sign-off on live accuracy and review workload |

## 6. Operations

| When | What | How |
|---|---|---|
| Every 5 min | Process new documents | Task `OCR - Worker` (`worker.py --once`). Exits with code 1 if Ollama is down; files wait in `incoming\` and are retried automatically. |
| Daily 06:00 | Dashboard | Task `OCR - Daily report` writes `reports\dashboard_<date>.html`. Check flag rate, failures and drift status. |
| Monday 07:00 | Review queue | Task `OCR - Weekly review`: last week's flagged documents plus a 5% audit sample go to `reviews\`. Reviewers fill it in, then `src\review.py import <csv>`. |
| Sunday 02:00 | Regression check | Task `OCR - Golden check`. A FAIL (exit 1) means accuracy dropped: investigate before trusting new results. |
| Any change | Change control | Model, prompt, code, Ollama or driver update: run `golden_check.py` **before and after**. Only keep the change if it passes. Each run records the model's digest, so results stay traceable. |
| Daily | Backups | SQL Server: full database backup (SQL Agent). Server: back up `archive\` and the app folder. Models can always be downloaded again. |
| Monthly | Housekeeping | Purge archive and runs past retention; check disk space; review `logs\` for repeated errors. |

**What to watch** (all on the dashboard):
- Flag rate per day: a sudden rise means the documents changed or something broke.
- Failed extractions.
- Drift status per metric. "drift" means the incoming documents differ from the baseline, e.g. a new supplier layout or a new scanner.
- Live accuracy from the audit sample, with its confidence interval.
- Golden-check result.

**Alerting**: Task Scheduler records each task's exit code. A non-zero exit from the Worker or Golden check task means attention is needed. Hook this into existing monitoring if you have it, or check `Get-ScheduledTask 'OCR - *' | Get-ScheduledTaskInfo`.

## 7. Capacity (one RTX 3090, measured)

| Document type | Time per document | Per 8-hour day | Per 24 hours |
|---|---|---|---|
| Digital invoice (text model) | ~4.5 s | ~6,400 | ~19,000 |
| Scanned invoice (vision + OCR cross-check) | ~9–11 s | ~2,900 | ~8,600 |
| Receipt (vision + OCR cross-check) | ~2–4 s | ~9,600 | ~29,000 |

Processing is one document at a time. If volume grows past this, the next steps are:
1. Let Ollama serve several requests at once (`OLLAMA_NUM_PARALLEL`) and run 2 workers on separate incoming folders.
2. Add a second GPU.

## 8. Using the server from the development PC (optional)

After `setup_server.ps1 -ExposeOllamaToLan -LanSubnet <subnet>`, point the development PC at the server's GPU:
```powershell
$env:OLLAMA_URL = "http://gpu-server:11434/api/chat"
```
Everything else (scripts, results) stays on the PC.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Silent accuracy drop after an update | Weekly golden check plus a check before and after every change; model digest recorded per run |
| New document types the model wasn't validated on | Drift and outlier monitoring; the audit sample measures real accuracy; rebuild baselines after review |
| Model makes up values (e.g. cut-off totals) | OCR cross-check (`total_not_on_page`); arithmetic checks; review flags |
| Ollama or GPU failure | Worker exits with code 1 and files wait in `incoming\`; Ollama task restarts at boot |
| Sensitive data exposure | Ollama local only; least-privilege service account; restricted folders; retention policy |
| Desktop Ollama app also running and holding the port | Disable the Ollama tray app's auto-start on the server; only the `OCR - Ollama` task runs it |

## 10. Tested on the development PC before handover

Results from testing this folder on the development PC (separate test database `ocr_server_test`, scratch data folder):

- **Worker**:
  - 2 digital invoices were routed to the text model and 2 scans to the vision model; 2 receipts were processed.
  - A non-PDF file was moved to `rejected\`.
  - A re-sent copy of an invoice was flagged `duplicate_file` and `duplicate_invoice`.
  - Every file was archived with a time stamp, and `incoming\` was left empty.
- **Ollama unreachable**: the worker exited with code 1, the waiting file stayed in `incoming\`, and no lock was left behind. The next run processed it.
- **Golden check**: see `docs\golden_check_result.txt`.
