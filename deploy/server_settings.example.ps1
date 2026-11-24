# server_settings.example.ps1 - machine-wide settings for the GPU server.
#
# Copy to server_settings.ps1, fill in the values, then run it ONCE in an elevated
# (Administrator) PowerShell:   .\deploy\server_settings.ps1
# It stores the settings as Windows machine environment variables, which every
# scheduled task and new terminal then sees. Re-run it after changing a value.
#
# server_settings.ps1 can hold a password (if you use a SQL login): keep it out of any
# shared folder or version control. Prefer Windows authentication (no password at all).

$settings = @{
    # Where working data lives (incoming, archive, runs, reports, reviews, logs).
    # Put it on a disk with space for the archived PDFs; it does not have to be the app folder.
    OCR_HOME           = "D:\ocr-data"

    # Results database on the network SQL Server. Windows authentication: the scheduled
    # tasks run as the service account, which needs a login on the SQL Server.
    OCR_DB_URL         = "mssql+pyodbc://@SQLSERVER01/ocr_pipeline?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes"

    # Ollama runs on this same server; no need to change unless it moves.
    OLLAMA_URL         = "http://127.0.0.1:11434/api/chat"

    # Keep models in GPU memory between documents (avoids a ~60 s reload).
    OLLAMA_KEEP_ALIVE  = "24h"
    OLLAMA_MAX_LOADED_MODELS = "2"

    # Models (validated in development; change only after a passing golden check).
    OCR_TEXT_MODEL     = "qwen2.5:7b"
    OCR_VISION_MODEL   = "qwen2.5vl:7b"
    OCR_PROMPT         = "v2"

    # Monitoring
    OCR_AUDIT_RATE     = "0.05"          # share of unflagged documents sent for review
    OCR_CURRENCIES     = "USD,GBP,EUR"   # currencies you expect on invoices
}

foreach ($name in $settings.Keys) {
    [Environment]::SetEnvironmentVariable($name, $settings[$name], "Machine")
    Write-Host "$name = $($settings[$name])"
}
Write-Host "`nSaved as machine environment variables. Open a new terminal (and restart Ollama) to pick them up."
