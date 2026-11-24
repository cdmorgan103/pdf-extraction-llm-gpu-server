<#
setup_server.ps1 - install and configure the GPU server. Safe to re-run.

Run in an elevated (Administrator) PowerShell from the app folder:
    Set-ExecutionPolicy -Scope Process Bypass
    .\deploy\setup_server.ps1

Before running: copy deploy\server_settings.example.ps1 to deploy\server_settings.ps1
and fill it in (data folder, SQL Server name).

What it does:
  1. Installs Ollama, Python 3.11, ODBC Driver 18 for SQL Server and sqlcmd (winget)
  2. Applies deploy\server_settings.ps1 (machine environment variables)
  3. Optionally opens Ollama to the LAN (-ExposeOllamaToLan), firewalled to one subnet
  4. Downloads the two models
  5. Creates the Python environment (.venv) with the pinned packages
  6. Creates the working folders
  7. Checks the GPU, Ollama and the database connection

Afterwards: run the golden check, then deploy\register_tasks.ps1 (see docs\SERVER_PLAN.md).
#>
param(
    # Let other machines on the LAN use this server's Ollama (e.g. a developer PC).
    # Off by default: the pipeline runs on this server and only needs Ollama locally.
    [switch]$ExposeOllamaToLan,
    # The only network allowed to reach Ollama when exposed, e.g. 192.168.10.0/24.
    [string]$LanSubnet = "",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $PSScriptRoot   # the app folder (parent of deploy\)

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this script in an elevated (Administrator) PowerShell." }

# 1. Software ---------------------------------------------------------------
if (-not $SkipInstall) {
    Step "Installing software (winget)"
    foreach ($id in "Ollama.Ollama", "Python.Python.3.11", "Microsoft.msodbcsql.18", "Microsoft.Sqlcmd") {
        winget list --id $id -e --accept-source-agreements | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host "$id already installed"; continue }
        winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements
    }
    # Make the newly installed programs visible in this session.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

# 2. Settings ---------------------------------------------------------------
Step "Applying settings"
$settingsFile = Join-Path $PSScriptRoot "server_settings.ps1"
if (-not (Test-Path $settingsFile)) {
    throw "Missing $settingsFile. Copy server_settings.example.ps1 to server_settings.ps1 and fill it in."
}
& $settingsFile
foreach ($name in "OCR_HOME", "OCR_DB_URL", "OLLAMA_URL", "OLLAMA_KEEP_ALIVE", "OLLAMA_MAX_LOADED_MODELS") {
    Set-Item "env:$name" ([Environment]::GetEnvironmentVariable($name, "Machine"))
}

# 3. Network access to Ollama (optional) ------------------------------------
Step "Ollama network access"
if ($ExposeOllamaToLan) {
    if (-not $LanSubnet) { throw "-ExposeOllamaToLan needs -LanSubnet, e.g. 192.168.10.0/24" }
    # Ollama has no login of its own: anyone who can reach port 11434 can use it.
    # So listen on all interfaces but let the firewall admit only the given subnet.
    [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "Machine")
    Get-NetFirewallRule -DisplayName "OCR - Ollama (LAN)" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    New-NetFirewallRule -DisplayName "OCR - Ollama (LAN)" -Direction Inbound -Protocol TCP -LocalPort 11434 `
        -RemoteAddress $LanSubnet -Action Allow -Profile Domain,Private | Out-Null
    Write-Host "Ollama will accept connections from $LanSubnet only."
} else {
    [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "127.0.0.1:11434", "Machine")
    Write-Host "Ollama listens on this server only (default)."
}

# 4. Models -----------------------------------------------------------------
Step "Downloading models (about 11 GB the first time)"
$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) { $ollama = (Get-Command ollama).Source }
try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 5 | Out-Null }
catch { Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep 5 }
foreach ($model in [Environment]::GetEnvironmentVariable("OCR_TEXT_MODEL", "Machine"),
                   [Environment]::GetEnvironmentVariable("OCR_VISION_MODEL", "Machine")) {
    & $ollama pull $model
}

# 5. Python environment -----------------------------------------------------
Step "Python environment"
Push-Location $App
if (-not (Test-Path ".venv\Scripts\python.exe")) { py -3.11 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
Pop-Location

# 6. Working folders --------------------------------------------------------
Step "Working folders"
$home_ = [Environment]::GetEnvironmentVariable("OCR_HOME", "Machine")
foreach ($sub in "incoming\invoice", "incoming\receipt", "archive", "runs", "reports", "reviews", "logs") {
    New-Item -ItemType Directory -Force (Join-Path $home_ $sub) | Out-Null
}
Write-Host "Drop PDFs into $home_\incoming\invoice or $home_\incoming\receipt"

# 7. Checks -----------------------------------------------------------------
Step "Checks"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
Invoke-RestMethod http://127.0.0.1:11434/api/tags | Select-Object -ExpandProperty models | Select-Object name, size
Push-Location $App
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import db; e=db.get_engine(); db.create_schema(e); print('database OK:', e.url.host, e.url.database)"
Pop-Location

Write-Host "`nSetup complete. Next: run the golden check, then deploy\register_tasks.ps1 (docs\SERVER_PLAN.md)." -ForegroundColor Green
