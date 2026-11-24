<#
register_tasks.ps1 - schedule the pipeline with Windows Task Scheduler. Safe to re-run.

Run in an elevated PowerShell from the app folder, AFTER setup_server.ps1 and a passing
golden check:
    .\deploy\register_tasks.ps1 -ServiceAccount "DOMAIN\svc_ocr"

You are asked for the service account's password once (Windows stores it for the tasks).
The account needs: read/write on OCR_HOME, a login on the SQL Server (deploy\sql), and
"Log on as a batch job" rights (granted automatically to administrators; otherwise via
Local Security Policy).

Tasks created (all named "OCR - ..."):
    OCR - Ollama          at start-up       keeps the model server running without anyone logged in
    OCR - Worker          every 5 minutes   processes new PDFs   (worker.py --once)
    OCR - Daily report    06:00 daily       refreshes the dashboard (report.py)
    OCR - Weekly review   Monday 07:00      exports the review queue for last week (review.py)
    OCR - Golden check    Sunday 02:00      re-checks accuracy on the golden set (golden_check.py)
Each task appends its output to OCR_HOME\logs\task_<name>.log.
#>
param(
    [Parameter(Mandatory)] [string]$ServiceAccount,
    [string]$OllamaExe = ""
)
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $App ".venv\Scripts\python.exe"
$Logs = Join-Path ([Environment]::GetEnvironmentVariable("OCR_HOME", "Machine")) "logs"
if (-not (Test-Path $Python)) { throw "No Python environment found; run setup_server.ps1 first." }
if (-not $OllamaExe) { $OllamaExe = (Get-Command ollama -ErrorAction Stop).Source }

$credential = Get-Credential -UserName $ServiceAccount -Message "Password for the account the OCR tasks run as"
$password = $credential.GetNetworkCredential().Password

function Register-OcrTask($name, $trigger, $command, $timeLimitHours = 12) {
    # cmd /c so output can be redirected to a log file
    $log = Join-Path $Logs ("task_" + ($name -replace '\W', '_').ToLower() + ".log")
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $command >> `"$log`" 2>&1" -WorkingDirectory $App
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours $timeLimitHours) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
    Register-ScheduledTask -TaskName "OCR - $name" -Action $action -Trigger $trigger -Settings $settings `
        -User $ServiceAccount -Password $password -RunLevel Limited -Force | Out-Null
    Write-Host "registered: OCR - $name"
}

# Ollama as a background server (the desktop app only runs while someone is logged in).
$ollamaTrigger = New-ScheduledTaskTrigger -AtStartup
Register-OcrTask "Ollama" $ollamaTrigger "`"$OllamaExe`" serve" 0

$every5 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5)
Register-OcrTask "Worker" $every5 "`"$Python`" src\worker.py --once"

Register-OcrTask "Daily report" (New-ScheduledTaskTrigger -Daily -At 6am) "`"$Python`" src\report.py" 1
Register-OcrTask "Weekly review" (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 7am) `
    "`"$Python`" src\review.py export --recent-live 7" 1
Register-OcrTask "Golden check" (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 2am) `
    "`"$Python`" src\golden_check.py" 2

Write-Host "`nStart Ollama now without rebooting:  Start-ScheduledTask 'OCR - Ollama'"
Write-Host "Check status:                        Get-ScheduledTask 'OCR - *' | Get-ScheduledTaskInfo"
