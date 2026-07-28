<#
.SYNOPSIS
  Crash-resilient supervisor for a long training run.

.DESCRIPTION
  This dev box bugchecks under sustained GPU load: WHEA logs a fatal hardware
  error alongside corrected PCI Express AER errors on the NVIDIA link
  (PCI\VEN_10DE&DEV_2DB8), and because VBS is running the machine check surfaces
  as HYPERVISOR_ERROR 0x00020001 rather than the usual 0x124. Two crashes so far,
  identical bugcheck parameters, both within ~3 minutes of a run starting.

  Until that hardware fault is fixed, the only way to get a night of training out
  of the machine is to make progress cheap to resume:

    * polytrain.train_league --auto-resume picks up <out>/latest.pt on restart
    * --save-every bounds what a crash costs to a handful of iterations
    * checkpoints are written atomically, so a crash mid-save cannot tear a file
    * this loop restarts training if the process dies
    * Register-Overnight.ps1 re-launches this loop after a *reboot*

  Exit code 0 from the trainer means the run finished; anything else is treated
  as a crash and retried.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\train_overnight.ps1
#>
[CmdletBinding()]
param(
  [string]$Out       = "checkpoints_big",
  # ~25.5s/iteration measured at 128x10, games=64, rollout=192 on 11x11
  # (7.2s rollout + 18.3s update, 5.5 GB of 8.5 GB peak). 2000 iterations is
  # about 14 hours before eval overhead and crash restarts.
  [int]   $Iters     = 2000,
  [int]   $Channels  = 128,
  [int]   $Blocks    = 10,
  [int]   $Size      = 11,
  [int]   $Games     = 64,
  [int]   $Rollout   = 192,
  [int]   $MaxRetries = 200,
  [string]$Log       = "runs_big.log"
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py" }

# Keep the trainer's own output ASCII-safe; the console here is cp1252.
$env:PYTHONIOENCODING = "utf-8"

$attempt = 0
while ($attempt -lt $MaxRetries) {
  $attempt++
  $banner = "=== attempt $attempt started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
  Write-Output $banner
  Add-Content -Path $Log -Value $banner -Encoding utf8

  & $py -u -m polytrain.train_league `
      --out $Out --auto-resume `
      --size $Size --games $Games --rollout $Rollout `
      --channels $Channels --blocks $Blocks `
      --iters $Iters `
      --league-scripted `
      --snapshot-every 10 --save-every 5 --league-save-every 20 `
      --eval-every 25 --eval-pairs 16 `
      --max-turns 100 2>&1 |
      ForEach-Object { Add-Content -Path $Log -Value $_ -Encoding utf8; $_ }

  $code = $LASTEXITCODE
  if ($code -eq 0) {
    $msg = "=== run finished cleanly after $attempt attempt(s) ==="
    Write-Output $msg
    Add-Content -Path $Log -Value $msg -Encoding utf8
    break
  }

  $msg = "=== trainer exited $code at $(Get-Date -Format 'HH:mm:ss'); resuming in 20s ==="
  Write-Output $msg
  Add-Content -Path $Log -Value $msg -Encoding utf8
  Start-Sleep -Seconds 20
}
