<#
.SYNOPSIS
  Register (or remove) a logon task that resumes training after a reboot.

.DESCRIPTION
  train_overnight.ps1 survives the *trainer* dying. It cannot survive the machine
  bugchecking, because the supervisor dies with it. Windows reboots itself after
  a bugcheck by default, so a logon-triggered scheduled task closes the loop:

      crash -> reboot -> logon -> task -> supervisor -> --auto-resume

  The task runs as the current user, so it needs no elevation and cannot outlive
  your account. Remove it with -Remove as soon as the run is done -- otherwise it
  will start training every time you log in, which is rarely what you want.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\Register-Overnight.ps1
  powershell -ExecutionPolicy Bypass -File scripts\Register-Overnight.ps1 -Remove
#>
[CmdletBinding()]
param(
  [string]$TaskName = "PolyZeroOvernight",
  [switch]$Remove
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if ($Remove) {
  try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "removed scheduled task '$TaskName'"
  } catch {
    Write-Output "no scheduled task '$TaskName' to remove"
  }
  return
}

$script = Join-Path $repo "scripts\train_overnight.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
  -WorkingDirectory $repo

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# A training run is not "idle work": it must not be throttled, stopped when the
# task runs long, or killed when the machine goes on battery.
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
  -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
  -Settings $settings -Description "Resume PolyZero training after a crash/reboot." `
  -Force | Out-Null

Write-Output "registered '$TaskName' (runs at logon as $env:USERNAME)"
Write-Output "remove it when the run is done:"
Write-Output "  powershell -ExecutionPolicy Bypass -File scripts\Register-Overnight.ps1 -Remove"
