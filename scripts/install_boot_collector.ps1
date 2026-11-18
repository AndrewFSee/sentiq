# One-time setup (run in an ADMIN PowerShell; registering "run whether logged on or not" tasks
# needs elevation):
#   powershell -ExecutionPolicy Bypass -File C:\Users\andre\projects\sentiq\scripts\install_boot_collector.ps1
#
# 1. \Sentiq\Collector: scripts\collect_service.py --mode boot at system start (1 min delay),
#    as user andre without a stored password (S4U), normal priority, no time limit, restarted on
#    failure. A Windows Update restart then costs minutes of collection, not the hours until the
#    next login (2026-11-11: 7.3 h).
# 2. \MarketCompass\SentiqHourly re-registered the same way (it was "only when logged on").
# 3. Windows Update: active hours 04:00-22:00 (restarts only 22:00-04:00), automatic active
#    hours off, "notify me when a restart is required" on.
# 4. Switch-over: stops the old collect_window.cmd loop (waits for a running cycle to finish),
#    installs the new scripts\collect_window.cmd (single-instance lock + standby), starts the
#    boot collector now, and reopens the window (it stands by while the boot collector runs).
# Safe to re-run.
$ErrorActionPreference = 'Stop'
$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this from an elevated (Run as administrator) PowerShell.'
}
$Root = Split-Path -Parent $PSScriptRoot                      # ...\projects\sentiq
$Py = 'C:\Users\andre\AppData\Roaming\uv\python\cpython-3.11.12-windows-x86_64-none\python.exe'
$Sid = 'S-1-5-21-818495848-3786255412-2037946006-1001'        # andre (same as the hub's tasks)
$Hub = 'C:\Users\andre\projects\market-compass'
if (-not (Test-Path $Py)) { throw "Base Python not found: $Py" }

$collectorXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>sentiq live collector, started at boot before login (scripts\collect_service.py --mode boot). Holds data\state\collector.lock; the console window (scripts\collect_window.cmd) stands by while this runs. Logs: data\collect.log, data\collector-service.log</Description>
    <URI>\Sentiq\Collector</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>$Sid</UserId>
      <LogonType>S4U</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <Priority>4</Priority>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
  </Settings>
  <Triggers>
    <BootTrigger>
      <Delay>PT1M</Delay>
    </BootTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>$Py</Command>
      <Arguments>scripts\collect_service.py --mode boot</Arguments>
      <WorkingDirectory>$Root</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

$hourlyXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>market-compass: hourly SentIQ "Filings &amp; Calls" refresh on weekdays 04:35-19:35 ET, pre-market through after-hours (run_jobs.py --only sentiq). Skips if another refresh holds data\jobs.lock. Log: data\logs\jobs.log</Description>
    <URI>\MarketCompass\SentiqHourly</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>$Sid</UserId>
      <LogonType>S4U</LogonType>
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT20M</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <WakeToRun>false</WakeToRun>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
  </Settings>
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>PT1H</Interval>
        <Duration>PT15H5M</Duration>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2026-11-18T04:35:00</StartBoundary>
      <ScheduleByWeek>
        <WeeksInterval>1</WeeksInterval>
        <DaysOfWeek><Monday /><Tuesday /><Wednesday /><Thursday /><Friday /></DaysOfWeek>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>$Hub\.venv\Scripts\pythonw.exe</Command>
      <Arguments>run_jobs.py --only sentiq</Arguments>
      <WorkingDirectory>$Hub</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

Write-Host '1/4 Registering \Sentiq\Collector (boot) and \MarketCompass\SentiqHourly (S4U)...'
Register-ScheduledTask -TaskPath '\Sentiq\' -TaskName 'Collector' -Xml $collectorXml -Force | Out-Null
Register-ScheduledTask -TaskPath '\MarketCompass\' -TaskName 'SentiqHourly' -Xml $hourlyXml -Force | Out-Null

Write-Host '2/4 Windows Update: active hours 04-22, notify before restarts...'
$wu = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
if (-not (Test-Path $wu)) { New-Item -Path $wu -Force | Out-Null }
Set-ItemProperty $wu -Name SmartActiveHoursState -Value 0 -Type DWord
Set-ItemProperty $wu -Name ActiveHoursStart -Value 4 -Type DWord
Set-ItemProperty $wu -Name ActiveHoursEnd -Value 22 -Type DWord
Set-ItemProperty $wu -Name RestartNotificationsAllowed2 -Value 1 -Type DWord

Write-Host '3/4 Stopping the old collector window loop...'
$old = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
    Where-Object { $_.CommandLine -match 'collect_window\.cmd' }
foreach ($p in $old) { Stop-Process -Id $p.ProcessId -Force; Write-Host "   stopped cmd.exe pid $($p.ProcessId)" }
$deadline = (Get-Date).AddMinutes(30)
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match 'nf\.py.+collect --once' }) {
    if ((Get-Date) -gt $deadline) { throw 'A collect cycle is still running after 30 min; re-run later.' }
    Write-Host '   waiting for the running collect cycle to finish...'
    Start-Sleep -Seconds 10
}
$next = Join-Path $Root 'scripts\collect_window.cmd.next'
if (Test-Path $next) {
    Move-Item -Force $next (Join-Path $Root 'scripts\collect_window.cmd')
    Write-Host '   installed the new scripts\collect_window.cmd'
}

Write-Host '4/4 Starting the boot collector and reopening the window (standby)...'
Start-ScheduledTask -TaskPath '\Sentiq\' -TaskName 'Collector'
Start-Sleep -Seconds 5
# explorer.exe hands the file to the (non-elevated) desktop shell, so the window isn't admin
Start-Process explorer.exe -ArgumentList (Join-Path $Root 'scripts\collect_window.cmd')

Get-ScheduledTask -TaskPath '\Sentiq\', '\MarketCompass\' | Where-Object TaskName -in 'Collector', 'SentiqHourly' |
    ForEach-Object { $i = $_ | Get-ScheduledTaskInfo; '{0,-14} {1,-8} next {2}' -f $_.TaskName, $_.State, $i.NextRunTime }
Write-Host 'Done. The new window should say "STANDBY, boot collector active" after its first check.'
