# One-time setup (run in an ADMIN PowerShell; "run whether logged on or not" tasks need elevation):
#   powershell -ExecutionPolicy Bypass -File <projects>\sentiq\scripts\install_mirror_task.ps1
#
# Registers \Sentiq\Mirror: scripts\mirror_to_external.ps1 as user andre without a stored
# password (S4U), low priority, two triggers:
#   - daily at 02:00 (run as soon as possible if the laptop was off or asleep then);
#   - when the Seagate Portable (serial NT36K7T3) is connected (Partition/Diagnostic event 1006).
# The script copies at most once per night and exits at once if the drive is absent, so a
# missed night is copied the next time the drive is plugged in. Safe to re-run.
$ErrorActionPreference = 'Stop'
$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this from an elevated (Run as administrator) PowerShell.'
}
$Root = Split-Path -Parent $PSScriptRoot                      # ...\projects\sentiq
$Sid = 'S-1-5-21-818495848-3786255412-2037946006-1001'        # andre (same as the other tasks)
$Serial = 'NT36K7T3'                                           # Seagate Portable Drive (5 TB)
$Script = Join-Path $Root 'scripts\mirror_to_external.ps1'
if (-not (Test-Path $Script)) { throw "Not found: $Script" }

$query = "&lt;QueryList&gt;&lt;Query Id=`"0`" Path=`"Microsoft-Windows-Partition/Diagnostic`"&gt;" +
    "&lt;Select Path=`"Microsoft-Windows-Partition/Diagnostic`"&gt;*[System[EventID=1006]] and " +
    "*[EventData[Data[@Name='SerialNumber']='$Serial']]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;"

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>sentiq: copy data\ (lake, raw, DuckDB, refs), D:\sentiq-data and D:\sentiq-archive to the Seagate Portable Drive (scripts\mirror_to_external.ps1). Daily 02:00 and when the drive is connected; at most one copy per night; new and changed files only, never deletes on the Seagate. Log: data\mirror.log</Description>
    <URI>\Sentiq\Mirror</URI>
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
    <ExecutionTimeLimit>PT8H</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <StartWhenAvailable>true</StartWhenAvailable>
    <WakeToRun>false</WakeToRun>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <Priority>7</Priority>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
  </Settings>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-11-27T02:00:00</StartBoundary>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
    <EventTrigger>
      <Subscription>$query</Subscription>
      <Delay>PT1M</Delay>
    </EventTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$Script"</Arguments>
      <WorkingDirectory>$Root</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

Register-ScheduledTask -TaskPath '\Sentiq\' -TaskName 'Mirror' -Xml $xml -Force | Out-Null
$t = Get-ScheduledTask -TaskPath '\Sentiq\' -TaskName 'Mirror'
$i = $t | Get-ScheduledTaskInfo
'{0} {1}, triggers: {2}, next run {3}' -f $t.TaskName, $t.State, $t.Triggers.Count, $i.NextRunTime
Write-Host 'Done. First copy: run it now with  Start-ScheduledTask -TaskPath \Sentiq\ -TaskName Mirror'
Write-Host '(or wait for 02:00). Progress: data\mirror.log'
