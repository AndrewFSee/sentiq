# Nightly copy of sentiq's data to the Seagate external drive (task \Sentiq\Mirror, installed by
# scripts\install_mirror_task.ps1). Runs at 02:00 and whenever the Seagate is connected; copies
# at most once per night (due = no successful copy since the most recent 02:00).
#
#   powershell -ExecutionPolicy Bypass -File scripts\mirror_to_external.ps1 [-Force] [-WaitSeconds 120]
#
# Copies (robocopy, new and changed files only; files deleted on D: are NOT deleted on the
# Seagate, so an accidental deletion can't wipe the copy):
#   <project>\data       -> <Seagate>\sentiq-mirror\data   (lake, raw documents, DuckDB, refs)
#   D:\sentiq-data       -> <Seagate>\sentiq-data          (10-K / 10-Q raw HTML)
#   D:\sentiq-archive    -> <Seagate>\sentiq-archive
# The drive is found by its volume label, not its letter (letters change). If it isn't
# connected the script exits; the next connection (or the next night) copies.
# Skipped: *.tmp / *.part (files being written) and *.lock. newsfeed.duckdb may be copied while
# the collector writes to it; a torn copy is replaced the next night (the Parquet lake is
# append-only and written by atomic rename, so it copies consistently).
# Status: data\state\mirror-status.json; log: data\mirror.log; robocopy logs: data\state\mirror\.
param(
    [switch]$Force,               # copy even if tonight's copy is already done
    [int]$WaitSeconds = 120       # after a connect event the volume can take a moment to mount
)
$ErrorActionPreference = 'Stop'
$Label = 'Seagate Portable Drive'
$Root = Split-Path -Parent $PSScriptRoot                     # ...\projects\sentiq
$State = Join-Path $Root 'data\state'
$StatusFile = Join-Path $State 'mirror-status.json'
$LogFile = Join-Path $Root 'data\mirror.log'
$RoboDir = Join-Path $State 'mirror'
$Jobs = @(
    @{ Src = (Join-Path $Root 'data'); Dst = 'sentiq-mirror\data' },
    @{ Src = 'D:\sentiq-data'; Dst = 'sentiq-data' },
    @{ Src = 'D:\sentiq-archive'; Dst = 'sentiq-archive' }
)

function Log([string]$msg) {
    $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    Write-Host $line
}

function Read-Status {
    if (Test-Path $StatusFile) {
        try { return Get-Content $StatusFile -Raw | ConvertFrom-Json } catch { }
    }
    return [pscustomobject]@{ last_success = $null; last_attempt = $null; last_result = $null }
}

function Write-Status($status) {
    $tmp = "$StatusFile.tmp"
    $status | ConvertTo-Json | Set-Content -Path $tmp -Encoding utf8
    Move-Item -Force $tmp $StatusFile
}

New-Item -ItemType Directory -Force $RoboDir | Out-Null

# one copy at a time (the task can fire on 02:00 and on a connect within the same minute)
$lockPath = Join-Path $State 'mirror.lock'
try {
    $lock = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
} catch {
    Write-Host 'Another mirror run is active; exiting.'
    exit 0
}

try {
    $status = Read-Status
    $now = Get-Date
    $boundary = $now.Date.AddHours(2)
    if ($now -lt $boundary) { $boundary = $boundary.AddDays(-1) }   # most recent 02:00
    $last = if ($status.last_success) { [datetime]$status.last_success } else { [datetime]::MinValue }
    if (-not $Force -and $last -ge $boundary) {
        Write-Host "Not due: last copy $last is after $boundary."
        exit 0
    }

    $vol = $null
    $deadline = $now.AddSeconds($WaitSeconds)
    while ($true) {
        $vol = Get-Volume -FileSystemLabel $Label -ErrorAction SilentlyContinue |
            Where-Object { $_.DriveLetter } | Select-Object -First 1
        if ($vol -or (Get-Date) -ge $deadline) { break }
        Start-Sleep -Seconds 10
    }
    if (-not $vol) {
        Log "skipped: '$Label' not connected (last copy $last); copies on the next connection or night"
        exit 0
    }
    $drive = "$($vol.DriveLetter):\"
    Log "start: copying to $drive (last copy $last)"
    $status.last_attempt = $now.ToString('s')

    $worst = 0
    $stamp = $now.ToString('yyyyMMdd-HHmmss')
    foreach ($j in $Jobs) {
        if (-not (Test-Path $j.Src)) { Log "  missing source $($j.Src)"; $worst = [math]::Max($worst, 16); continue }
        $dst = Join-Path $drive $j.Dst
        $rlog = Join-Path $RoboDir ("robocopy-$stamp-" + ($j.Dst -replace '[\\:]', '_') + '.log')
        $rargs = @($j.Src, $dst, '/E', '/COPY:DAT', '/DCOPY:T', '/R:2', '/W:5', '/MT:8', '/XJ',
                  '/XF', '*.tmp', '*.part', '*.lock', '/NP', '/NFL', '/NDL', "/LOG:$rlog")
        # the mirror itself writes into data\state\mirror (its logs); copy those next time
        if ($j.Dst -eq 'sentiq-mirror\data') { $rargs += @('/XD', $RoboDir) }
        $t0 = Get-Date
        $p = Start-Process -FilePath robocopy.exe -ArgumentList $rargs -NoNewWindow -PassThru
        $null = $p.Handle            # without a cached handle ExitCode comes back empty
        try { $p.PriorityClass = 'BelowNormal' } catch { }
        $p.WaitForExit()
        $code = $p.ExitCode
        if ($null -eq $code) { $code = 16 }   # unknown result counts as a failure
        $worst = [math]::Max($worst, $code)
        $summary = (Select-String -Path $rlog -Pattern '^\s+(Files|Bytes) :\s+\d' | ForEach-Object { $_.Line.Trim() }) -join ' | '
        Log ("  {0} -> {1}: robocopy {2} in {3:N0}s; {4}" -f $j.Src, $dst, $code, ((Get-Date) - $t0).TotalSeconds, $summary)
    }
    # robocopy exit codes 0-7 = success (bit flags: copied / extra / mismatched), >= 8 = failures
    if ($worst -lt 8) {
        $status.last_success = $now.ToString('s')
        $status.last_result = "ok (robocopy max $worst)"
        Log "done: ok"
    } else {
        $status.last_result = "failed (robocopy max $worst); retried on the next trigger"
        Log "done: FAILED (robocopy max $worst) - see $RoboDir"
    }
    Write-Status $status
    Get-ChildItem $RoboDir -Filter 'robocopy-*.log' | Sort-Object LastWriteTime -Descending |
        Select-Object -Skip 60 | Remove-Item -Force                 # keep ~20 nights of logs
    if ($worst -ge 8) { exit 1 }
} finally {
    $lock.Dispose()
}
