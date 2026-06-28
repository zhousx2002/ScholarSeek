param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 5174
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "E:\miniconda3\envs\DP_learn\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $PythonExe)) {
  throw "Python environment not found: $PythonExe"
}

$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "src"

Write-Host "ScholarSeek Agent"
Write-Host "Project: $ProjectRoot"
Write-Host "URL:     http://$HostName`:$Port"
Write-Host "Press Ctrl+C to stop."

& $PythonExe -m scholarseek.web_server --host $HostName --port $Port
