param(
    [Parameter(Mandatory = $true)]
    [string]$TeamName,

    [Parameter(Mandatory = $true)]
    [string]$ProjectName
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$submissionDir = Join-Path $root "submission"
$stageDir = Join-Path $submissionDir "package"
$otherLabel = "$([char]0x5176)$([char]0x4ED6)"
$zipName = "${TeamName}_${ProjectName}_${otherLabel}.zip"
$zipPath = Join-Path $submissionDir $zipName
$limitBytes = 200MB

if (Test-Path -LiteralPath $stageDir) {
    Remove-Item -LiteralPath $stageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

$directories = @("src", "frontend", "tests", "scripts")
foreach ($directory in $directories) {
    Copy-Item -LiteralPath (Join-Path $root $directory) -Destination $stageDir -Recurse
}

$files = @("README.md", "pyproject.toml", ".env.example", ".gitignore")
foreach ($file in $files) {
    Copy-Item -LiteralPath (Join-Path $root $file) -Destination $stageDir
}

$compactSource = Join-Path $root "outputs\reranker_compact\compact_reranker.json"
if (Test-Path -LiteralPath $compactSource) {
    $compactTarget = Join-Path $stageDir "outputs\reranker_compact"
    New-Item -ItemType Directory -Force -Path $compactTarget | Out-Null
    Copy-Item -LiteralPath $compactSource -Destination $compactTarget
}

Get-ChildItem -LiteralPath $stageDir -Recurse -Directory -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") -or $_.Name -like "*.egg-info" } |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $stageDir -Recurse -File -Force -Include "*.pyc", "*.pyo" |
    Remove-Item -Force

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath -CompressionLevel Optimal

$size = (Get-Item -LiteralPath $zipPath).Length
if ($size -gt $limitBytes) {
    throw "Submission archive is $([math]::Round($size / 1MB, 2)) MB, exceeding the 200 MB limit."
}

Write-Host "Created: $zipPath"
Write-Host "Size: $([math]::Round($size / 1MB, 2)) MB"
Write-Host "Excluded: .env, datasets, full model weights, Hugging Face cache, training pairs, evaluation outputs"
