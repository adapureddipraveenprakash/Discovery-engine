# Full offline pipeline: install -> data -> train -> index
# Usage: .\scripts\pipeline.ps1 [-SkipInstall]

param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $SkipInstall) {
    & "$PSScriptRoot\install.ps1"
}
& "$PSScriptRoot\data.ps1"
& "$PSScriptRoot\train.ps1"
& "$PSScriptRoot\index.ps1"

Write-Host ""
Write-Host "Pipeline complete. Start API with: .\scripts\api.ps1"
