# Discovery Engine — Windows setup (PowerShell)
# Usage: .\scripts\install.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Installing Python dependencies..."
python -m pip install -r requirements.txt

Write-Host "Done. Next: .\scripts\data.ps1"
