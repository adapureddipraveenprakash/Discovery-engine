# Run unit tests
# Usage: .\scripts\test.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -m pytest tests/ -v
