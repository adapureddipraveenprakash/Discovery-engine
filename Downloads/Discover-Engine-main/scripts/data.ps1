# Generate synthetic H&M-shaped dataset and build artifacts
# Usage: .\scripts\data.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

python training/prepare_dataset.py
