# Verify item/user embedding pipelines (Step 4 checkpoint)
# Usage: .\scripts\verify.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python scripts/verify_embeddings.py
