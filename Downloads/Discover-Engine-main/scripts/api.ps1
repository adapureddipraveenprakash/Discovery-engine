# Start FastAPI server on http://localhost:8000
# Usage: .\scripts\api.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -m uvicorn serving.api:app --reload --host 0.0.0.0 --port 8000
