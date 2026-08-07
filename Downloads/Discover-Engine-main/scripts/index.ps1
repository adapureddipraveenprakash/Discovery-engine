# Build FAISS IndexFlatIP from trained item embeddings
# Usage: .\scripts\index.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

python scripts/build_index.py
