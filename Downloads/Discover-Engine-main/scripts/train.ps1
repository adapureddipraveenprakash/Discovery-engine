# Train user tower + FashionCLIP projection (InfoNCE)
# Usage: .\scripts\train.ps1 [-Epochs 5]

param(
    [int]$Epochs = 5
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python training/train_user_tower.py --epochs $Epochs
