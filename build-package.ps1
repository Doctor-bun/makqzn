$ErrorActionPreference = "Stop"

$envRoot = "C:\Users\81229\miniconda3\envs\stocklab"
$pythonExe = Join-Path $envRoot "python.exe"

& $pythonExe -m pip install pyinstaller | Out-Host

$distDir = Join-Path $PSScriptRoot "dist"
$buildDir = Join-Path $PSScriptRoot "build"

if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }

& $pythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --name StockLab `
    --onedir `
    --add-data "app.py;." `
    --add-data "analysis_engine.py;." `
    --add-data "market_overview.py;." `
    --add-data "local_store.py;." `
    --add-data ".streamlit;.streamlit" `
    --hidden-import streamlit `
    --hidden-import streamlit.web.cli `
    --hidden-import akshare `
    --hidden-import charset_normalizer `
    --hidden-import chardet `
    --collect-all akshare `
    --collect-all charset_normalizer `
    --collect-all chardet `
    --collect-all yfinance `
    --collect-all streamlit `
    desktop_launcher.py | Out-Host

$zipPath = Join-Path $distDir "StockLab-portable.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $distDir "StockLab\*") -DestinationPath $zipPath -Force

Write-Host "Package created in $distDir\\StockLab"
