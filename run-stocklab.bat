@echo off
setlocal

call C:\Users\81229\miniconda3\condabin\conda.bat activate stocklab

set "REQUESTED_PORT=%~1"
if "%REQUESTED_PORT%"=="" set "REQUESTED_PORT=8501"

for /f %%p in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$start=[int]'%REQUESTED_PORT%'; $chosen=$null; for($p=$start; $p -lt ($start+30); $p++){ try { $listener=[System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$p); $listener.Start(); $listener.Stop(); $chosen=$p; break } catch {} }; if($chosen){ $chosen } else { $start }"') do set "PORT=%%p"

echo StockLab will use http://127.0.0.1:%PORT%
python -m streamlit run S:\work\personal\gpt\app.py --global.developmentMode false --server.address 127.0.0.1 --server.port %PORT% --server.headless true

endlocal
