@echo off
echo Installing uv...
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
echo.
echo uv installed. You may need to restart your terminal.
pause
exit /b 0
