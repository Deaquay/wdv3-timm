@echo off

set "cwd=%~dp0"

echo This will install UV to handle venv and Python versions to "%cwd%". Press any key to continue...

pause

:: 0. Change active directory to where the bat file is located
cd /d "%~dp0"

echo ------------
echo  		Installing UV
echo ------------

:: 1. Check if UV is installed to path, else install it
where uv.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo UV not found in PATH. Installing UV...
    powershell -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR = '%~dp0UV'; iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
    
    :: Wait for installation to complete
    :wait_for_uv
    where uv.exe >nul 2>&1
    if %errorlevel% neq 0 (
        timeout /t 1 >nul
        goto wait_for_uv
    )
    echo UV installation completed.
) else (
	echo UV is already installed.
)

pause

exit /b 0
