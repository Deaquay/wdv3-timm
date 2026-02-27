@echo off
cd /d "%~dp0"

where uv.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo Using uv

    if not exist .venv (
        echo Creating virtual environment (Python 3.13^)...
        uv venv -p 3.13
    )

    call .venv\Scripts\activate

    echo Installing torch...
    uv pip install torch torchvision --torch-backend auto

    echo Installing requirements...
    uv pip install -r requirements.txt
) else (
    echo uv not found, falling back to pip
    echo   (recommended: install uv first with install-uv.bat^)

    if not exist .venv (
        echo Creating virtual environment...
        python -m venv .venv
    )

    call .venv\Scripts\activate
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
)

echo.
echo Setup complete. Virtual environment is active.
echo You can deactivate it by running: deactivate
pause
exit /b 0
