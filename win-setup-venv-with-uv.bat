@echo off

set "cwd=%~dp0"

echo This will install Python v3.10.6 and its requirements to "%cwd%". Press any key to continue...

pause

:: 0. Change active directory to where the bat file is located
cd /d "%cwd%"

echo ------------
echo  		Installing Python and creating venv
echo ------------

:: 2. Create virtual environment
uv venv --seed --python 3.10.6

echo ------------
echo 		Activating venv
echo ------------

:: 3. Activate the virtual environment
call .venv\Scripts\activate

:: 4. Install torch based on user choice
echo ------------
echo  		Installing torch
echo ------------
echo Select an option:
echo 1. CPU
echo 2. Cuda v11.8
echo 3. Cuda v12.4
echo 4. Cuda v12.6
echo.

choice /c 1234 /m "Enter your choice (1, 2, 3, or 4): "

if errorlevel 4 (
    echo You chose Cuda v12.6
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
) else if errorlevel 3 (    
    echo You chose Cuda v12.4
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
) else if errorlevel 2 (
    echo You chose Cuda v11.8
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if errorlevel 1 (
    echo You chose CPU
    uv pip install torch torchvision torchaudio
)
echo ------------
echo 		Installing requirements.
echo ------------
:: 5. Install requirements
if exist requirements.txt (
    echo Installing requirements from requirements.txt...
    uv pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping...
)

:: 6. Finished
echo.
echo Installation completed successfully!
echo Virtual environment is now active.
echo You can deactivate it by running: deactivate
pause

exit /b 0