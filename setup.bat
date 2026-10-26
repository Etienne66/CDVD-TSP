@echo off
SETLOCAL EnableDelayedExpansion

:: Name of the virtual environment folder
set "VENV_DIR=.venv"

echo ===================================================
echo             Python Project Setup Script
echo ===================================================

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python from https://python.org before running this script.
    goto :error
)

:: 2. Create Virtual Environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment in .\%VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :error
    )
    echo [SUCCESS] Virtual environment created.
) else (
    echo [INFO] Virtual environment folder .\%VENV_DIR% already exists. Skipping creation.
)

:: 3. Activate Virtual Environment
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    goto :error
)

:: 4. Upgrade pip inside the environment
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: 5. Install dependencies from requirements.txt
if exist "requirements.txt" (
    echo [INFO] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] Dependency installation failed.
        goto :error
    )
    echo [SUCCESS] All packages installed successfully.
) else (
    echo [WARNING] requirements.txt not found. No packages were installed.
)

echo ===================================================
echo [SUCCESS] Environment setup complete!
echo To activate this environment in the future, run:
echo call %VENV_DIR%\Scripts\activate.bat
echo ===================================================
pause
exit /b 0

:error
echo ===================================================
echo [FAILURE] Setup failed. Please check the errors above.
echo ===================================================
pause
exit /b 1
