@echo off
setlocal
cd /d "%~dp0"
if exist "dist\ShroomsMacro.exe" (
    start "" "dist\ShroomsMacro.exe"
    exit /b
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "main.py"
    exit /b
)
echo Install the source dependencies or build the EXE first. See README.md.
pause
