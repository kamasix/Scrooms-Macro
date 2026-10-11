@echo off
setlocal
cd /d "%~dp0"
if exist "ShroomsMacro-1.0.10.exe" (
    start "" "%~dp0ShroomsMacro-1.0.10.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.9.exe" (
    start "" "%~dp0ShroomsMacro-1.0.9.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.8.exe" (
    start "" "%~dp0ShroomsMacro-1.0.8.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.7.exe" (
    start "" "%~dp0ShroomsMacro-1.0.7.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.6-final.exe" (
    start "" "%~dp0ShroomsMacro-1.0.6-final.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.6.exe" (
    start "" "%~dp0ShroomsMacro-1.0.6.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.5.exe" (
    start "" "%~dp0ShroomsMacro-1.0.5.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.4.exe" (
    start "" "%~dp0ShroomsMacro-1.0.4.exe"
    exit /b
)
if exist "ShroomsMacro-1.0.3.exe" (
    start "" "%~dp0ShroomsMacro-1.0.3.exe"
    exit /b
)
if exist "ShroomsMacro.exe" (
    start "" "%~dp0ShroomsMacro.exe"
    exit /b
)
if exist "dist\ShroomsMacro.exe" (
    start "" "%~dp0dist\ShroomsMacro.exe"
    exit /b
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py"
    exit /b
)
echo Install the source dependencies or build the EXE first. See README.md.
pause
