@echo off
REM ============================================================
REM  Rocket League AI - Classical Search Algorithms
REM  Single-click launcher (requires Python 3.11)
REM ============================================================

title medo dyaa - Rocket League Search AI

cd /d "%~dp0"

set "PY311=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"

REM -- Create venv with Python 3.11 if it doesn't exist --------
if not exist ".venv\Scripts\python.exe" (
    echo [ENV] Creating virtual environment with Python 3.11...
    "%PY311%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [ENV] Installing dependencies...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM -- Show Python version -------------------------------------
python --version

REM -- Install dependencies if needed --------------------------
echo [DEPS] Checking dependencies...
python -c "import rlbot" >nul 2>&1
if errorlevel 1 (
    echo [DEPS] Installing required packages...
    python -m pip install -r requirements.txt
)

REM -- Launch the AI system ------------------------------------
echo.
echo ============================================================
echo.
echo   ##   ##  #####  ####    ####     ####   ##  ##    #####     #####
echo   ### ###  ##     ## ##  ##  ##    ## ##  ##   ## ##     ## ##     ##
echo   ## # ##  ####   ##  ## ##  ##    ##  ##  ###### ######### #########
echo   ##   ##  ##     ## ##  ##  ##    ## ##    ####  ##     ## ##     ##
echo   ##   ##  #####  ####    ####     ####      #    ##     ## ##     ##
echo.
echo        Rocket League AI - Classical Search Algorithms
echo   A* ^| BFS ^| UCS ^| Greedy Best First ^| DFS ^| Decision Tree
echo ============================================================
echo.

REM -- Launch the AI Dashboard (main-thread GUI, no startup menu) ------
python runtime\launcher.py

echo.
echo [DONE] Rocket League Search AI has exited.
pause

