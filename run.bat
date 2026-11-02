@echo off
title Semantic Codebase RAG - Local Desktop Server
cd /d "%~dp0"

echo ============================================================
echo   Semantic Codebase RAG - Developer Onboarding Assistant
echo ============================================================

if not exist ".env" (
    echo [ERROR] .env file not found! Please create .env with GROQ_API_KEY.
    pause
    exit /b 1
)

if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
) else (
    set PYTHON_EXE=python
)

echo [1/2] Launching browser at http://localhost:8000 ...
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

echo [2/2] Starting FastAPI + LangGraph + ChromaDB Server...
echo (Press Ctrl+C in this window to stop the application)
echo.
"%PYTHON_EXE%" server.py
pause