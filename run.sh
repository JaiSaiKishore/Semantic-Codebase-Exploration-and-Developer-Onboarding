#!/bin/bash
# ==============================================================================
# Semantic Codebase RAG - Linux / macOS Startup Script
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo "  Semantic Codebase RAG - Developer Onboarding Assistant"
echo "============================================================"

if [ ! -f ".env" ]; then
    echo "[ERROR] .env file not found!"
    echo "Please copy .env.example to .env and configure your GROQ_API_KEY:"
    echo "  cp .env.example .env"
    exit 1
fi

if [ -f "venv/bin/python" ]; then
    PYTHON_EXE="venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
else
    PYTHON_EXE="python3"
fi

echo "[1/2] Launching browser at http://localhost:8000 ..."
(sleep 2 && (xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null)) &

echo "[2/2] Starting FastAPI + LangGraph + ChromaDB Server..."
echo "(Press Ctrl+C to stop the application)"
echo ""
exec "$PYTHON_EXE" server.py

