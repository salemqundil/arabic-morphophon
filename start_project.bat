@echo off
title Arabic NLP Project - Safe Environment
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1

echo 🚀 Starting Arabic NLP Project Environment
echo ============================================

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ⚠️  Virtual environment not found
    echo Creating new virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install --upgrade pip
    echo ✅ New virtual environment created
)

if exist setup_environment.py (
    python setup_environment.py
)

cd safe_workspace 2>nul || (
    echo 📁 Creating safe workspace...
    mkdir safe_workspace
    cd safe_workspace
)

echo 🎉 Ready for development!
echo Current directory: %CD%
echo Python version:
python --version

cmd /k
