@echo off
REM Silent Arabic Morphophonological Platform Launcher
REM Runs the platform with zero output for production use

cd /d "c:\Users\Administrator\new engine"
python -W ignore run_silent.py > nul 2>&1
