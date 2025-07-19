@echo off
echo 🚀 Starting Arabic Phonology Engine...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found!

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies!
    pause
    exit /b 1
)

echo ✅ Dependencies installed!

REM Start the server
echo 🌐 Starting web server...
echo.
echo 📍 The application will be available at: http://localhost:5000
echo 🔄 Press Ctrl+C to stop the server
echo.

python app.py

pause
