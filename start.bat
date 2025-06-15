@echo off
title AI Chatbot Web GUI

echo 🤖 AI Chatbot Web GUI
echo =====================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "chatbot_env" (
    echo 📦 Creating virtual environment...
    python -m venv chatbot_env
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call chatbot_env\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

echo.
echo 📦 Dependencies installed

REM Check if Ollama is installed
where ollama >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama not found. Please install Ollama manually from https://ollama.ai/
    echo    Or continue without Ollama if using a different AI service.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    REM Check if Ollama is running
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
    if errorlevel 1 (
        echo 🚀 Starting Ollama server...
        start /B ollama serve
        timeout /t 3 /nobreak >nul
        
        REM Check if default model exists
        ollama list | find "llama2" >nul
        if errorlevel 1 (
            echo 📥 Downloading default model (llama2)...
            echo    This may take a few minutes...
            ollama pull llama2
        )
    ) else (
        echo ✅ Ollama server is already running
    )
)

echo.
echo ✅ Setup complete!
echo 📍 Project location: %cd%
echo.
echo 🌐 Starting web application...
echo    Server will be available at: http://localhost:5000
echo    Press Ctrl+C to stop
echo.

REM Start the Flask application
python app.py

echo.
echo 👋 Goodbye!
pause