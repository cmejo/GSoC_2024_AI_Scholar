#!/bin/bash

# AI Chatbot Web GUI Startup Script

echo "🤖 AI Chatbot Web GUI"
echo "====================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "chatbot_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv chatbot_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source chatbot_env/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "📦 Dependencies installed"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install ollama
    else
        echo "❌ Homebrew not found. Please install Ollama manually from https://ollama.ai/"
        echo "   Or continue without Ollama if using a different AI service."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo ""
echo "✅ Setup complete!"
echo "📍 Project location: $(pwd)"
echo ""

# Check if Ollama is running
if command -v ollama &> /dev/null; then
    if ! pgrep -f "ollama serve" > /dev/null; then
        echo "🚀 Starting Ollama server..."
        ollama serve &
        OLLAMA_PID=$!
        sleep 3
        
        # Check if default model exists
        if ! ollama list | grep -q "llama2"; then
            echo "📥 Downloading default model (llama2)..."
            echo "   This may take a few minutes..."
            ollama pull llama2
        fi
    else
        echo "✅ Ollama server is already running"
    fi
fi

echo ""
echo "🌐 Starting web application..."
echo "   Server will be available at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

# Start the Flask application
python app.py

# Cleanup
if [ ! -z "$OLLAMA_PID" ]; then
    echo ""
    echo "🛑 Stopping Ollama server..."
    kill $OLLAMA_PID 2>/dev/null
fi

echo "👋 Goodbye!"