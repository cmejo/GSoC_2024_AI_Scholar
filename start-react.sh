#!/bin/bash

# AI Chatbot React Frontend Startup Script

echo "🚀 Starting AI Chatbot React Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Navigate to frontend directory
cd frontend

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Please run this script from the project root."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if backend is running
echo "🔍 Checking if backend is running..."
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend is not running. Please start the backend first:"
    echo "   python app.py"
    echo ""
    echo "Continuing anyway..."
fi

# Start the development server
echo "🎯 Starting React development server..."
echo "📍 Frontend will be available at: http://localhost:3000"
echo "📍 Backend should be running at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm start