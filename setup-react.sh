#!/bin/bash

# AI Chatbot React Setup Script

echo "🤖 AI Chatbot React Frontend Setup"
echo "=================================="

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    DOCKER_AVAILABLE=true
else
    echo "⚠️  Docker is not installed"
    DOCKER_AVAILABLE=false
fi

# Check if Docker Compose is installed
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is installed"
    COMPOSE_AVAILABLE=true
else
    echo "⚠️  Docker Compose is not installed"
    COMPOSE_AVAILABLE=false
fi

# Check if Node.js is installed
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js is installed: $NODE_VERSION"
    NODE_AVAILABLE=true
else
    echo "⚠️  Node.js is not installed"
    NODE_AVAILABLE=false
fi

echo ""
echo "Setup Options:"
echo "1. 🐳 Docker Development (Recommended)"
echo "2. 🔧 Manual Development Setup"
echo "3. 🚀 Production Docker Setup"
echo "4. 📋 Show Requirements"
echo ""

read -p "Choose an option (1-4): " choice

case $choice in
    1)
        if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
            echo ""
            echo "🐳 Starting Docker Development Environment..."
            echo "This will start both frontend and backend in development mode."
            echo ""
            docker-compose -f docker-compose.dev.yml up --build
        else
            echo "❌ Docker and Docker Compose are required for this option."
            echo "Please install Docker Desktop or Docker + Docker Compose."
        fi
        ;;
    2)
        if [ "$NODE_AVAILABLE" = true ]; then
            echo ""
            echo "🔧 Setting up Manual Development Environment..."
            
            # Install frontend dependencies
            echo "📦 Installing frontend dependencies..."
            cd frontend
            npm install
            
            echo ""
            echo "✅ Setup complete!"
            echo ""
            echo "To start development:"
            echo "1. Start backend: python app.py"
            echo "2. Start frontend: ./start-react.sh"
            echo ""
            echo "Or use the startup script: ./start-react.sh"
        else
            echo "❌ Node.js is required for manual setup."
            echo "Please install Node.js 18+ from https://nodejs.org/"
        fi
        ;;
    3)
        if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
            echo ""
            echo "🚀 Starting Production Docker Environment..."
            echo "This will build and start optimized containers."
            echo ""
            docker-compose up --build -d
            echo ""
            echo "✅ Production environment started!"
            echo "Frontend: http://localhost:3000"
            echo "Backend: http://localhost:5000"
            echo ""
            echo "To stop: docker-compose down"
        else
            echo "❌ Docker and Docker Compose are required for this option."
        fi
        ;;
    4)
        echo ""
        echo "📋 Requirements:"
        echo ""
        echo "For Docker Setup (Recommended):"
        echo "- Docker Desktop or Docker + Docker Compose"
        echo "- 4GB+ RAM available"
        echo "- 2GB+ disk space"
        echo ""
        echo "For Manual Setup:"
        echo "- Node.js 18+"
        echo "- npm or yarn"
        echo "- Python 3.11+"
        echo "- Flask dependencies (see requirements.txt)"
        echo ""
        echo "Installation Links:"
        echo "- Docker Desktop: https://www.docker.com/products/docker-desktop/"
        echo "- Node.js: https://nodejs.org/"
        echo "- Python: https://www.python.org/"
        ;;
    *)
        echo "❌ Invalid option. Please choose 1-4."
        ;;
esac

echo ""
echo "📚 For more information, see README-REACT.md"