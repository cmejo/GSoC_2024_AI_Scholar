#!/bin/bash

# AI Chatbot Web GUI - GitHub Setup Script
# This script helps you set up your GitHub repository

echo "🤖 AI Chatbot Web GUI - GitHub Setup"
echo "===================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    echo "   Visit: https://git-scm.com/downloads"
    exit 1
fi

# Check if we're in a git repository
if [ -d ".git" ]; then
    echo "📁 Git repository already exists"
    echo "   Current status:"
    git status --short
    echo ""
    read -p "Do you want to add files to existing repository? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "👋 Setup cancelled"
        exit 0
    fi
else
    echo "📁 Initializing new Git repository..."
    git init
    echo "✅ Git repository initialized"
    echo ""
fi

# Add all files
echo "📦 Adding files to Git..."
git add .

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short
echo ""

# Get commit message
echo "💬 Creating commit..."
read -p "Enter commit message (or press Enter for default): " commit_message

if [ -z "$commit_message" ]; then
    commit_message="Add AI Chatbot Web GUI

- Responsive, mobile-ready web interface
- Flask backend with WebSocket support
- Progressive Web App (PWA) capabilities
- Dark/light theme support
- Real-time chat with typing indicators
- Offline functionality with service worker
- Complete documentation and deployment guides"
fi

# Create commit
git commit -m "$commit_message"
echo "✅ Commit created"
echo ""

# Check if remote exists
if git remote get-url origin &> /dev/null; then
    echo "🔗 Remote 'origin' already exists:"
    git remote get-url origin
    echo ""
    read -p "Do you want to push to existing remote? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Pushing to GitHub..."
        git push origin main 2>/dev/null || git push origin master 2>/dev/null || git push -u origin main
        echo "✅ Pushed to GitHub!"
    fi
else
    echo "🔗 No remote repository configured"
    echo ""
    echo "To connect to GitHub:"
    echo "1. Create a new repository on GitHub.com"
    echo "2. Copy the repository URL"
    echo "3. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    
    read -p "Do you have a GitHub repository URL? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your GitHub repository URL: " repo_url
        if [ ! -z "$repo_url" ]; then
            echo "🔗 Adding remote repository..."
            git remote add origin "$repo_url"
            echo "🚀 Pushing to GitHub..."
            git branch -M main
            git push -u origin main
            echo "✅ Repository set up and pushed to GitHub!"
        fi
    fi
fi

echo ""
echo "🎉 GitHub setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Visit your repository on GitHub"
echo "2. Add a description: 'Responsive, mobile-ready web GUI for AI chatbot services'"
echo "3. Add topics: ai-chatbot, web-gui, flask, responsive-design, mobile-first, pwa"
echo "4. Enable Issues and Discussions"
echo "5. Create your first release (v1.0.0)"
echo ""
echo "📚 For detailed instructions, see: GITHUB_SETUP.md"
echo ""
echo "🚀 To test your chatbot:"
echo "   ./start.sh (Linux/Mac) or start.bat (Windows)"
echo ""
echo "👋 Happy coding!"