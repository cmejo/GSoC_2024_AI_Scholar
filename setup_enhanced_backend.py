#!/usr/bin/env python3
"""
Enhanced Backend Setup Script
Sets up the AI Chatbot backend with Ollama, Hugging Face integration, and model management
"""

import os
import sys
import subprocess
import requests
import time
import json
from pathlib import Path

def print_step(step, message):
    """Print a formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {message}")
    print('='*60)

def check_command_exists(command):
    """Check if a command exists in PATH"""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ollama():
    """Install Ollama if not present"""
    print_step(1, "Installing Ollama")
    
    if check_command_exists('ollama'):
        print("✅ Ollama is already installed")
        return True
    
    print("📥 Installing Ollama...")
    
    # Detect OS and install accordingly
    import platform
    system = platform.system().lower()
    
    if system == 'linux' or system == 'darwin':  # Linux or macOS
        try:
            # Download and run Ollama install script
            result = subprocess.run([
                'curl', '-fsSL', 'https://ollama.ai/install.sh'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Run the install script
                subprocess.run(['sh'], input=result.stdout, text=True, check=True)
                print("✅ Ollama installed successfully")
                return True
            else:
                print("❌ Failed to download Ollama install script")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Ollama: {e}")
            return False
    
    elif system == 'windows':
        print("🪟 For Windows, please download Ollama from: https://ollama.ai/download")
        print("   Then run this script again.")
        return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

def start_ollama_service():
    """Start Ollama service"""
    print_step(2, "Starting Ollama Service")
    
    try:
        # Check if Ollama is already running
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is already running")
            return True
    except requests.exceptions.RequestException:
        pass
    
    print("🚀 Starting Ollama service...")
    
    # Start Ollama service in background
    try:
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=2)
                if response.status_code == 200:
                    print("✅ Ollama service started successfully")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        
        print("❌ Ollama service failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start Ollama service: {e}")
        return False

def pull_default_models():
    """Pull default models for the chatbot"""
    print_step(3, "Pulling Default Models")
    
    # List of recommended models to pull
    models_to_pull = [
        'llama2:7b-chat',
        'mistral:7b-instruct',
        'tinyllama:1.1b'
    ]
    
    for model in models_to_pull:
        print(f"📥 Pulling model: {model}")
        try:
            result = subprocess.run(['ollama', 'pull', model], 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✅ Successfully pulled {model}")
            else:
                print(f"⚠️ Failed to pull {model}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⚠️ Timeout while pulling {model} (this is normal for large models)")
        except Exception as e:
            print(f"❌ Error pulling {model}: {e}")

def install_python_dependencies():
    """Install Python dependencies"""
    print_step(4, "Installing Python Dependencies")
    
    try:
        print("📦 Installing requirements...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def setup_database():
    """Set up the database"""
    print_step(5, "Setting Up Database")
    
    try:
        # Run database migrations
        print("🗄️ Running database migrations...")
        subprocess.run([sys.executable, 'manage_db.py', 'upgrade'], check=True)
        print("✅ Database setup completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set up database: {e}")
        return False

def create_env_file():
    """Create .env file with default configuration"""
    print_step(6, "Creating Environment Configuration")
    
    env_file = Path('.env')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Default environment configuration
    env_content = """# AI Chatbot Backend Configuration

# Flask Configuration
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=jwt-secret-string-change-in-production
JWT_EXPIRES_HOURS=24
JWT_REFRESH_EXPIRES_DAYS=30

# Database Configuration (PostgreSQL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chatbot_db
DB_USER=chatbot_user
DB_PASSWORD=chatbot_password

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama2:7b-chat

# Model Management
MODEL_CACHE_TTL=300
MAX_CONTEXT_LENGTH=4000
DEFAULT_TEMPERATURE=0.7

# System Monitoring
ENABLE_MONITORING=true
MONITORING_INTERVAL=30

# Hugging Face (optional)
# HF_TOKEN=your_huggingface_token_here

# Development
FLASK_ENV=development
FLASK_DEBUG=true
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
        print("⚠️ Please update the configuration values as needed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_backend_functionality():
    """Test basic backend functionality"""
    print_step(7, "Testing Backend Functionality")
    
    try:
        # Import and test services
        print("🧪 Testing service imports...")
        
        from services.ollama_service import ollama_service
        from services.huggingface_service import hf_service
        from services.chat_service import chat_service
        from services.model_manager import model_manager
        
        print("✅ All services imported successfully")
        
        # Test Ollama connection
        print("🧪 Testing Ollama connection...")
        if ollama_service.is_available():
            print("✅ Ollama service is accessible")
            
            # List available models
            models = ollama_service.list_models()
            print(f"✅ Found {len(models)} available models")
            
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model.name}")
        else:
            print("⚠️ Ollama service is not accessible")
        
        # Test model manager
        print("🧪 Testing model manager...")
        model_manager.start_monitoring()
        print("✅ Model manager started successfully")
        
        print("✅ Backend functionality test completed")
        return True
        
    except Exception as e:
        print(f"❌ Backend functionality test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print_step(8, "Setup Complete!")
    
    print("""
🎉 Enhanced AI Chatbot Backend Setup Complete!

Next Steps:
1. Review and update the .env file with your specific configuration
2. Start the backend server:
   python app.py

3. The backend will be available at: http://localhost:5000

Available API Endpoints:
- /api/health - System health check
- /api/models - Model management
- /api/chat/stream - Streaming chat responses
- /api/system/status - System monitoring
- /api/huggingface/search - HuggingFace model search

Features Available:
✅ Local LLM support via Ollama
✅ Multiple model management
✅ Real-time streaming responses
✅ System monitoring and analytics
✅ Hugging Face model integration
✅ Advanced conversation management
✅ Model performance tracking

For frontend integration, ensure your React app connects to:
http://localhost:5000

Documentation:
- API documentation: Check the /api endpoints
- Model management: Use /api/models endpoints
- System monitoring: Use /api/system endpoints

Enjoy your enhanced AI chatbot! 🤖
""")

def main():
    """Main setup function"""
    print("🤖 Enhanced AI Chatbot Backend Setup")
    print("=====================================")
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    success_steps = 0
    total_steps = 7
    
    # Step 1: Install Ollama
    if install_ollama():
        success_steps += 1
    
    # Step 2: Start Ollama service
    if start_ollama_service():
        success_steps += 1
    
    # Step 3: Pull default models
    pull_default_models()  # This might partially fail, so we don't count it
    success_steps += 1
    
    # Step 4: Install Python dependencies
    if install_python_dependencies():
        success_steps += 1
    
    # Step 5: Setup database
    if setup_database():
        success_steps += 1
    
    # Step 6: Create environment file
    if create_env_file():
        success_steps += 1
    
    # Step 7: Test functionality
    if test_backend_functionality():
        success_steps += 1
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Setup Results: {success_steps}/{total_steps} steps completed successfully")
    print('='*60)
    
    if success_steps >= 6:  # Allow for some model pulling failures
        print_next_steps()
    else:
        print("❌ Setup incomplete. Please check the errors above and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()