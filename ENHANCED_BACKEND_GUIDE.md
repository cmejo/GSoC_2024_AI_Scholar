# Enhanced AI Chatbot Backend Guide

## Overview

This enhanced backend system provides a comprehensive AI chatbot platform with local LLM support via Ollama, Hugging Face integration, advanced model management, and real-time streaming capabilities.

## Features

### 🤖 Local LLM Support
- **Ollama Integration**: Run open-source LLMs locally
- **Multiple Model Support**: Switch between different models per conversation
- **Model Management**: Download, update, and remove models
- **Performance Monitoring**: Track model usage and performance metrics

### 🔄 Real-time Streaming
- **Server-Sent Events (SSE)**: Stream responses in real-time
- **WebSocket Support**: Bidirectional communication
- **Typing Indicators**: Show when AI is generating responses
- **Progress Tracking**: Monitor model download and processing progress

### 📊 System Monitoring
- **Resource Monitoring**: CPU, memory, GPU usage tracking
- **Model Analytics**: Usage statistics, response times, success rates
- **Health Checks**: Comprehensive system health monitoring
- **Performance Reports**: Detailed analytics and recommendations

### 🤗 Hugging Face Integration
- **Model Discovery**: Search and browse Hugging Face models
- **Model Download**: Download and convert models for Ollama
- **Compatibility Check**: Automatic compatibility assessment
- **Recommended Models**: Curated list of tested models

### 💬 Advanced Chat Features
- **Context Management**: Intelligent conversation context handling
- **System Prompts**: Customizable AI behavior and personality
- **Parameter Tuning**: Adjust temperature, top-p, top-k per session
- **Model Switching**: Change models mid-conversation
- **Session Management**: Organize conversations with metadata

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Ollama        │
│   (React)       │◄──►│   (Flask)       │◄──►│   (Local LLM)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Database      │
                       │   (PostgreSQL)  │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Hugging Face   │
                       │     (API)       │
                       └─────────────────┘
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - User logout
- `POST /api/auth/refresh` - Refresh token

### Chat Management
- `POST /api/chat` - Send chat message
- `POST /api/chat/stream` - Stream chat responses
- `GET /api/chat/sessions` - Get user sessions
- `POST /api/chat/sessions` - Create new session
- `GET /api/chat/sessions/{id}` - Get session messages
- `PUT /api/chat/sessions/{id}` - Update session
- `DELETE /api/chat/sessions/{id}` - Delete session

### Enhanced Chat Features
- `PUT /api/chat/sessions/{id}/model` - Switch session model
- `PUT /api/chat/sessions/{id}/parameters` - Update model parameters
- `PUT /api/chat/sessions/{id}/system-prompt` - Update system prompt

### Model Management
- `GET /api/models` - List available models
- `GET /api/models/{name}` - Get model details
- `POST /api/models/pull` - Download new model
- `DELETE /api/models/{name}` - Delete model
- `GET /api/models/recommendations` - Get model recommendations

### System Monitoring
- `GET /api/system/status` - System status
- `GET /api/system/performance` - Performance report
- `GET /api/system/health` - Health check
- `POST /api/system/cleanup` - Clean unused models

### Hugging Face Integration
- `GET /api/huggingface/search` - Search HF models
- `GET /api/huggingface/recommended` - Get recommended models
- `POST /api/huggingface/download` - Download HF model

### Health & Monitoring
- `GET /api/health` - Comprehensive health check

## Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL (or SQLite for development)
- Git
- curl (for Ollama installation)

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai-chatbot

# Run the enhanced setup script
python setup_enhanced_backend.py
```

### Manual Setup

1. **Install Ollama**
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai/download
   ```

2. **Start Ollama Service**
   ```bash
   ollama serve
   ```

3. **Pull Default Models**
   ```bash
   ollama pull llama2:7b-chat
   ollama pull mistral:7b-instruct
   ollama pull tinyllama:1.1b
   ```

4. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Setup Database**
   ```bash
   python manage_db.py upgrade
   ```

6. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

7. **Start the Backend**
   ```bash
   python app.py
   ```

## Configuration

### Environment Variables

```bash
# Flask Configuration
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=jwt-secret-string-change-in-production
JWT_EXPIRES_HOURS=24

# Database Configuration
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
```

### Model Parameters

Default model parameters can be customized per session:

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 2048,
  "repeat_penalty": 1.1,
  "stop": ["User:", "Human:"]
}
```

### System Prompts

Available system prompt types:
- `general`: General helpful assistant
- `creative`: Creative writing assistant
- `technical`: Technical support assistant
- `casual`: Casual conversation
- `professional`: Professional business assistant

## Usage Examples

### Frontend Integration

#### Streaming Chat
```javascript
// Stream chat responses
const response = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    message: 'Hello, how are you?',
    session_id: sessionId,
    model: 'llama2:7b-chat'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      console.log('Received:', data.content);
      
      if (data.is_complete) {
        console.log('Response complete');
        break;
      }
    }
  }
}
```

#### Model Management
```javascript
// Get available models
const models = await fetch('/api/models', {
  headers: { 'Authorization': `Bearer ${token}` }
}).then(r => r.json());

// Switch model for session
await fetch(`/api/chat/sessions/${sessionId}/model`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({ model: 'mistral:7b-instruct' })
});

// Update model parameters
await fetch(`/api/chat/sessions/${sessionId}/parameters`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    parameters: {
      temperature: 0.8,
      top_p: 0.95,
      max_tokens: 1024
    }
  })
});
```

#### System Monitoring
```javascript
// Get system status
const status = await fetch('/api/system/status', {
  headers: { 'Authorization': `Bearer ${token}` }
}).then(r => r.json());

console.log('System Status:', status);

// Get performance report
const performance = await fetch('/api/system/performance', {
  headers: { 'Authorization': `Bearer ${token}` }
}).then(r => r.json());

console.log('Performance Report:', performance);
```

### Backend Service Usage

#### Chat Service
```python
from services.chat_service import chat_service

# Generate streaming response
for response in chat_service.generate_response(
    session_id=1,
    user_id=1,
    message="Hello, how are you?",
    model="llama2:7b-chat",
    stream=True
):
    print(response.content)
    if response.is_complete:
        break
```

#### Model Manager
```python
from services.model_manager import model_manager

# Get optimal model for use case
optimal_model = model_manager.get_optimal_model(
    use_case="code_assistance",
    max_memory_gb=8.0
)

# Record model usage
model_manager.record_model_usage(
    model_name="llama2:7b-chat",
    response_time=2.5,
    token_count=150,
    success=True
)

# Get performance report
report = model_manager.get_model_performance_report()
```

#### Ollama Service
```python
from services.ollama_service import ollama_service

# List available models
models = ollama_service.list_models()

# Generate response
for response in ollama_service.chat(
    model="llama2:7b-chat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
):
    print(response.content)
```

## Performance Optimization

### Model Selection
- Use smaller models (tinyllama:1.1b) for quick responses
- Use larger models (llama2:13b) for complex tasks
- Switch models based on conversation context

### Resource Management
- Monitor system resources via `/api/system/status`
- Clean up unused models with `/api/system/cleanup`
- Adjust model parameters based on performance

### Caching
- Model information is cached for 5 minutes
- Conversation contexts are cached in memory
- Database queries are optimized with indexes

## Troubleshooting

### Common Issues

1. **Ollama Service Not Available**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama service
   ollama serve
   ```

2. **Model Not Found**
   ```bash
   # List available models
   ollama list
   
   # Pull missing model
   ollama pull llama2:7b-chat
   ```

3. **Database Connection Error**
   ```bash
   # Check database configuration in .env
   # Run database migrations
   python manage_db.py upgrade
   ```

4. **Memory Issues**
   - Use smaller models
   - Adjust `max_tokens` parameter
   - Monitor system resources

### Logs and Debugging

- Application logs: Check console output
- Ollama logs: Check Ollama service logs
- Database logs: Check PostgreSQL logs
- System monitoring: Use `/api/system/health`

## Security Considerations

- JWT tokens for authentication
- Input validation and sanitization
- Rate limiting (implement as needed)
- CORS configuration for frontend
- Environment variable protection
- Database connection security

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation
- Check system health endpoints
- Submit issues on GitHub

---

**Happy Chatting! 🤖**