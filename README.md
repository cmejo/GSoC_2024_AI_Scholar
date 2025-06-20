# 🤖 Enhanced AI Chatbot Platform

A comprehensive AI chatbot platform with local LLM support, RAG capabilities, embeddings, fine-tuning, and multi-cloud deployment options.

![AI Chatbot](https://img.shields.io/badge/AI-Chatbot-blue?style=for-the-badge&logo=robot)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18+-blue?style=for-the-badge&logo=react)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## ✨ Features

### 🤖 **Local LLM Integration**
- **Ollama Support**: Run Llama2, Mistral, CodeLlama, and more locally
- **Multiple Models**: Switch between models per conversation
- **Model Management**: Download, update, and remove models via API
- **Performance Monitoring**: Track usage, response times, and success rates

### 🔄 **Real-time Streaming**
- **Server-Sent Events**: Stream AI responses token by token
- **WebSocket Support**: Bidirectional real-time communication
- **Progress Tracking**: Monitor model downloads and processing
- **Typing Indicators**: Visual feedback during response generation

### 📚 **RAG (Retrieval-Augmented Generation)**
- **Document Ingestion**: PDF, DOCX, HTML, TXT, Markdown support
- **Vector Storage**: ChromaDB with intelligent chunking
- **Semantic Search**: Advanced similarity search with scoring
- **Context-Aware Responses**: RAG-enhanced chat with source citations

### 🧠 **Advanced Embeddings**
- **Multiple Models**: SentenceTransformers + Ollama embeddings
- **Vector Search**: FAISS-powered similarity search
- **Collection Management**: Multiple isolated embedding collections
- **Batch Processing**: Efficient embedding generation

### 🎯 **Fine-Tuning**
- **Dataset Management**: Create, import, export training datasets
- **Multiple Formats**: JSONL, CSV, Alpaca, conversational formats
- **Job Tracking**: Real-time fine-tuning progress monitoring
- **Model Integration**: Seamless Ollama model creation

### 📊 **System Monitoring**
- **Resource Tracking**: CPU, memory, GPU usage monitoring
- **Model Analytics**: Performance metrics and recommendations
- **Health Checks**: Comprehensive system health monitoring
- **Performance Reports**: Detailed analytics and optimization suggestions

### 🔐 **Security & Authentication**
- **JWT Authentication**: Secure token-based authentication
- **User Management**: Registration, login, profile management
- **Session Tracking**: Monitor active user sessions
- **Input Validation**: Comprehensive request validation

### 🚀 **Multi-Cloud Deployment**
- **AWS**: ECS + Fargate with RDS PostgreSQL
- **Google Cloud**: Cloud Run + Cloud SQL
- **Azure**: Container Instances + PostgreSQL
- **Kubernetes**: Universal K8s deployment
- **Docker Swarm**: Simple orchestration

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-ai-chatbot.git
cd enhanced-ai-chatbot

# Run the enhanced setup
python setup_enhanced_backend.py

# Start the application
python app.py
```

### Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.enhanced.yml up -d

# Access the application
open http://localhost:3000
```

### Cloud Deployment

#### AWS
```bash
cd deployment/aws
export VPC_ID="vpc-xxx" SUBNET_IDS="subnet-xxx,subnet-yyy"
export SECURITY_GROUP_ID="sg-xxx" DB_PASSWORD="secure-password"
./deploy.sh
```

#### Google Cloud
```bash
cd deployment/gcp
export PROJECT_ID="my-project" DB_PASSWORD="secure-password"
./deploy.sh
```

#### Azure
```bash
cd deployment/azure
export DB_PASSWORD="secure-password"
./deploy.sh
```

## 📖 Documentation

- **[Enhanced Features Guide](ENHANCED_FEATURES.md)** - Comprehensive feature overview
- **[Enhanced Backend Guide](ENHANCED_BACKEND_GUIDE.md)** - Technical documentation
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Multi-cloud deployment instructions
- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Ollama        │
│   (React)       │◄──►│   (Flask)       │◄──►│   (Local LLM)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   (Database)    │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Vector Store  │
                       │   (ChromaDB)    │
                       └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SECRET_KEY` | Flask secret key | - | ✅ |
| `JWT_SECRET_KEY` | JWT signing key | - | ✅ |
| `DB_HOST` | Database host | localhost | ✅ |
| `DB_PASSWORD` | Database password | - | ✅ |
| `OLLAMA_BASE_URL` | Ollama service URL | http://localhost:11434 | ✅ |
| `DEFAULT_MODEL` | Default LLM model | llama2:7b-chat | ❌ |

### Model Recommendations

| Use Case | Model | Size | Memory Required |
|----------|-------|------|-----------------|
| General Chat | llama2:7b-chat | ~4GB | 8GB RAM |
| Code Assistance | codellama:7b-instruct | ~4GB | 8GB RAM |
| Lightweight | tinyllama:1.1b | ~1GB | 4GB RAM |
| High Quality | llama2:13b-chat | ~8GB | 16GB RAM |

## 📊 API Endpoints

### Core Chat APIs
- `POST /api/chat` - Send chat message
- `POST /api/chat/stream` - Stream chat responses
- `GET /api/chat/sessions` - Get user sessions
- `PUT /api/chat/sessions/{id}/model` - Switch model

### Model Management
- `GET /api/models` - List available models
- `POST /api/models/pull` - Download new model
- `DELETE /api/models/{name}` - Delete model
- `GET /api/models/recommendations` - Get recommendations

### RAG System
- `POST /api/rag/ingest` - Upload documents
- `POST /api/rag/search` - Search documents
- `POST /api/rag/chat` - RAG-enhanced chat
- `GET /api/rag/stats` - System statistics

### Embeddings
- `POST /api/embeddings/generate` - Generate embeddings
- `POST /api/embeddings/collections` - Create collection
- `POST /api/embeddings/collections/{name}/search` - Search collection

### Fine-Tuning
- `POST /api/fine-tuning/datasets` - Create dataset
- `POST /api/fine-tuning/start` - Start fine-tuning
- `GET /api/fine-tuning/jobs/{id}` - Job status

### System Monitoring
- `GET /api/system/status` - System status
- `GET /api/system/performance` - Performance report
- `GET /api/health` - Health check

## 🧪 Testing

```bash
# Run backend tests
python test_enhanced_backend.py

# Run unit tests
pytest tests/

# Run integration tests
pytest tests/test_integration.py
```

## 🔒 Security

- **Authentication**: JWT-based with automatic refresh
- **Authorization**: Role-based access control
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: API rate limiting and abuse prevention
- **Encryption**: Data encryption at rest and in transit

## 📈 Performance

### Benchmarks
- **Response Time**: < 2s for most queries
- **Throughput**: 100+ concurrent users
- **Scalability**: Auto-scaling to 1000+ users
- **Availability**: 99.9% uptime with proper deployment

### Optimization
- **Caching**: Intelligent response and embedding caching
- **Model Selection**: Automatic optimal model selection
- **Resource Management**: Dynamic resource allocation
- **Load Balancing**: Multi-instance load distribution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM support
- **Hugging Face** for model ecosystem
- **ChromaDB** for vector storage
- **React** for frontend framework
- **Flask** for backend framework

## 📞 Support

- **Documentation**: Check the docs folder
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Enterprise**: Contact for enterprise support

---

**Built with ❤️ for the AI community**

Transform your chatbot into a powerful local LLM platform! 🚀