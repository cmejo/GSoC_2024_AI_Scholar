# Contributing to Enhanced AI Chatbot Platform

Thank you for your interest in contributing to the Enhanced AI Chatbot Platform! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Operating system and version
   - Python version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs

### Suggesting Features

1. **Check existing feature requests** first
2. **Use the feature request template**
3. **Provide clear use cases** and benefits
4. **Consider implementation complexity**

### Code Contributions

#### Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/enhanced-ai-chatbot.git
   cd enhanced-ai-chatbot
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes**
3. **Write tests** for new functionality
4. **Run tests**:
   ```bash
   pytest tests/
   python test_enhanced_backend.py
   ```
5. **Check code quality**:
   ```bash
   black .
   isort .
   flake8 .
   ```
6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request**

## 📝 Code Standards

### Python Code Style

- **Follow PEP 8** style guidelines
- **Use Black** for code formatting
- **Use isort** for import sorting
- **Use type hints** where appropriate
- **Write docstrings** for all functions and classes

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

Examples:
```
feat: add RAG document ingestion API
fix: resolve memory leak in embedding service
docs: update deployment guide for AWS
```

### Testing

- **Write unit tests** for new functions
- **Write integration tests** for API endpoints
- **Maintain test coverage** above 80%
- **Test edge cases** and error conditions

### Documentation

- **Update README.md** for new features
- **Add API documentation** for new endpoints
- **Include code examples** in docstrings
- **Update deployment guides** if needed

## 🏗️ Project Structure

```
enhanced-ai-chatbot/
├── app.py                 # Main Flask application
├── models.py              # Database models
├── requirements.txt       # Python dependencies
├── services/              # Core services
│   ├── ollama_service.py
│   ├── rag_service.py
│   ├── embeddings_service.py
│   └── fine_tuning_service.py
├── frontend/              # React frontend
├── deployment/            # Cloud deployment scripts
├── tests/                 # Test suite
├── migrations/            # Database migrations
└── docs/                  # Documentation
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_rag_service.py

# Run with coverage
pytest --cov=services tests/

# Run integration tests
python test_enhanced_backend.py
```

### Writing Tests

```python
import pytest
from services.rag_service import rag_service

def test_document_ingestion():
    """Test document ingestion functionality."""
    # Arrange
    text = "Sample document content"
    metadata = {"source": "test"}
    
    # Act
    result = rag_service.ingest_text(text, metadata)
    
    # Assert
    assert result is True
```

## 🚀 Development Environment

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- Docker and Docker Compose
- PostgreSQL (optional, can use Docker)

### Setup

1. **Backend setup**:
   ```bash
   python setup_enhanced_backend.py
   ```

2. **Frontend setup**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

3. **Database setup**:
   ```bash
   python manage_db.py upgrade
   ```

### Environment Variables

Create a `.env` file:
```bash
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
DB_HOST=localhost
DB_PASSWORD=your-db-password
OLLAMA_BASE_URL=http://localhost:11434
```

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts

### PR Description

Include:
- **Summary** of changes
- **Motivation** and context
- **Testing** performed
- **Screenshots** (if UI changes)
- **Breaking changes** (if any)

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in review environment
4. **Approval** and merge

## 🎯 Areas for Contribution

### High Priority

- **Performance optimizations**
- **Additional model integrations**
- **Enhanced security features**
- **Mobile responsiveness**
- **API rate limiting**

### Medium Priority

- **Additional deployment options**
- **Monitoring and alerting**
- **Backup and recovery**
- **Multi-language support**
- **Plugin system**

### Documentation

- **API documentation**
- **Tutorial videos**
- **Best practices guides**
- **Troubleshooting guides**
- **Architecture diagrams**

## 🐛 Bug Reports

### Information to Include

- **Environment details**:
  - OS and version
  - Python version
  - Browser (if frontend issue)
  - Deployment method

- **Steps to reproduce**:
  1. Step one
  2. Step two
  3. Step three

- **Expected behavior**
- **Actual behavior**
- **Error messages**
- **Logs** (if available)

### Bug Report Template

```markdown
**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Browser: [e.g., Chrome 96]

**Steps to Reproduce:**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior:**
A clear description of what you expected to happen.

**Actual Behavior:**
A clear description of what actually happened.

**Screenshots:**
If applicable, add screenshots to help explain your problem.

**Additional Context:**
Add any other context about the problem here.
```

## 💡 Feature Requests

### Information to Include

- **Use case** description
- **Proposed solution**
- **Alternative solutions** considered
- **Implementation complexity** estimate
- **Benefits** to users

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Hall of Fame** for major contributors

## 📞 Getting Help

- **GitHub Discussions** for questions
- **Discord** for real-time chat (link in README)
- **Email** for private inquiries

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Enhanced AI Chatbot Platform! 🚀