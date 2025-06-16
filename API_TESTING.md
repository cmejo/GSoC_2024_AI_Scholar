# API Testing Guide

This document provides comprehensive information about testing the AI Chatbot API endpoints, including automated tests, manual testing procedures, and testing best practices.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [API Endpoints](#api-endpoints)
- [Test Scenarios](#test-scenarios)
- [Manual Testing](#manual-testing)
- [Performance Testing](#performance-testing)
- [Security Testing](#security-testing)
- [Continuous Integration](#continuous-integration)

## Overview

The AI Chatbot application includes comprehensive API tests covering:

- **Authentication**: User registration, login, token validation
- **Chat Functionality**: Message sending, session management, conversation history
- **Data Persistence**: Database operations, data integrity
- **Error Handling**: Edge cases, invalid inputs, service failures
- **Integration**: End-to-end workflows, cross-component interactions

## Test Structure

### Test Organization

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_auth_api.py           # Authentication API tests
├── test_chat_api.py           # Chat functionality tests
├── test_models.py             # Database model tests
├── test_health_api.py         # Health check and utility tests
└── test_integration.py        # Integration and workflow tests
```

### Test Categories

#### Unit Tests
- **Models** (`test_models.py`): Database model functionality, relationships, validation
- **Services**: Individual service class testing

#### API Tests
- **Authentication** (`test_auth_api.py`): Registration, login, token management
- **Chat** (`test_chat_api.py`): Message handling, session management
- **Health** (`test_health_api.py`): System health, error handling

#### Integration Tests
- **Workflows** (`test_integration.py`): Complete user journeys, cross-component testing

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run specific test type
python run_tests.py --type api --verbose
```

### Detailed Test Execution

#### All Tests
```bash
# Complete test suite with coverage report
python run_tests.py --type all --coverage --html-report

# Generate comprehensive report
python run_tests.py --report
```

#### Specific Test Categories
```bash
# Unit tests only
python run_tests.py --type unit

# API tests only
python run_tests.py --type api

# Integration tests only
python run_tests.py --type integration
```

#### Individual Test Files
```bash
# Specific test file
python run_tests.py --type specific --test-path tests/test_auth_api.py

# Specific test function
python run_tests.py --type specific --test-path tests/test_auth_api.py::TestAuthRegistration::test_register_success
```

#### Using Pytest Directly
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov=models --cov-report=html

# Run specific test
pytest tests/test_auth_api.py::TestAuthRegistration::test_register_success -v

# Run tests matching pattern
pytest tests/ -k "test_auth" -v
```

### Test Environment Setup

#### Automatic Setup
The test runner automatically sets up the test environment:
- Creates isolated test database
- Loads environment variables
- Configures test fixtures

#### Manual Setup
```bash
# Set test environment variables
export FLASK_ENV=testing
export TESTING=true
export DB_NAME=chatbot_test_db

# Run tests
pytest tests/ -v
```

## API Endpoints

### Authentication Endpoints

#### POST /api/auth/register
**Purpose**: User registration
**Test Coverage**:
- Successful registration
- Duplicate username/email
- Invalid input validation
- Password requirements

```python
def test_register_success(self, client):
    data = {
        'username': 'newuser',
        'email': 'newuser@example.com',
        'password': 'password123'
    }
    response = client.post('/api/auth/register', json=data)
    assert response.status_code == 201
```

#### POST /api/auth/login
**Purpose**: User authentication
**Test Coverage**:
- Valid credentials
- Invalid credentials
- Username vs email login
- Session creation

#### GET /api/auth/me
**Purpose**: Get current user info
**Test Coverage**:
- Valid token
- Invalid token
- Expired token
- Missing token

#### POST /api/auth/logout
**Purpose**: User logout
**Test Coverage**:
- Successful logout
- Token invalidation

### Chat Endpoints

#### POST /api/chat
**Purpose**: Send chat message
**Test Coverage**:
- Successful message sending
- AI response generation
- Session creation
- Conversation context
- Error handling

```python
def test_chat_success(self, client, auth_headers, mock_ollama_api):
    data = {
        'message': 'Hello, how are you?',
        'session_id': None
    }
    response = client.post('/api/chat', json=data, headers=auth_headers)
    assert response.status_code == 200
    assert response.json['success'] is True
```

#### GET /api/chat/sessions
**Purpose**: Get user's chat sessions
**Test Coverage**:
- Session listing
- Pagination
- User isolation
- Empty results

#### POST /api/chat/sessions
**Purpose**: Create new chat session
**Test Coverage**:
- Session creation
- Custom names
- Default naming

#### GET /api/chat/sessions/{id}
**Purpose**: Get session messages
**Test Coverage**:
- Message retrieval
- Chronological ordering
- Access control
- Non-existent sessions

#### DELETE /api/chat/sessions/{id}
**Purpose**: Delete chat session
**Test Coverage**:
- Successful deletion
- Cascade deletion of messages
- Access control
- Non-existent sessions

### Utility Endpoints

#### GET /api/health
**Purpose**: System health check
**Test Coverage**:
- Service availability
- AI service connectivity
- Database connectivity
- Error conditions

## Test Scenarios

### Authentication Flow Tests

#### Complete Registration Flow
```python
def test_complete_user_journey(self, client, mock_ollama_api):
    # 1. Register new user
    register_data = {
        'username': 'journeyuser',
        'email': 'journey@example.com',
        'password': 'password123'
    }
    response = client.post('/api/auth/register', json=register_data)
    assert response.status_code == 201
    
    # 2. Extract token
    token = response.json['token']
    headers = {'Authorization': f'Bearer {token}'}
    
    # 3. Verify authentication
    response = client.get('/api/auth/me', headers=headers)
    assert response.status_code == 200
```

#### Token Validation Tests
```python
def test_expired_token(self, client, app, sample_user):
    # Create expired token
    expired_token = jwt.encode({
        'user_id': sample_user.id,
        'exp': datetime.utcnow() - timedelta(hours=1)
    }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
    
    headers = {'Authorization': f'Bearer {expired_token}'}
    response = client.get('/api/auth/me', headers=headers)
    assert response.status_code == 401
```

### Chat Functionality Tests

#### Message Flow Tests
```python
def test_conversation_flow(self, client, auth_headers, mock_ollama_api):
    # 1. Send first message
    data = {'message': 'Hello!', 'session_id': None}
    response = client.post('/api/chat', json=data, headers=auth_headers)
    session_id = response.json['session_id']
    
    # 2. Send follow-up message
    data = {'message': 'How are you?', 'session_id': session_id}
    response = client.post('/api/chat', json=data, headers=auth_headers)
    
    # 3. Verify conversation history
    response = client.get(f'/api/chat/sessions/{session_id}', headers=auth_headers)
    messages = response.json['messages']
    assert len(messages) == 4  # 2 user + 2 bot messages
```

#### Session Management Tests
```python
def test_multiple_sessions(self, client, auth_headers):
    # Create multiple sessions
    session_ids = []
    for i in range(3):
        data = {'name': f'Session {i+1}'}
        response = client.post('/api/chat/sessions', json=data, headers=auth_headers)
        session_ids.append(response.json['session_id'])
    
    # Verify all sessions exist
    response = client.get('/api/chat/sessions', headers=auth_headers)
    assert len(response.json['sessions']) == 3
```

### Error Handling Tests

#### Invalid Input Tests
```python
def test_invalid_inputs(self, client, auth_headers):
    test_cases = [
        {'message': ''},  # Empty message
        {'message': None},  # Null message
        {},  # Missing message
    ]
    
    for data in test_cases:
        response = client.post('/api/chat', json=data, headers=auth_headers)
        assert response.status_code == 400
```

#### Service Failure Tests
```python
def test_ai_service_failure(self, client, auth_headers):
    with patch('requests.post', side_effect=ConnectionError('Service down')):
        data = {'message': 'Hello', 'session_id': None}
        response = client.post('/api/chat', json=data, headers=auth_headers)
        assert response.json['success'] is False
        assert 'Connection Error' in response.json['error']
```

### Data Integrity Tests

#### Cascade Deletion Tests
```python
def test_session_deletion_cascade(self, client, auth_headers, app):
    # Create session with messages
    session_data = {'name': 'Test Session'}
    response = client.post('/api/chat/sessions', json=session_data, headers=auth_headers)
    session_id = response.json['session_id']
    
    # Add messages
    chat_data = {'message': 'Test message', 'session_id': session_id}
    client.post('/api/chat', json=chat_data, headers=auth_headers)
    
    # Delete session
    response = client.delete(f'/api/chat/sessions/{session_id}', headers=auth_headers)
    assert response.status_code == 200
    
    # Verify messages were also deleted
    with app.app_context():
        messages = ChatMessage.query.filter_by(session_id=session_id).count()
        assert messages == 0
```

#### User Isolation Tests
```python
def test_user_isolation(self, client, mock_ollama_api):
    # Create two users
    users = []
    for i in range(2):
        data = {
            'username': f'user{i}',
            'email': f'user{i}@example.com',
            'password': 'password123'
        }
        response = client.post('/api/auth/register', json=data)
        users.append(response.json)
    
    # User 1 creates session
    headers1 = {'Authorization': f'Bearer {users[0]["token"]}'}
    session_data = {'name': 'User 1 Session'}
    response = client.post('/api/chat/sessions', json=session_data, headers=headers1)
    session_id = response.json['session_id']
    
    # User 2 cannot access User 1's session
    headers2 = {'Authorization': f'Bearer {users[1]["token"]}'}
    response = client.get(f'/api/chat/sessions/{session_id}', headers=headers2)
    assert response.status_code == 404
```

## Manual Testing

### Using curl

#### Register User
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'
```

#### Login
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }'
```

#### Send Chat Message
```bash
# Replace TOKEN with actual JWT token
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "message": "Hello, how are you?",
    "session_id": null
  }'
```

#### Get Sessions
```bash
curl -X GET http://localhost:5000/api/chat/sessions \
  -H "Authorization: Bearer TOKEN"
```

### Using Postman

#### Environment Setup
Create a Postman environment with:
- `base_url`: `http://localhost:5000`
- `token`: (will be set after login)

#### Collection Structure
1. **Authentication**
   - Register User
   - Login User
   - Get Current User
   - Logout

2. **Chat**
   - Send Message
   - Create Session
   - Get Sessions
   - Get Session Messages
   - Delete Session

3. **Health**
   - Health Check

#### Pre-request Scripts
For authenticated requests, add:
```javascript
pm.request.headers.add({
    key: 'Authorization',
    value: 'Bearer ' + pm.environment.get('token')
});
```

#### Test Scripts
For login request:
```javascript
if (pm.response.code === 200) {
    const response = pm.response.json();
    pm.environment.set('token', response.token);
    pm.environment.set('user_id', response.user.id);
}
```

## Performance Testing

### Load Testing with pytest-benchmark

```python
def test_chat_performance(self, client, auth_headers, benchmark):
    data = {'message': 'Performance test message', 'session_id': None}
    
    def send_message():
        return client.post('/api/chat', json=data, headers=auth_headers)
    
    result = benchmark(send_message)
    assert result.status_code == 200
```

### Concurrent User Testing

```python
import threading
import time

def test_concurrent_users(self, client):
    results = []
    
    def user_session():
        # Register user
        user_data = {
            'username': f'user_{threading.current_thread().ident}',
            'email': f'user_{threading.current_thread().ident}@example.com',
            'password': 'password123'
        }
        response = client.post('/api/auth/register', json=user_data)
        results.append(response.status_code)
    
    # Create 10 concurrent users
    threads = []
    for i in range(10):
        thread = threading.Thread(target=user_session)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # All registrations should succeed
    assert all(code == 201 for code in results)
```

## Security Testing

### Authentication Security

#### SQL Injection Tests
```python
def test_sql_injection_protection(self, client):
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "admin' OR '1'='1",
        "' UNION SELECT * FROM users --"
    ]
    
    for malicious_input in malicious_inputs:
        data = {
            'username': malicious_input,
            'password': 'password123'
        }
        response = client.post('/api/auth/login', json=data)
        # Should not cause server error
        assert response.status_code in [400, 401]
```

#### XSS Protection Tests
```python
def test_xss_protection(self, client, auth_headers):
    xss_payloads = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>"
    ]
    
    for payload in xss_payloads:
        data = {'message': payload, 'session_id': None}
        response = client.post('/api/chat', json=data, headers=auth_headers)
        # Should handle safely
        assert response.status_code == 200
```

#### Rate Limiting Tests
```python
def test_rate_limiting(self, client):
    # Attempt many rapid requests
    for i in range(100):
        data = {
            'username': 'testuser',
            'password': 'wrongpassword'
        }
        response = client.post('/api/auth/login', json=data)
        
        # Should eventually rate limit (if implemented)
        if response.status_code == 429:
            break
    else:
        # If no rate limiting, should still handle gracefully
        assert True
```

## Continuous Integration

### GitHub Actions Integration

The tests are integrated with GitHub Actions for automated testing:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: python run_tests.py --type all --coverage
      env:
        DB_HOST: localhost
        DB_USER: postgres
        DB_PASSWORD: postgres
        DB_NAME: chatbot_test_db
```

### Test Reporting

#### Coverage Reports
```bash
# Generate HTML coverage report
python run_tests.py --coverage --html-report

# View coverage
open htmlcov/index.html
```

#### JUnit XML Reports
```bash
# Generate JUnit XML for CI
pytest tests/ --junit-xml=test-results.xml
```

#### Test Metrics
- **Code Coverage**: Target >80%
- **Test Execution Time**: <30 seconds for full suite
- **Test Success Rate**: 100% for CI/CD pipeline

## Best Practices

### Test Writing Guidelines

1. **Descriptive Names**: Use clear, descriptive test function names
2. **Single Responsibility**: Each test should test one specific behavior
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Test Isolation**: Tests should not depend on each other
5. **Mock External Services**: Use mocks for external dependencies

### Test Data Management

1. **Use Fixtures**: Leverage pytest fixtures for test data
2. **Clean State**: Ensure clean database state for each test
3. **Realistic Data**: Use realistic test data that reflects production scenarios
4. **Edge Cases**: Test boundary conditions and edge cases

### Debugging Tests

#### Verbose Output
```bash
python run_tests.py --type all --verbose
```

#### Debug Specific Test
```bash
pytest tests/test_auth_api.py::test_register_success -v -s --pdb
```

#### Database Inspection
```python
def test_debug_database(self, client, app):
    with app.app_context():
        users = User.query.all()
        print(f"Users in database: {len(users)}")
        for user in users:
            print(f"  - {user.username}: {user.email}")
```

For additional testing support, refer to the main documentation or contact the development team.