"""
Pytest configuration and fixtures for AI Chatbot tests
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from app import app as flask_app
from models import db, User, ChatSession, ChatMessage, UserSession
import jwt
from datetime import datetime, timedelta


@pytest.fixture(scope='session')
def app():
    """Create application for testing"""
    # Create a temporary database for testing
    db_fd, db_path = tempfile.mkstemp()
    
    flask_app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'SECRET_KEY': 'test-secret-key',
        'JWT_SECRET_KEY': 'test-jwt-secret',
        'JWT_ACCESS_TOKEN_EXPIRES': timedelta(hours=1),
        'WTF_CSRF_ENABLED': False
    })
    
    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.drop_all()
    
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create test CLI runner"""
    return app.test_cli_runner()


@pytest.fixture
def db_session(app):
    """Create database session for testing"""
    with app.app_context():
        db.session.begin()
        yield db.session
        db.session.rollback()


@pytest.fixture
def sample_user(app):
    """Create a sample user for testing"""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@example.com',
            password='testpassword123'
        )
        db.session.add(user)
        db.session.commit()
        yield user
        # Cleanup is handled by the app fixture


@pytest.fixture
def admin_user(app):
    """Create an admin user for testing"""
    with app.app_context():
        user = User(
            username='admin',
            email='admin@example.com',
            password='adminpassword123'
        )
        db.session.add(user)
        db.session.commit()
        yield user


@pytest.fixture
def auth_token(app, sample_user):
    """Generate JWT token for authenticated requests"""
    with app.app_context():
        token = jwt.encode({
            'user_id': sample_user.id,
            'username': sample_user.username,
            'exp': datetime.utcnow() + timedelta(hours=1)
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        return token


@pytest.fixture
def auth_headers(auth_token):
    """Create authorization headers for API requests"""
    return {'Authorization': f'Bearer {auth_token}'}


@pytest.fixture
def sample_chat_session(app, sample_user):
    """Create a sample chat session for testing"""
    with app.app_context():
        session = ChatSession(
            user_id=sample_user.id,
            session_name='Test Session'
        )
        db.session.add(session)
        db.session.commit()
        yield session


@pytest.fixture
def sample_messages(app, sample_user, sample_chat_session):
    """Create sample chat messages for testing"""
    with app.app_context():
        messages = []
        
        # User message
        user_msg = ChatMessage(
            session_id=sample_chat_session.id,
            user_id=sample_user.id,
            message_type='user',
            content='Hello, this is a test message'
        )
        db.session.add(user_msg)
        messages.append(user_msg)
        
        # Bot response
        bot_msg = ChatMessage(
            session_id=sample_chat_session.id,
            user_id=sample_user.id,
            message_type='bot',
            content='Hello! I am a test bot response.',
            model_used='test-model',
            response_time=0.5
        )
        db.session.add(bot_msg)
        messages.append(bot_msg)
        
        db.session.commit()
        yield messages


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {
        'response': 'This is a mocked AI response for testing purposes.',
        'model': 'test-model',
        'created_at': '2024-01-01T00:00:00Z',
        'done': True
    }


@pytest.fixture
def mock_ollama_api(mock_ollama_response):
    """Mock Ollama API calls"""
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_ollama_health():
    """Mock Ollama health check"""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'ok'}
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture(autouse=True)
def clean_db(app):
    """Clean database after each test"""
    yield
    with app.app_context():
        # Clean up any remaining data
        db.session.query(ChatMessage).delete()
        db.session.query(ChatSession).delete()
        db.session.query(UserSession).delete()
        db.session.query(User).delete()
        db.session.commit()


# Test data factories
class UserFactory:
    """Factory for creating test users"""
    
    @staticmethod
    def create(username='testuser', email='test@example.com', password='testpass123'):
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        return user


class ChatSessionFactory:
    """Factory for creating test chat sessions"""
    
    @staticmethod
    def create(user_id, session_name='Test Session'):
        session = ChatSession(user_id=user_id, session_name=session_name)
        db.session.add(session)
        db.session.commit()
        return session


class ChatMessageFactory:
    """Factory for creating test chat messages"""
    
    @staticmethod
    def create(session_id, user_id, message_type='user', content='Test message'):
        message = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            message_type=message_type,
            content=content
        )
        db.session.add(message)
        db.session.commit()
        return message