"""
Tests for database models
"""

import pytest
from datetime import datetime, timedelta
from models import User, ChatSession, ChatMessage, UserSession, DatabaseQueries, db


class TestUserModel:
    """Test User model"""
    
    def test_create_user(self, app):
        """Test creating a new user"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            db.session.add(user)
            db.session.commit()
            
            assert user.id is not None
            assert user.username == 'testuser'
            assert user.email == 'test@example.com'
            assert user.password_hash is not None
            assert user.password_hash != 'password123'  # Should be hashed
            assert user.is_active is True
            assert user.created_at is not None
    
    def test_user_password_hashing(self, app):
        """Test password hashing and verification"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            
            # Test password verification
            assert user.check_password('password123') is True
            assert user.check_password('wrongpassword') is False
    
    def test_user_set_password(self, app):
        """Test setting a new password"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password='oldpassword'
            )
            
            old_hash = user.password_hash
            user.set_password('newpassword')
            
            assert user.password_hash != old_hash
            assert user.check_password('newpassword') is True
            assert user.check_password('oldpassword') is False
    
    def test_user_update_last_login(self, app):
        """Test updating last login timestamp"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            db.session.add(user)
            db.session.commit()
            
            original_last_login = user.last_login
            user.update_last_login()
            
            assert user.last_login != original_last_login
            assert user.last_login is not None
    
    def test_user_to_dict(self, app):
        """Test user serialization to dictionary"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            db.session.add(user)
            db.session.commit()
            
            user_dict = user.to_dict()
            
            assert user_dict['id'] == user.id
            assert user_dict['username'] == 'testuser'
            assert user_dict['email'] == 'test@example.com'
            assert user_dict['is_active'] is True
            assert 'password_hash' not in user_dict  # Should not include password
            assert 'created_at' in user_dict
    
    def test_user_relationships(self, app):
        """Test user relationships with sessions and messages"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            db.session.add(user)
            db.session.commit()
            
            # Create chat session
            session = ChatSession(user_id=user.id, session_name='Test Session')
            db.session.add(session)
            db.session.commit()
            
            # Create chat message
            message = ChatMessage(
                session_id=session.id,
                user_id=user.id,
                message_type='user',
                content='Test message'
            )
            db.session.add(message)
            db.session.commit()
            
            # Test relationships
            assert len(user.chat_sessions) == 1
            assert user.chat_sessions[0].id == session.id
            assert len(user.chat_messages) == 1
            assert user.chat_messages[0].id == message.id


class TestChatSessionModel:
    """Test ChatSession model"""
    
    def test_create_chat_session(self, app, sample_user):
        """Test creating a new chat session"""
        with app.app_context():
            session = ChatSession(
                user_id=sample_user.id,
                session_name='Test Session'
            )
            db.session.add(session)
            db.session.commit()
            
            assert session.id is not None
            assert session.user_id == sample_user.id
            assert session.session_name == 'Test Session'
            assert session.created_at is not None
            assert session.updated_at is not None
    
    def test_chat_session_default_name(self, app, sample_user):
        """Test chat session with default name"""
        with app.app_context():
            session = ChatSession(user_id=sample_user.id)
            
            assert 'Chat' in session.session_name
            assert datetime.now().strftime('%Y-%m-%d') in session.session_name
    
    def test_chat_session_message_count(self, app, sample_user):
        """Test getting message count for session"""
        with app.app_context():
            session = ChatSession(user_id=sample_user.id, session_name='Test Session')
            db.session.add(session)
            db.session.commit()
            
            # Initially no messages
            assert session.get_message_count() == 0
            
            # Add messages
            for i in range(3):
                message = ChatMessage(
                    session_id=session.id,
                    user_id=sample_user.id,
                    message_type='user',
                    content=f'Message {i}'
                )
                db.session.add(message)
            db.session.commit()
            
            assert session.get_message_count() == 3
    
    def test_chat_session_last_message_time(self, app, sample_user):
        """Test getting last message time for session"""
        with app.app_context():
            session = ChatSession(user_id=sample_user.id, session_name='Test Session')
            db.session.add(session)
            db.session.commit()
            
            # Initially should return updated_at
            last_time = session.get_last_message_time()
            assert last_time == session.updated_at
            
            # Add message
            message = ChatMessage(
                session_id=session.id,
                user_id=sample_user.id,
                message_type='user',
                content='Test message'
            )
            db.session.add(message)
            db.session.commit()
            
            # Should now return message timestamp
            last_time = session.get_last_message_time()
            assert last_time == message.timestamp
    
    def test_chat_session_to_dict(self, app, sample_user):
        """Test chat session serialization to dictionary"""
        with app.app_context():
            session = ChatSession(user_id=sample_user.id, session_name='Test Session')
            db.session.add(session)
            db.session.commit()
            
            session_dict = session.to_dict()
            
            assert session_dict['id'] == session.id
            assert session_dict['user_id'] == sample_user.id
            assert session_dict['name'] == 'Test Session'
            assert 'created_at' in session_dict
            assert 'updated_at' in session_dict
            assert 'message_count' in session_dict
            assert 'last_message_time' in session_dict


class TestChatMessageModel:
    """Test ChatMessage model"""
    
    def test_create_chat_message(self, app, sample_user, sample_chat_session):
        """Test creating a new chat message"""
        with app.app_context():
            message = ChatMessage(
                session_id=sample_chat_session.id,
                user_id=sample_user.id,
                message_type='user',
                content='Test message content',
                model_used='test-model',
                response_time=1.5,
                token_count=10
            )
            db.session.add(message)
            db.session.commit()
            
            assert message.id is not None
            assert message.session_id == sample_chat_session.id
            assert message.user_id == sample_user.id
            assert message.message_type == 'user'
            assert message.content == 'Test message content'
            assert message.model_used == 'test-model'
            assert message.response_time == 1.5
            assert message.token_count == 10
            assert message.timestamp is not None
    
    def test_chat_message_types(self, app, sample_user, sample_chat_session):
        """Test different message types"""
        with app.app_context():
            # User message
            user_msg = ChatMessage(
                session_id=sample_chat_session.id,
                user_id=sample_user.id,
                message_type='user',
                content='User message'
            )
            db.session.add(user_msg)
            
            # Bot message
            bot_msg = ChatMessage(
                session_id=sample_chat_session.id,
                user_id=sample_user.id,
                message_type='bot',
                content='Bot response'
            )
            db.session.add(bot_msg)
            
            db.session.commit()
            
            assert user_msg.message_type == 'user'
            assert bot_msg.message_type == 'bot'
    
    def test_chat_message_to_dict(self, app, sample_user, sample_chat_session):
        """Test chat message serialization to dictionary"""
        with app.app_context():
            message = ChatMessage(
                session_id=sample_chat_session.id,
                user_id=sample_user.id,
                message_type='user',
                content='Test message',
                model_used='test-model',
                response_time=1.0,
                token_count=5
            )
            db.session.add(message)
            db.session.commit()
            
            message_dict = message.to_dict()
            
            assert message_dict['id'] == message.id
            assert message_dict['session_id'] == sample_chat_session.id
            assert message_dict['user_id'] == sample_user.id
            assert message_dict['type'] == 'user'
            assert message_dict['content'] == 'Test message'
            assert message_dict['model_used'] == 'test-model'
            assert message_dict['response_time'] == 1.0
            assert message_dict['token_count'] == 5
            assert 'timestamp' in message_dict


class TestUserSessionModel:
    """Test UserSession model"""
    
    def test_create_user_session(self, app, sample_user):
        """Test creating a new user session"""
        with app.app_context():
            expires_at = datetime.utcnow() + timedelta(hours=24)
            session = UserSession(
                user_id=sample_user.id,
                token_hash='test-token-hash',
                expires_at=expires_at,
                ip_address='127.0.0.1',
                user_agent='Test User Agent'
            )
            db.session.add(session)
            db.session.commit()
            
            assert session.id is not None
            assert session.user_id == sample_user.id
            assert session.token_hash == 'test-token-hash'
            assert session.expires_at == expires_at
            assert session.ip_address == '127.0.0.1'
            assert session.user_agent == 'Test User Agent'
            assert session.is_active is True
            assert session.created_at is not None
    
    def test_user_session_expiry(self, app, sample_user):
        """Test user session expiry checking"""
        with app.app_context():
            # Create expired session
            expired_session = UserSession(
                user_id=sample_user.id,
                token_hash='expired-token',
                expires_at=datetime.utcnow() - timedelta(hours=1)
            )
            
            # Create valid session
            valid_session = UserSession(
                user_id=sample_user.id,
                token_hash='valid-token',
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            
            assert expired_session.is_expired() is True
            assert valid_session.is_expired() is False
    
    def test_user_session_update_activity(self, app, sample_user):
        """Test updating session activity"""
        with app.app_context():
            session = UserSession(
                user_id=sample_user.id,
                token_hash='test-token',
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            db.session.add(session)
            db.session.commit()
            
            original_activity = session.last_activity
            session.update_activity()
            
            assert session.last_activity != original_activity
    
    def test_user_session_deactivate(self, app, sample_user):
        """Test deactivating a session"""
        with app.app_context():
            session = UserSession(
                user_id=sample_user.id,
                token_hash='test-token',
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            db.session.add(session)
            db.session.commit()
            
            assert session.is_active is True
            session.deactivate()
            assert session.is_active is False


class TestDatabaseQueries:
    """Test DatabaseQueries helper class"""
    
    def test_get_user_by_username_or_email(self, app, sample_user):
        """Test getting user by username or email"""
        with app.app_context():
            # Test by username
            user_by_username = DatabaseQueries.get_user_by_username_or_email(sample_user.username)
            assert user_by_username.id == sample_user.id
            
            # Test by email
            user_by_email = DatabaseQueries.get_user_by_username_or_email(sample_user.email)
            assert user_by_email.id == sample_user.id
            
            # Test non-existent user
            non_existent = DatabaseQueries.get_user_by_username_or_email('nonexistent')
            assert non_existent is None
    
    def test_get_user_sessions(self, app, sample_user):
        """Test getting user's chat sessions"""
        with app.app_context():
            # Create multiple sessions
            sessions = []
            for i in range(3):
                session = ChatSession(
                    user_id=sample_user.id,
                    session_name=f'Session {i}'
                )
                db.session.add(session)
                sessions.append(session)
            db.session.commit()
            
            # Get sessions
            user_sessions = DatabaseQueries.get_user_sessions(sample_user.id, limit=2)
            
            assert len(user_sessions) == 2
            # Should be ordered by updated_at desc
            assert user_sessions[0].updated_at >= user_sessions[1].updated_at
    
    def test_get_session_messages(self, app, sample_user, sample_chat_session):
        """Test getting messages for a session"""
        with app.app_context():
            # Create multiple messages
            messages = []
            for i in range(3):
                message = ChatMessage(
                    session_id=sample_chat_session.id,
                    user_id=sample_user.id,
                    message_type='user',
                    content=f'Message {i}'
                )
                db.session.add(message)
                messages.append(message)
            db.session.commit()
            
            # Get messages
            session_messages = DatabaseQueries.get_session_messages(sample_chat_session.id)
            
            assert len(session_messages) == 3
            # Should be ordered by timestamp asc
            for i in range(2):
                assert session_messages[i].timestamp <= session_messages[i + 1].timestamp
    
    def test_get_recent_messages(self, app, sample_user, sample_chat_session):
        """Test getting recent messages for context"""
        with app.app_context():
            # Create multiple messages
            for i in range(5):
                message = ChatMessage(
                    session_id=sample_chat_session.id,
                    user_id=sample_user.id,
                    message_type='user',
                    content=f'Message {i}'
                )
                db.session.add(message)
            db.session.commit()
            
            # Get recent messages
            recent_messages = DatabaseQueries.get_recent_messages(sample_chat_session.id, limit=3)
            
            assert len(recent_messages) == 3
            # Should be ordered by timestamp desc (most recent first)
            for i in range(2):
                assert recent_messages[i].timestamp >= recent_messages[i + 1].timestamp
    
    def test_cleanup_expired_sessions(self, app, sample_user):
        """Test cleaning up expired user sessions"""
        with app.app_context():
            # Create expired session
            expired_session = UserSession(
                user_id=sample_user.id,
                token_hash='expired-token',
                expires_at=datetime.utcnow() - timedelta(hours=1)
            )
            db.session.add(expired_session)
            
            # Create valid session
            valid_session = UserSession(
                user_id=sample_user.id,
                token_hash='valid-token',
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            db.session.add(valid_session)
            db.session.commit()
            
            # Cleanup expired sessions
            count = DatabaseQueries.cleanup_expired_sessions()
            
            assert count == 1
            
            # Verify only valid session remains
            remaining_sessions = UserSession.query.all()
            assert len(remaining_sessions) == 1
            assert remaining_sessions[0].id == valid_session.id
    
    def test_get_user_stats(self, app, sample_user, sample_chat_session, sample_messages):
        """Test getting user statistics"""
        with app.app_context():
            stats = DatabaseQueries.get_user_stats(sample_user.id)
            
            assert stats is not None
            assert stats['user_id'] == sample_user.id
            assert stats['username'] == sample_user.username
            assert stats['total_sessions'] == 1
            assert stats['total_messages'] == 2  # From sample_messages fixture
            assert stats['user_messages'] == 1
            assert stats['bot_messages'] == 1
            assert 'member_since' in stats
    
    def test_get_user_stats_nonexistent(self, app):
        """Test getting stats for non-existent user"""
        with app.app_context():
            stats = DatabaseQueries.get_user_stats(99999)
            assert stats is None