"""
Database models for AI Chatbot application
"""

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import uuid

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication and profile management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    bio = db.Column(db.Text)
    avatar = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    # Relationships
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')
    chat_messages = db.relationship('ChatMessage', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def to_dict(self):
        """Convert user to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'firstName': self.first_name,
            'lastName': self.last_name,
            'bio': self.bio,
            'avatar': self.avatar,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    def __repr__(self):
        return f'<User {self.username}>'

class ChatSession(db.Model):
    """Chat session model for organizing conversations"""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    session_name = db.Column(db.String(255))
    model_name = db.Column(db.String(100), default='llama2')  # Current model for this session
    system_prompt_type = db.Column(db.String(50), default='general')  # System prompt type
    custom_system_prompt = db.Column(db.Text)  # Custom system prompt
    model_parameters = db.Column(db.JSON)  # Model parameters as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, user_id, session_name=None):
        self.user_id = user_id
        self.session_name = session_name or f'Chat {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}'
    
    def get_message_count(self):
        """Get total number of messages in this session"""
        return ChatMessage.query.filter_by(session_id=self.id).count()
    
    def get_last_message_time(self):
        """Get timestamp of the last message in this session"""
        last_message = ChatMessage.query.filter_by(session_id=self.id)\
                                       .order_by(ChatMessage.timestamp.desc())\
                                       .first()
        return last_message.timestamp if last_message else self.updated_at
    
    def to_dict(self):
        """Convert session to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.session_name,
            'model_name': self.model_name,
            'system_prompt_type': self.system_prompt_type,
            'custom_system_prompt': self.custom_system_prompt,
            'model_parameters': self.model_parameters,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'message_count': self.get_message_count(),
            'last_message_time': self.get_last_message_time().isoformat()
        }
    
    def __repr__(self):
        return f'<ChatSession {self.id}: {self.session_name}>'

class ChatMessage(db.Model):
    """Chat message model for storing conversation history"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    message_type = db.Column(db.Enum('user', 'bot', name='message_type_enum'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Additional metadata
    model_used = db.Column(db.String(100))  # AI model used for bot responses
    response_time = db.Column(db.Float)     # Response time in seconds
    token_count = db.Column(db.Integer)     # Token count for the message
    
    def __init__(self, session_id, user_id, message_type, content, **kwargs):
        self.session_id = session_id
        self.user_id = user_id
        self.message_type = message_type
        self.content = content
        self.model_used = kwargs.get('model_used')
        self.response_time = kwargs.get('response_time')
        self.token_count = kwargs.get('token_count')
    
    def to_dict(self):
        """Convert message to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'type': self.message_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'model_used': self.model_used,
            'response_time': self.response_time,
            'token_count': self.token_count
        }
    
    def __repr__(self):
        return f'<ChatMessage {self.id}: {self.message_type}>'

class UserSession(db.Model):
    """User session model for tracking active sessions"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    token_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ip_address = db.Column(db.String(45))  # IPv6 compatible
    user_agent = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    # Relationship
    user = db.relationship('User', backref='sessions')
    
    def __init__(self, user_id, token_hash, expires_at, ip_address=None, user_agent=None):
        self.user_id = user_id
        self.token_hash = token_hash
        self.expires_at = expires_at
        self.ip_address = ip_address
        self.user_agent = user_agent
    
    def is_expired(self):
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
        db.session.commit()
    
    def deactivate(self):
        """Deactivate the session"""
        self.is_active = False
        db.session.commit()
    
    def to_dict(self):
        """Convert session to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'ip_address': self.ip_address,
            'is_active': self.is_active,
            'is_expired': self.is_expired()
        }
    
    def __repr__(self):
        return f'<UserSession {self.id}: User {self.user_id}>'

# Database utility functions
def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    
def create_tables():
    """Create all database tables"""
    db.create_all()

def drop_tables():
    """Drop all database tables"""
    db.drop_all()

def reset_db():
    """Reset database (drop and recreate all tables)"""
    drop_tables()
    create_tables()

# Query helpers
class DatabaseQueries:
    """Helper class for common database queries"""
    
    @staticmethod
    def get_user_by_username_or_email(identifier):
        """Get user by username or email"""
        return User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).filter(User.is_active == True).first()
    
    @staticmethod
    def get_user_sessions(user_id, limit=10):
        """Get user's chat sessions ordered by last activity"""
        return ChatSession.query.filter_by(user_id=user_id)\
                               .order_by(ChatSession.updated_at.desc())\
                               .limit(limit).all()
    
    @staticmethod
    def get_session_messages(session_id, limit=None):
        """Get messages for a session ordered by timestamp"""
        query = ChatMessage.query.filter_by(session_id=session_id)\
                                 .order_by(ChatMessage.timestamp.asc())
        if limit:
            query = query.limit(limit)
        return query.all()
    
    @staticmethod
    def get_recent_messages(session_id, limit=10):
        """Get recent messages for context"""
        return ChatMessage.query.filter_by(session_id=session_id)\
                                .order_by(ChatMessage.timestamp.desc())\
                                .limit(limit).all()
    
    @staticmethod
    def cleanup_expired_sessions():
        """Remove expired user sessions"""
        expired_sessions = UserSession.query.filter(
            UserSession.expires_at < datetime.utcnow()
        ).all()
        
        for session in expired_sessions:
            db.session.delete(session)
        
        db.session.commit()
        return len(expired_sessions)
    
    @staticmethod
    def get_user_stats(user_id):
        """Get user statistics"""
        user = User.query.get(user_id)
        if not user:
            return None
        
        total_sessions = ChatSession.query.filter_by(user_id=user_id).count()
        total_messages = ChatMessage.query.filter_by(user_id=user_id).count()
        user_messages = ChatMessage.query.filter_by(user_id=user_id, message_type='user').count()
        bot_messages = ChatMessage.query.filter_by(user_id=user_id, message_type='bot').count()
        
        return {
            'user_id': user_id,
            'username': user.username,
            'total_sessions': total_sessions,
            'total_messages': total_messages,
            'user_messages': user_messages,
            'bot_messages': bot_messages,
            'member_since': user.created_at.isoformat() if user.created_at else None,
            'last_login': user.last_login.isoformat() if user.last_login else None
        }