#!/usr/bin/env python3
"""
AI Chatbot Web Application
A responsive, mobile-ready web GUI for AI chatbot service with authentication and chat history
"""

from flask import Flask, render_template, request, jsonify, session, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from functools import wraps
import requests
import json
import uuid
import os
from datetime import datetime, timedelta
import jwt
import hashlib
from pathlib import Path
from models import db, User, ChatSession, ChatMessage, UserSession, DatabaseQueries, init_db
from services.ollama_service import ollama_service
from services.huggingface_service import hf_service
from services.chat_service import chat_service
from services.model_manager import model_manager
from services.rag_service import rag_service
from services.embeddings_service import embeddings_service
from services.fine_tuning_service import fine_tuning_service

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-string-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=int(os.environ.get('JWT_EXPIRES_HOURS', 24)))
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=int(os.environ.get('JWT_REFRESH_EXPIRES_DAYS', 30)))

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    # Production database (PostgreSQL)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
else:
    # Development database (PostgreSQL or SQLite fallback)
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', 'chatbot_db')
    DB_USER = os.environ.get('DB_USER', 'chatbot_user')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'chatbot_password')
    
    app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# Initialize database
init_db(app)
migrate = Migrate(app, db)

# Enable CORS for React frontend
CORS(app, origins=["http://localhost:3000", "http://localhost:80"], supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000", "http://localhost:80"], supports_credentials=True)

# AI Configuration
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'llama2')

class ChatbotService:
    """Service class to handle AI chatbot interactions"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
    
    def generate_response(self, message, conversation_history=None):
        """Generate AI response using Ollama API"""
        try:
            # Prepare the prompt with conversation history
            prompt = self._build_prompt(message, conversation_history)
            
            # Make request to Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", "Sorry, I couldn't generate a response."),
                    "model": self.model
                }
            else:
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Connection Error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected Error: {str(e)}"
            }
    
    def _build_prompt(self, message, conversation_history=None):
        """Build prompt with conversation context"""
        if not conversation_history:
            return message
        
        # Build conversation context
        context = "Previous conversation:\n"
        # Reverse the list since we get recent messages in desc order
        for msg in reversed(conversation_history[-5:]):  # Last 5 messages for context
            if hasattr(msg, 'message_type'):
                role = "Human" if msg.message_type == "user" else "Assistant"
                content = msg.content
            else:
                role = "Human" if msg.get("type") == "user" else "Assistant"
                content = msg.get('content', '')
            context += f"{role}: {content}\n"
        
        return f"{context}\nHuman: {message}\nAssistant:"

# Database initialization
def create_tables():
    """Create all database tables"""
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully")

# Authentication helpers
def token_required(f):
    """Decorator to require JWT token for protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user_id, *args, **kwargs)
    return decorated

def get_user_by_id(user_id):
    """Get user information by ID"""
    user = User.query.filter_by(id=user_id, is_active=True).first()
    return user.to_dict() if user else None

def save_chat_message(user_id, session_id, message_type, content, **kwargs):
    """Save a chat message to the database"""
    try:
        message = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            message_type=message_type,
            content=content,
            **kwargs
        )
        db.session.add(message)
        
        # Update session timestamp
        session = ChatSession.query.get(session_id)
        if session:
            session.updated_at = datetime.utcnow()
        
        db.session.commit()
        return message.id
    except Exception as e:
        db.session.rollback()
        raise e

# Initialize chatbot service
chatbot = ChatbotService()

@app.route('/')
def index():
    """Serve the main chat interface"""
    # Generate unique session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html', session_id=session['session_id'])

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        first_name = data.get('firstName', '').strip()
        last_name = data.get('lastName', '').strip()
        
        if not username or not email or not password:
            return jsonify({'message': 'Username, email, and password are required'}), 400
        
        if len(password) < 8:
            return jsonify({'message': 'Password must be at least 8 characters long'}), 400
        
        if len(username) < 3:
            return jsonify({'message': 'Username must be at least 3 characters long'}), 400
        
        # Validate email format
        import re
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return jsonify({'message': 'Please enter a valid email address'}), 400
        
        # Check if user already exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            if existing_user.username == username:
                return jsonify({'message': 'Username already exists'}), 400
            else:
                return jsonify({'message': 'Email already exists'}), 400
        
        # Create new user
        user = User(username=username, email=email, password=password)
        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        
        db.session.add(user)
        db.session.commit()
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        # Create user session record
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        user_session = UserSession(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES'],
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        db.session.add(user_session)
        db.session.commit()
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'message': 'Username and password are required'}), 400
        
        # Find user by username or email
        user = DatabaseQueries.get_user_by_username_or_email(username)
        
        if not user or not user.check_password(password):
            return jsonify({'message': 'Invalid credentials'}), 401
        
        # Update last login
        user.update_last_login()
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        # Create user session record
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        user_session = UserSession(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES'],
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        db.session.add(user_session)
        db.session.commit()
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(current_user_id):
    """Get current user information"""
    user = get_user_by_id(current_user_id)
    if user:
        return jsonify({'user': user}), 200
    return jsonify({'message': 'User not found'}), 404

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user_id):
    """User logout endpoint (client-side token removal)"""
    return jsonify({'message': 'Logout successful'}), 200

@app.route('/api/auth/refresh', methods=['POST'])
@token_required
def refresh_token(current_user_id):
    """Refresh JWT token"""
    try:
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Generate new JWT token
        new_token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        # Update user session with new token
        current_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        current_token_hash = hashlib.sha256(current_token.encode()).hexdigest()
        
        user_session = UserSession.query.filter_by(
            user_id=current_user_id,
            token_hash=current_token_hash,
            is_active=True
        ).first()
        
        if user_session:
            # Update existing session with new token
            new_token_hash = hashlib.sha256(new_token.encode()).hexdigest()
            user_session.token_hash = new_token_hash
            user_session.expires_at = datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
            user_session.last_activity = datetime.utcnow()
            db.session.commit()
        else:
            # Create new session if not found
            new_token_hash = hashlib.sha256(new_token.encode()).hexdigest()
            user_session = UserSession(
                user_id=current_user_id,
                token_hash=new_token_hash,
                expires_at=datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES'],
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
            db.session.add(user_session)
            db.session.commit()
        
        return jsonify({
            'message': 'Token refreshed successfully',
            'token': new_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Token refresh failed: {str(e)}'}), 500

@app.route('/api/auth/profile', methods=['PUT'])
@token_required
def update_profile(current_user_id):
    """Update user profile"""
    try:
        data = request.get_json()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Update profile fields
        if 'firstName' in data:
            user.first_name = data['firstName'].strip()
        if 'lastName' in data:
            user.last_name = data['lastName'].strip()
        if 'email' in data:
            email = data['email'].strip()
            # Check if email is already taken by another user
            existing_user = User.query.filter(User.email == email, User.id != current_user_id).first()
            if existing_user:
                return jsonify({'message': 'Email is already taken'}), 400
            user.email = email
        if 'username' in data:
            username = data['username'].strip()
            # Check if username is already taken by another user
            existing_user = User.query.filter(User.username == username, User.id != current_user_id).first()
            if existing_user:
                return jsonify({'message': 'Username is already taken'}), 400
            user.username = username
        if 'bio' in data:
            user.bio = data.get('bio', '')
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Failed to update profile: {str(e)}'}), 500

@app.route('/api/auth/password', methods=['PUT'])
@token_required
def change_password(current_user_id):
    """Change user password"""
    try:
        data = request.get_json()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')
        
        if not current_password or not new_password:
            return jsonify({'message': 'Current password and new password are required'}), 400
        
        # Verify current password
        if not user.check_password(current_password):
            return jsonify({'message': 'Current password is incorrect'}), 400
        
        # Validate new password
        if len(new_password) < 8:
            return jsonify({'message': 'New password must be at least 8 characters long'}), 400
        
        # Update password
        user.set_password(new_password)
        db.session.commit()
        
        return jsonify({'message': 'Password changed successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Failed to change password: {str(e)}'}), 500

@app.route('/api/auth/stats', methods=['GET'])
@token_required
def get_user_stats(current_user_id):
    """Get user statistics"""
    try:
        stats = DatabaseQueries.get_user_stats(current_user_id)
        if not stats:
            return jsonify({'message': 'User not found'}), 404
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get user stats: {str(e)}'}), 500

@app.route('/api/auth/sessions', methods=['GET'])
@token_required
def get_user_sessions(current_user_id):
    """Get user's active sessions"""
    try:
        sessions = UserSession.query.filter_by(user_id=current_user_id, is_active=True)\
                                   .order_by(UserSession.last_activity.desc()).all()
        
        sessions_data = []
        current_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        current_token_hash = hashlib.sha256(current_token.encode()).hexdigest() if current_token else None
        
        for session in sessions:
            session_dict = session.to_dict()
            session_dict['isCurrent'] = session.token_hash == current_token_hash
            sessions_data.append(session_dict)
        
        return jsonify({'sessions': sessions_data}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get user sessions: {str(e)}'}), 500

@app.route('/api/auth/logout-all', methods=['POST'])
@token_required
def logout_all_sessions(current_user_id):
    """Logout from all sessions"""
    try:
        # Deactivate all user sessions
        UserSession.query.filter_by(user_id=current_user_id).update({'is_active': False})
        db.session.commit()
        
        return jsonify({'message': 'Logged out from all sessions successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Failed to logout from all sessions: {str(e)}'}), 500

@app.route('/api/auth/account', methods=['DELETE'])
@token_required
def delete_account(current_user_id):
    """Delete user account"""
    try:
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Delete user (cascade will handle sessions, messages, etc.)
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'Account deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Failed to delete account: {str(e)}'}), 500

@app.route('/api/auth/check-username/<username>', methods=['GET'])
def check_username_availability(username):
    """Check if username is available"""
    try:
        if len(username) < 3:
            return jsonify({'available': False, 'message': 'Username must be at least 3 characters long'}), 400
        
        existing_user = User.query.filter_by(username=username).first()
        available = existing_user is None
        
        return jsonify({'available': available}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to check username availability: {str(e)}'}), 500

@app.route('/api/auth/check-email/<email>', methods=['GET'])
def check_email_availability(email):
    """Check if email is available"""
    try:
        import re
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return jsonify({'available': False, 'message': 'Invalid email format'}), 400
        
        existing_user = User.query.filter_by(email=email).first()
        available = existing_user is None
        
        return jsonify({'available': available}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to check email availability: {str(e)}'}), 500

# Chat routes
@app.route('/api/chat', methods=['POST'])
@token_required
def chat_api(current_user_id):
    """REST API endpoint for chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not message:
            return jsonify({
                "success": False,
                "error": "Message cannot be empty"
            }), 400
        
        # Create new session if not provided
        if not session_id:
            session = ChatSession(user_id=current_user_id)
            db.session.add(session)
            db.session.commit()
            session_id = session.id
        
        # Save user message
        save_chat_message(current_user_id, session_id, 'user', message)
        
        # Get conversation history for context
        conversation_history = DatabaseQueries.get_recent_messages(session_id, limit=10)
        
        # Generate AI response
        result = chatbot.generate_response(message, conversation_history)
        
        if result['success']:
            # Save bot response
            save_chat_message(current_user_id, session_id, 'bot', result['response'])
        
        # Add timestamp and session info
        result['timestamp'] = datetime.now().isoformat()
        result['session_id'] = session_id
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server Error: {str(e)}"
        }), 500

@app.route('/api/chat/sessions', methods=['GET'])
@token_required
def get_chat_sessions(current_user_id):
    """Get user's chat sessions"""
    try:
        sessions = DatabaseQueries.get_user_sessions(current_user_id, limit=50)
        sessions_data = [session.to_dict() for session in sessions]
        return jsonify({'sessions': sessions_data}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get sessions: {str(e)}'}), 500

@app.route('/api/chat/sessions', methods=['POST'])
@token_required
def create_chat_session_api(current_user_id):
    """Create a new chat session"""
    try:
        data = request.get_json() or {}
        session_name = data.get('name', '')
        
        session = ChatSession(user_id=current_user_id, session_name=session_name)
        db.session.add(session)
        db.session.commit()
        
        return jsonify({
            'message': 'Session created successfully',
            'session_id': session.id,
            'session': session.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Failed to create session: {str(e)}'}), 500

@app.route('/api/chat/sessions/<int:session_id>', methods=['GET'])
@token_required
def get_session_messages_api(current_user_id, session_id):
    """Get messages for a specific session"""
    try:
        # Verify session belongs to user
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
        
        if not session:
            return jsonify({'message': 'Session not found'}), 404
        
        messages = DatabaseQueries.get_session_messages(session_id)
        messages_data = [message.to_dict() for message in messages]
        
        return jsonify({
            'messages': messages_data,
            'session': session.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get messages: {str(e)}'}), 500

@app.route('/api/chat/sessions/<int:session_id>', methods=['PUT'])
@token_required
def update_chat_session(current_user_id, session_id):
    """Update a chat session (rename)"""
    try:
        # Verify session belongs to user
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
        
        if not session:
            return jsonify({'message': 'Session not found'}), 404
        
        data = request.get_json()
        new_name = data.get('name', '').strip()
        
        if not new_name:
            return jsonify({'message': 'Session name cannot be empty'}), 400
        
        session.session_name = new_name
        session.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Session updated successfully',
            'session': session.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Failed to update session: {str(e)}'}), 500

@app.route('/api/chat/sessions/<int:session_id>', methods=['DELETE'])
@token_required
def delete_chat_session(current_user_id, session_id):
    """Delete a chat session"""
    try:
        # Verify session belongs to user
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
        
        if not session:
            return jsonify({'message': 'Session not found'}), 404
        
        # Delete session (messages will be deleted automatically due to cascade)
        db.session.delete(session)
        db.session.commit()
        
        return jsonify({'message': 'Session deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Failed to delete session: {str(e)}'}), 500

# Enhanced Model Management APIs
@app.route('/api/models', methods=['GET'])
@token_required
def get_available_models(current_user_id):
    """Get list of available models"""
    try:
        models = ollama_service.list_models()
        models_data = []
        
        for model in models:
            model_dict = {
                'name': model.name,
                'size': model.size,
                'modified': model.modified.isoformat(),
                'digest': model.digest,
                'details': model.details
            }
            
            # Add usage statistics if available
            usage_stats = model_manager.get_model_usage_stats(model.name)
            if usage_stats:
                model_dict['usage_stats'] = usage_stats
            
            models_data.append(model_dict)
        
        return jsonify({
            'models': models_data,
            'total_count': len(models_data)
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get models: {str(e)}'}), 500

@app.route('/api/models/<model_name>', methods=['GET'])
@token_required
def get_model_info(current_user_id, model_name):
    """Get detailed information about a specific model"""
    try:
        model_info = ollama_service.get_model_info(model_name)
        if not model_info:
            return jsonify({'message': 'Model not found'}), 404
        
        model_data = {
            'name': model_info.name,
            'size': model_info.size,
            'modified': model_info.modified.isoformat(),
            'digest': model_info.digest,
            'details': model_info.details,
            'parameters': model_info.parameters,
            'template': model_info.template,
            'system': model_info.system
        }
        
        # Add usage statistics
        usage_stats = model_manager.get_model_usage_stats(model_name)
        if usage_stats:
            model_data['usage_stats'] = usage_stats
        
        return jsonify({'model': model_data}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/models/pull', methods=['POST'])
@token_required
def pull_model(current_user_id):
    """Pull/download a new model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'message': 'Model name is required'}), 400
        
        # Start model pull in background
        def pull_progress():
            for progress in ollama_service.pull_model(model_name):
                # In a real implementation, you'd use WebSocket or SSE for real-time updates
                yield f"data: {json.dumps(progress)}\n\n"
        
        return jsonify({
            'message': f'Started pulling model {model_name}',
            'model_name': model_name
        }), 202
        
    except Exception as e:
        return jsonify({'message': f'Failed to pull model: {str(e)}'}), 500

@app.route('/api/models/<model_name>', methods=['DELETE'])
@token_required
def delete_model(current_user_id, model_name):
    """Delete a model"""
    try:
        success = ollama_service.delete_model(model_name)
        if success:
            return jsonify({'message': f'Model {model_name} deleted successfully'}), 200
        else:
            return jsonify({'message': f'Failed to delete model {model_name}'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to delete model: {str(e)}'}), 500

@app.route('/api/models/recommendations', methods=['GET'])
@token_required
def get_model_recommendations(current_user_id):
    """Get model recommendations based on use case"""
    try:
        use_case = request.args.get('use_case', 'general_chat')
        max_memory = float(request.args.get('max_memory', 8.0))
        
        recommendations = model_manager.get_model_recommendations(use_case)
        optimal_model = model_manager.get_optimal_model(use_case, max_memory)
        
        return jsonify({
            'use_case': use_case,
            'recommendations': recommendations,
            'optimal_model': optimal_model,
            'available_use_cases': list(model_manager.model_recommendations.keys())
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get recommendations: {str(e)}'}), 500

# Enhanced Chat APIs with Streaming
@app.route('/api/chat/stream', methods=['POST'])
@token_required
def chat_stream(current_user_id):
    """Stream chat responses in real-time"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        model = data.get('model')
        parameters = data.get('parameters', {})
        
        if not message:
            return jsonify({'message': 'Message cannot be empty'}), 400
        
        # Create new session if not provided
        if not session_id:
            session = ChatSession(user_id=current_user_id)
            db.session.add(session)
            db.session.commit()
            session_id = session.id
        
        def generate_stream():
            try:
                start_time = time.time()
                total_content = ""
                
                for response in chat_service.generate_response(
                    session_id=session_id,
                    user_id=current_user_id,
                    message=message,
                    model=model,
                    parameters=parameters,
                    stream=True
                ):
                    total_content += response.content
                    
                    # Record usage statistics
                    if response.is_complete:
                        response_time = time.time() - start_time
                        model_manager.record_model_usage(
                            model_name=response.metadata.get('model', 'unknown'),
                            response_time=response_time,
                            token_count=response.metadata.get('eval_count', 0),
                            success=response.error is None
                        )
                    
                    yield f"data: {json.dumps({
                        'content': response.content,
                        'is_complete': response.is_complete,
                        'metadata': response.metadata,
                        'error': response.error,
                        'session_id': session_id
                    })}\n\n"
                    
                    if response.is_complete:
                        break
                        
            except Exception as e:
                yield f"data: {json.dumps({
                    'content': '',
                    'is_complete': True,
                    'error': str(e),
                    'session_id': session_id
                })}\n\n"
        
        return Response(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        return jsonify({'message': f'Failed to start chat stream: {str(e)}'}), 500

@app.route('/api/chat/sessions/<int:session_id>/model', methods=['PUT'])
@token_required
def switch_session_model(current_user_id, session_id):
    """Switch model for a chat session"""
    try:
        # Verify session belongs to user
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
        if not session:
            return jsonify({'message': 'Session not found'}), 404
        
        data = request.get_json()
        new_model = data.get('model')
        
        if not new_model:
            return jsonify({'message': 'Model name is required'}), 400
        
        success = chat_service.switch_model(session_id, new_model)
        if success:
            return jsonify({
                'message': f'Switched to model {new_model}',
                'session_id': session_id,
                'model': new_model
            }), 200
        else:
            return jsonify({'message': 'Failed to switch model'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to switch model: {str(e)}'}), 500

@app.route('/api/chat/sessions/<int:session_id>/parameters', methods=['PUT'])
@token_required
def update_session_parameters(current_user_id, session_id):
    """Update model parameters for a chat session"""
    try:
        # Verify session belongs to user
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
        if not session:
            return jsonify({'message': 'Session not found'}), 404
        
        data = request.get_json()
        parameters = data.get('parameters', {})
        
        success = chat_service.update_model_parameters(session_id, parameters)
        if success:
            return jsonify({
                'message': 'Parameters updated successfully',
                'session_id': session_id,
                'parameters': parameters
            }), 200
        else:
            return jsonify({'message': 'Failed to update parameters'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to update parameters: {str(e)}'}), 500

@app.route('/api/chat/sessions/<int:session_id>/system-prompt', methods=['PUT'])
@token_required
def update_session_system_prompt(current_user_id, session_id):
    """Update system prompt for a chat session"""
    try:
        # Verify session belongs to user
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user_id).first()
        if not session:
            return jsonify({'message': 'Session not found'}), 404
        
        data = request.get_json()
        prompt_type = data.get('prompt_type')
        custom_prompt = data.get('custom_prompt')
        
        success = chat_service.update_system_prompt(session_id, prompt_type, custom_prompt)
        if success:
            return jsonify({
                'message': 'System prompt updated successfully',
                'session_id': session_id,
                'prompt_type': prompt_type
            }), 200
        else:
            return jsonify({'message': 'Failed to update system prompt'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to update system prompt: {str(e)}'}), 500

# System Monitoring and Analytics APIs
@app.route('/api/system/status', methods=['GET'])
@token_required
def get_system_status(current_user_id):
    """Get comprehensive system status"""
    try:
        status = model_manager.get_system_status()
        return jsonify(status), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get system status: {str(e)}'}), 500

@app.route('/api/system/performance', methods=['GET'])
@token_required
def get_performance_report(current_user_id):
    """Get detailed performance report"""
    try:
        report = model_manager.get_model_performance_report()
        return jsonify(report), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get performance report: {str(e)}'}), 500

@app.route('/api/system/health', methods=['GET'])
@token_required
def get_health_check(current_user_id):
    """Get system health check"""
    try:
        health_report = model_manager.get_model_health_check()
        return jsonify(health_report), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get health check: {str(e)}'}), 500

@app.route('/api/system/cleanup', methods=['POST'])
@token_required
def cleanup_unused_models(current_user_id):
    """Clean up unused models"""
    try:
        data = request.get_json() or {}
        days_unused = data.get('days_unused', 7)
        
        removed_models = model_manager.cleanup_unused_models(days_unused)
        
        return jsonify({
            'message': f'Cleanup completed',
            'removed_models': removed_models,
            'count': len(removed_models)
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to cleanup models: {str(e)}'}), 500

# Hugging Face Integration APIs
@app.route('/api/huggingface/search', methods=['GET'])
@token_required
def search_huggingface_models(current_user_id):
    """Search for models on Hugging Face"""
    try:
        query = request.args.get('query', '')
        task = request.args.get('task', 'text-generation')
        limit = int(request.args.get('limit', 20))
        
        models = hf_service.search_models(query, task, limit=limit)
        models_data = []
        
        for model in models:
            models_data.append({
                'model_id': model.model_id,
                'author': model.author,
                'model_name': model.model_name,
                'downloads': model.downloads,
                'likes': model.likes,
                'tags': model.tags,
                'pipeline_tag': model.pipeline_tag,
                'description': model.description,
                'compatible_with_ollama': model.compatible_with_ollama
            })
        
        return jsonify({
            'models': models_data,
            'total_count': len(models_data)
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to search models: {str(e)}'}), 500

@app.route('/api/huggingface/recommended', methods=['GET'])
@token_required
def get_recommended_huggingface_models(current_user_id):
    """Get recommended Hugging Face models"""
    try:
        models = hf_service.get_available_models_for_download()
        return jsonify({'models': models}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get recommended models: {str(e)}'}), 500

@app.route('/api/huggingface/download', methods=['POST'])
@token_required
def download_huggingface_model(current_user_id):
    """Download and convert Hugging Face model for Ollama"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({'message': 'Model ID is required'}), 400
        
        # Start download in background
        # In a real implementation, you'd use a task queue like Celery
        def download_progress():
            progress_data = []
            
            def progress_callback(progress):
                progress_data.append({
                    'status': progress.status,
                    'progress': progress.progress,
                    'message': progress.message,
                    'timestamp': datetime.now().isoformat()
                })
            
            final_progress = hf_service.download_model_for_ollama(model_id, progress_callback)
            return final_progress
        
        return jsonify({
            'message': f'Started downloading model {model_id}',
            'model_id': model_id
        }), 202
        
    except Exception as e:
        return jsonify({'message': f'Failed to download model: {str(e)}'}), 500

# RAG (Retrieval-Augmented Generation) APIs
@app.route('/api/rag/ingest', methods=['POST'])
@token_required
def ingest_document(current_user_id):
    """Ingest document into RAG system"""
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            
            # Prepare metadata
            metadata = {
                'uploaded_by': current_user_id,
                'original_filename': file.filename,
                'file_size': os.path.getsize(tmp_file.name)
            }
            
            # Ingest the file
            success = rag_service.ingest_file(tmp_file.name, metadata)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            if success:
                return jsonify({
                    'message': 'Document ingested successfully',
                    'filename': file.filename
                }), 200
            else:
                return jsonify({'message': 'Failed to ingest document'}), 500
                
    except Exception as e:
        return jsonify({'message': f'Failed to ingest document: {str(e)}'}), 500

@app.route('/api/rag/ingest-text', methods=['POST'])
@token_required
def ingest_text(current_user_id):
    """Ingest text into RAG system"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        metadata = data.get('metadata', {})
        
        if not text:
            return jsonify({'message': 'Text content is required'}), 400
        
        # Add user metadata
        metadata['uploaded_by'] = current_user_id
        metadata['source'] = 'text_input'
        
        success = rag_service.ingest_text(text, metadata)
        
        if success:
            return jsonify({'message': 'Text ingested successfully'}), 200
        else:
            return jsonify({'message': 'Failed to ingest text'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to ingest text: {str(e)}'}), 500

@app.route('/api/rag/search', methods=['POST'])
@token_required
def search_documents(current_user_id):
    """Search documents in RAG system"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        max_results = data.get('max_results', 5)
        
        if not query:
            return jsonify({'message': 'Query is required'}), 400
        
        results = rag_service.search_documents(query, max_results)
        
        search_results = []
        for result in results:
            search_results.append({
                'content': result.document.content,
                'metadata': result.document.metadata,
                'score': result.score,
                'rank': result.rank
            })
        
        return jsonify({
            'query': query,
            'results': search_results,
            'total_results': len(search_results)
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to search documents: {str(e)}'}), 500

@app.route('/api/rag/chat', methods=['POST'])
@token_required
def rag_chat(current_user_id):
    """Chat with RAG-enhanced responses"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        model = data.get('model', 'llama2:7b-chat')
        max_sources = data.get('max_sources', 5)
        
        if not query:
            return jsonify({'message': 'Query is required'}), 400
        
        # Generate RAG response
        rag_response = rag_service.generate_rag_response(query, model, max_sources)
        
        # Format sources
        sources = []
        for source in rag_response.sources:
            sources.append({
                'content': source.document.content[:200] + '...' if len(source.document.content) > 200 else source.document.content,
                'metadata': source.document.metadata,
                'score': source.score,
                'rank': source.rank
            })
        
        return jsonify({
            'query': query,
            'answer': rag_response.answer,
            'sources': sources,
            'confidence': rag_response.confidence,
            'model_used': rag_response.model_used,
            'retrieval_time': rag_response.retrieval_time,
            'generation_time': rag_response.generation_time
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to generate RAG response: {str(e)}'}), 500

@app.route('/api/rag/stats', methods=['GET'])
@token_required
def get_rag_stats(current_user_id):
    """Get RAG system statistics"""
    try:
        stats = rag_service.get_stats()
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get RAG stats: {str(e)}'}), 500

@app.route('/api/rag/clear', methods=['POST'])
@token_required
def clear_rag_documents(current_user_id):
    """Clear all documents from RAG system"""
    try:
        success = rag_service.clear_documents()
        
        if success:
            return jsonify({'message': 'RAG documents cleared successfully'}), 200
        else:
            return jsonify({'message': 'Failed to clear RAG documents'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to clear RAG documents: {str(e)}'}), 500

# Embeddings APIs
@app.route('/api/embeddings/models', methods=['GET'])
@token_required
def get_embedding_models(current_user_id):
    """Get available embedding models"""
    try:
        models = embeddings_service.get_available_models()
        return jsonify({'models': models}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get embedding models: {str(e)}'}), 500

@app.route('/api/embeddings/generate', methods=['POST'])
@token_required
def generate_embeddings(current_user_id):
    """Generate embeddings for texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        model_name = data.get('model_name')
        normalize = data.get('normalize', True)
        
        if not texts:
            return jsonify({'message': 'Texts are required'}), 400
        
        if not isinstance(texts, list):
            texts = [texts]
        
        result = embeddings_service.generate_embeddings(texts, model_name, normalize)
        
        return jsonify({
            'embeddings': result.embeddings,
            'model_used': result.model_used,
            'dimension': result.dimension,
            'processing_time': result.processing_time,
            'token_count': result.token_count
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to generate embeddings: {str(e)}'}), 500

@app.route('/api/embeddings/collections', methods=['POST'])
@token_required
def create_embedding_collection(current_user_id):
    """Create a new embedding collection"""
    try:
        data = request.get_json()
        collection_name = data.get('collection_name', '').strip()
        dimension = data.get('dimension')
        
        if not collection_name:
            return jsonify({'message': 'Collection name is required'}), 400
        
        success = embeddings_service.create_search_index(collection_name, dimension)
        
        if success:
            return jsonify({
                'message': f'Collection "{collection_name}" created successfully',
                'collection_name': collection_name
            }), 201
        else:
            return jsonify({'message': 'Failed to create collection'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to create collection: {str(e)}'}), 500

@app.route('/api/embeddings/collections/<collection_name>/add', methods=['POST'])
@token_required
def add_to_embedding_collection(current_user_id, collection_name):
    """Add texts to embedding collection"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        metadata = data.get('metadata', [])
        model_name = data.get('model_name')
        
        if not texts:
            return jsonify({'message': 'Texts are required'}), 400
        
        if not isinstance(texts, list):
            texts = [texts]
        
        success = embeddings_service.add_to_search_index(
            collection_name, texts, None, metadata, model_name
        )
        
        if success:
            return jsonify({
                'message': f'Added {len(texts)} texts to collection "{collection_name}"',
                'count': len(texts)
            }), 200
        else:
            return jsonify({'message': 'Failed to add texts to collection'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to add to collection: {str(e)}'}), 500

@app.route('/api/embeddings/collections/<collection_name>/search', methods=['POST'])
@token_required
def search_embedding_collection(current_user_id, collection_name):
    """Search in embedding collection"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 5)
        threshold = data.get('threshold', 0.0)
        model_name = data.get('model_name')
        
        if not query:
            return jsonify({'message': 'Query is required'}), 400
        
        results = embeddings_service.search_similar(
            collection_name, query, k, threshold, model_name
        )
        
        search_results = []
        for result in results:
            search_results.append({
                'text': result.text,
                'score': result.score,
                'rank': result.rank,
                'metadata': result.metadata
            })
        
        return jsonify({
            'query': query,
            'results': search_results,
            'collection': collection_name
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to search collection: {str(e)}'}), 500

@app.route('/api/embeddings/stats', methods=['GET'])
@token_required
def get_embeddings_stats(current_user_id):
    """Get embeddings service statistics"""
    try:
        stats = embeddings_service.get_service_stats()
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get embeddings stats: {str(e)}'}), 500

# Fine-tuning APIs
@app.route('/api/fine-tuning/datasets', methods=['GET'])
@token_required
def list_training_datasets(current_user_id):
    """List all training datasets"""
    try:
        datasets = fine_tuning_service.list_datasets()
        return jsonify({'datasets': datasets}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to list datasets: {str(e)}'}), 500

@app.route('/api/fine-tuning/datasets', methods=['POST'])
@token_required
def create_training_dataset(current_user_id):
    """Create a new training dataset"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        description = data.get('description', '')
        
        if not name:
            return jsonify({'message': 'Dataset name is required'}), 400
        
        success = fine_tuning_service.create_dataset(name, description)
        
        if success:
            return jsonify({
                'message': f'Dataset "{name}" created successfully',
                'name': name
            }), 201
        else:
            return jsonify({'message': 'Failed to create dataset'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to create dataset: {str(e)}'}), 500

@app.route('/api/fine-tuning/datasets/<dataset_name>', methods=['GET'])
@token_required
def get_dataset_details(current_user_id, dataset_name):
    """Get detailed information about a dataset"""
    try:
        details = fine_tuning_service.get_dataset_details(dataset_name)
        
        if details:
            return jsonify({'dataset': details}), 200
        else:
            return jsonify({'message': 'Dataset not found'}), 404
            
    except Exception as e:
        return jsonify({'message': f'Failed to get dataset details: {str(e)}'}), 500

@app.route('/api/fine-tuning/datasets/<dataset_name>/examples', methods=['POST'])
@token_required
def add_training_examples(current_user_id, dataset_name):
    """Add training examples to a dataset"""
    try:
        data = request.get_json()
        examples = data.get('examples', [])
        
        if not examples:
            return jsonify({'message': 'Examples are required'}), 400
        
        # Validate example format
        for example in examples:
            if 'input' not in example or 'output' not in example:
                return jsonify({'message': 'Each example must have "input" and "output" fields'}), 400
        
        success = fine_tuning_service.add_training_examples(dataset_name, examples)
        
        if success:
            return jsonify({
                'message': f'Added {len(examples)} examples to dataset "{dataset_name}"',
                'count': len(examples)
            }), 200
        else:
            return jsonify({'message': 'Failed to add examples'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to add examples: {str(e)}'}), 500

@app.route('/api/fine-tuning/datasets/<dataset_name>/import', methods=['POST'])
@token_required
def import_training_data(current_user_id, dataset_name):
    """Import training data from file"""
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided'}), 400
        
        file = request.files['file']
        format_type = request.form.get('format', 'jsonl')
        
        if file.filename == '':
            return jsonify({'message': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            
            success = fine_tuning_service.import_training_data(dataset_name, tmp_file.name, format_type)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            if success:
                return jsonify({
                    'message': f'Training data imported successfully to dataset "{dataset_name}"',
                    'filename': file.filename
                }), 200
            else:
                return jsonify({'message': 'Failed to import training data'}), 500
                
    except Exception as e:
        return jsonify({'message': f'Failed to import training data: {str(e)}'}), 500

@app.route('/api/fine-tuning/start', methods=['POST'])
@token_required
def start_fine_tuning(current_user_id):
    """Start a fine-tuning job"""
    try:
        data = request.get_json()
        
        # Required fields
        required_fields = ['base_model', 'dataset_name', 'output_model_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'message': f'Field "{field}" is required'}), 400
        
        job_id = fine_tuning_service.start_fine_tuning(data)
        
        if job_id:
            return jsonify({
                'message': 'Fine-tuning job started successfully',
                'job_id': job_id
            }), 202
        else:
            return jsonify({'message': 'Failed to start fine-tuning job'}), 500
            
    except Exception as e:
        return jsonify({'message': f'Failed to start fine-tuning: {str(e)}'}), 500

@app.route('/api/fine-tuning/jobs/<job_id>', methods=['GET'])
@token_required
def get_fine_tuning_job_status(current_user_id, job_id):
    """Get fine-tuning job status"""
    try:
        job_status = fine_tuning_service.get_job_status(job_id)
        
        if job_status:
            return jsonify({'job': job_status}), 200
        else:
            return jsonify({'message': 'Job not found'}), 404
            
    except Exception as e:
        return jsonify({'message': f'Failed to get job status: {str(e)}'}), 500

@app.route('/api/fine-tuning/jobs', methods=['GET'])
@token_required
def list_fine_tuning_jobs(current_user_id):
    """List all fine-tuning jobs"""
    try:
        jobs = fine_tuning_service.list_jobs()
        return jsonify({'jobs': jobs}), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to list jobs: {str(e)}'}), 500

@app.route('/api/fine-tuning/jobs/<job_id>/cancel', methods=['POST'])
@token_required
def cancel_fine_tuning_job(current_user_id, job_id):
    """Cancel a fine-tuning job"""
    try:
        success = fine_tuning_service.cancel_job(job_id)
        
        if success:
            return jsonify({'message': f'Job {job_id} cancelled successfully'}), 200
        else:
            return jsonify({'message': 'Job not found or cannot be cancelled'}), 404
            
    except Exception as e:
        return jsonify({'message': f'Failed to cancel job: {str(e)}'}), 500

@app.route('/api/fine-tuning/recommend-config', methods=['POST'])
@token_required
def get_recommended_fine_tuning_config(current_user_id):
    """Get recommended fine-tuning configuration"""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset_name', '').strip()
        use_case = data.get('use_case', 'general')
        
        if not dataset_name:
            return jsonify({'message': 'Dataset name is required'}), 400
        
        config = fine_tuning_service.get_recommended_config(dataset_name, use_case)
        
        if config:
            return jsonify({'recommended_config': config}), 200
        else:
            return jsonify({'message': 'Dataset not found'}), 404
            
    except Exception as e:
        return jsonify({'message': f'Failed to get recommended config: {str(e)}'}), 500

@app.route('/api/fine-tuning/stats', methods=['GET'])
@token_required
def get_fine_tuning_stats(current_user_id):
    """Get fine-tuning service statistics"""
    try:
        stats = fine_tuning_service.get_service_stats()
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'message': f'Failed to get fine-tuning stats: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test Ollama connection
        ollama_status = ollama_service.is_available()
        ollama_version = ollama_service.get_version() if ollama_status else {}
        
        # Get system resources
        system_status = model_manager.get_system_status()
        
        # Database check
        try:
            db.session.execute('SELECT 1')
            db_status = True
        except:
            db_status = False
        
        overall_status = "healthy" if all([ollama_status, db_status]) else "unhealthy"
        
        return jsonify({
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "ollama": {
                    "status": "up" if ollama_status else "down",
                    "version": ollama_version
                },
                "database": {
                    "status": "up" if db_status else "down"
                },
                "model_manager": {
                    "status": "up" if model_manager.monitoring_active else "down",
                    "active_models": len(model_manager.active_models)
                }
            },
            "system": system_status
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to chatbot service'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle real-time chat messages via WebSocket"""
    try:
        message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not message:
            emit('error', {'message': 'Message cannot be empty'})
            return
        
        # Emit typing indicator
        emit('typing', {'typing': True})
        
        # Generate AI response
        result = chatbot.generate_response(message, conversation_history)
        result['timestamp'] = datetime.now().isoformat()
        
        # Stop typing indicator and send response
        emit('typing', {'typing': False})
        emit('chat_response', result)
        
    except Exception as e:
        emit('typing', {'typing': False})
        emit('error', {'message': f'Server Error: {str(e)}'})

if __name__ == '__main__':
    print("🤖 Starting AI Chatbot Web Application...")
    print(f"📡 Ollama URL: {OLLAMA_BASE_URL}")
    print(f"🧠 Default Model: {DEFAULT_MODEL}")
    print(f"🗄️ Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("🌐 Server will be available at: http://localhost:5000")
    
    # Create database tables
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
    
    # Start model manager monitoring
    try:
        model_manager.start_monitoring()
        print("✅ Model manager monitoring started")
    except Exception as e:
        print(f"⚠️ Model manager initialization warning: {e}")
    
    # Run with SocketIO support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)