#!/usr/bin/env python3
"""
AI Chatbot Web Application
A responsive, mobile-ready web GUI for AI chatbot service with authentication and chat history
"""

from flask import Flask, render_template, request, jsonify, session
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
from models import db, User, ChatSession, ChatMessage, UserSession, DatabaseQueries, init_db

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

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_status = response.status_code == 200
    except:
        ollama_status = False
    
    return jsonify({
        "status": "healthy",
        "ollama_connected": ollama_status,
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now().isoformat()
    })

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
    
    # Run with SocketIO support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)