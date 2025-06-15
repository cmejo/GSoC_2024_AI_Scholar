#!/usr/bin/env python3
"""
AI Chatbot Web Application
A responsive, mobile-ready web GUI for AI chatbot service
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import requests
import json
import uuid
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
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
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = "Human" if msg.get("type") == "user" else "Assistant"
            context += f"{role}: {msg.get('content', '')}\n"
        
        return f"{context}\nHuman: {message}\nAssistant:"

# Initialize chatbot service
chatbot = ChatbotService()

@app.route('/')
def index():
    """Serve the main chat interface"""
    # Generate unique session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html', session_id=session['session_id'])

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """REST API endpoint for chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not message:
            return jsonify({
                "success": False,
                "error": "Message cannot be empty"
            }), 400
        
        # Generate AI response
        result = chatbot.generate_response(message, conversation_history)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server Error: {str(e)}"
        }), 500

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
    print("🌐 Server will be available at: http://localhost:5000")
    
    # Run with SocketIO support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)