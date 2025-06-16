"""
Tests for chat API endpoints
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from models import ChatSession, ChatMessage, db


class TestChatAPI:
    """Test chat API endpoint"""
    
    def test_chat_success(self, client, auth_headers, sample_user, mock_ollama_api):
        """Test successful chat interaction"""
        data = {
            'message': 'Hello, how are you?',
            'session_id': None  # Will create new session
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert response_data['success'] is True
        assert 'response' in response_data
        assert 'session_id' in response_data
        assert 'timestamp' in response_data
        
        # Verify session was created
        session = ChatSession.query.filter_by(user_id=sample_user.id).first()
        assert session is not None
        
        # Verify messages were saved
        messages = ChatMessage.query.filter_by(session_id=session.id).all()
        assert len(messages) == 2  # User message + bot response
        assert messages[0].message_type == 'user'
        assert messages[0].content == 'Hello, how are you?'
        assert messages[1].message_type == 'bot'
    
    def test_chat_with_existing_session(self, client, auth_headers, sample_user, 
                                      sample_chat_session, mock_ollama_api):
        """Test chat with existing session"""
        data = {
            'message': 'This is a follow-up message',
            'session_id': sample_chat_session.id
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert response_data['success'] is True
        assert response_data['session_id'] == sample_chat_session.id
        
        # Verify messages were added to existing session
        messages = ChatMessage.query.filter_by(session_id=sample_chat_session.id).all()
        assert len(messages) == 2  # User message + bot response
    
    def test_chat_empty_message(self, client, auth_headers):
        """Test chat with empty message"""
        data = {
            'message': '',
            'session_id': None
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 400
        response_data = response.get_json()
        
        assert response_data['success'] is False
        assert 'Message cannot be empty' in response_data['error']
    
    def test_chat_no_auth(self, client):
        """Test chat without authentication"""
        data = {
            'message': 'Hello',
            'session_id': None
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 401
        response_data = response.get_json()
        assert response_data['message'] == 'Token is missing'
    
    def test_chat_ollama_error(self, client, auth_headers):
        """Test chat when Ollama API returns error"""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            data = {
                'message': 'Hello',
                'session_id': None
            }
            
            response = client.post('/api/chat',
                                 data=json.dumps(data),
                                 content_type='application/json',
                                 headers=auth_headers)
            
            assert response.status_code == 200  # API still returns 200
            response_data = response.get_json()
            
            assert response_data['success'] is False
            assert 'API Error' in response_data['error']
    
    def test_chat_conversation_context(self, client, auth_headers, sample_user, 
                                     sample_chat_session, sample_messages, mock_ollama_api):
        """Test that conversation context is passed to AI"""
        data = {
            'message': 'What did I just say?',
            'session_id': sample_chat_session.id
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 200
        
        # Verify that the AI service was called with conversation history
        mock_ollama_api.assert_called_once()
        call_args = mock_ollama_api.call_args
        request_data = call_args[1]['json']
        
        # The prompt should contain conversation history
        assert 'Previous conversation:' in request_data['prompt']


class TestChatSessions:
    """Test chat session management endpoints"""
    
    def test_get_chat_sessions(self, client, auth_headers, sample_user, sample_chat_session):
        """Test getting user's chat sessions"""
        response = client.get('/api/chat/sessions', headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert 'sessions' in response_data
        assert len(response_data['sessions']) == 1
        assert response_data['sessions'][0]['id'] == sample_chat_session.id
        assert response_data['sessions'][0]['name'] == sample_chat_session.session_name
    
    def test_get_chat_sessions_no_auth(self, client):
        """Test getting chat sessions without authentication"""
        response = client.get('/api/chat/sessions')
        
        assert response.status_code == 401
        response_data = response.get_json()
        assert response_data['message'] == 'Token is missing'
    
    def test_create_chat_session(self, client, auth_headers, sample_user):
        """Test creating a new chat session"""
        data = {
            'name': 'My New Chat Session'
        }
        
        response = client.post('/api/chat/sessions',
                             data=json.dumps(data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 201
        response_data = response.get_json()
        
        assert response_data['message'] == 'Session created successfully'
        assert 'session_id' in response_data
        assert 'session' in response_data
        assert response_data['session']['name'] == 'My New Chat Session'
        
        # Verify session was created in database
        session = ChatSession.query.get(response_data['session_id'])
        assert session is not None
        assert session.user_id == sample_user.id
        assert session.session_name == 'My New Chat Session'
    
    def test_create_chat_session_no_name(self, client, auth_headers, sample_user):
        """Test creating chat session without name (should use default)"""
        response = client.post('/api/chat/sessions',
                             data=json.dumps({}),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 201
        response_data = response.get_json()
        
        # Should have default name with timestamp
        assert 'Chat' in response_data['session']['name']
    
    def test_get_session_messages(self, client, auth_headers, sample_user, 
                                sample_chat_session, sample_messages):
        """Test getting messages for a specific session"""
        response = client.get(f'/api/chat/sessions/{sample_chat_session.id}', 
                            headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert 'messages' in response_data
        assert 'session' in response_data
        assert len(response_data['messages']) == 2
        assert response_data['session']['id'] == sample_chat_session.id
        
        # Check message order (should be chronological)
        messages = response_data['messages']
        assert messages[0]['type'] == 'user'
        assert messages[1]['type'] == 'bot'
    
    def test_get_session_messages_not_found(self, client, auth_headers):
        """Test getting messages for non-existent session"""
        response = client.get('/api/chat/sessions/99999', headers=auth_headers)
        
        assert response.status_code == 404
        response_data = response.get_json()
        assert response_data['message'] == 'Session not found'
    
    def test_get_session_messages_wrong_user(self, client, auth_headers, admin_user):
        """Test getting messages for session belonging to another user"""
        # Create session for admin user
        admin_session = ChatSession(user_id=admin_user.id, session_name='Admin Session')
        db.session.add(admin_session)
        db.session.commit()
        
        response = client.get(f'/api/chat/sessions/{admin_session.id}', 
                            headers=auth_headers)
        
        assert response.status_code == 404
        response_data = response.get_json()
        assert response_data['message'] == 'Session not found'
    
    def test_delete_chat_session(self, client, auth_headers, sample_user, 
                                sample_chat_session, sample_messages):
        """Test deleting a chat session"""
        session_id = sample_chat_session.id
        
        response = client.delete(f'/api/chat/sessions/{session_id}', 
                               headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        assert response_data['message'] == 'Session deleted successfully'
        
        # Verify session was deleted
        session = ChatSession.query.get(session_id)
        assert session is None
        
        # Verify messages were also deleted (cascade)
        messages = ChatMessage.query.filter_by(session_id=session_id).all()
        assert len(messages) == 0
    
    def test_delete_session_not_found(self, client, auth_headers):
        """Test deleting non-existent session"""
        response = client.delete('/api/chat/sessions/99999', headers=auth_headers)
        
        assert response.status_code == 404
        response_data = response.get_json()
        assert response_data['message'] == 'Session not found'
    
    def test_delete_session_wrong_user(self, client, auth_headers, admin_user):
        """Test deleting session belonging to another user"""
        # Create session for admin user
        admin_session = ChatSession(user_id=admin_user.id, session_name='Admin Session')
        db.session.add(admin_session)
        db.session.commit()
        
        response = client.delete(f'/api/chat/sessions/{admin_session.id}', 
                               headers=auth_headers)
        
        assert response.status_code == 404
        response_data = response.get_json()
        assert response_data['message'] == 'Session not found'


class TestChatbotService:
    """Test ChatbotService class directly"""
    
    def test_generate_response_success(self, mock_ollama_api):
        """Test successful response generation"""
        from app import ChatbotService
        
        chatbot = ChatbotService()
        result = chatbot.generate_response('Hello, how are you?')
        
        assert result['success'] is True
        assert 'response' in result
        assert result['model'] == 'test-model'
    
    def test_generate_response_with_history(self, mock_ollama_api, sample_messages):
        """Test response generation with conversation history"""
        from app import ChatbotService
        
        chatbot = ChatbotService()
        result = chatbot.generate_response('What did I say?', sample_messages)
        
        assert result['success'] is True
        
        # Verify that conversation history was included in the prompt
        mock_ollama_api.assert_called_once()
        call_args = mock_ollama_api.call_args
        request_data = call_args[1]['json']
        
        assert 'Previous conversation:' in request_data['prompt']
        assert 'Hello, this is a test message' in request_data['prompt']
    
    def test_generate_response_api_error(self):
        """Test response generation when API returns error"""
        from app import ChatbotService
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            chatbot = ChatbotService()
            result = chatbot.generate_response('Hello')
            
            assert result['success'] is False
            assert 'API Error' in result['error']
    
    def test_generate_response_connection_error(self):
        """Test response generation when connection fails"""
        from app import ChatbotService
        
        with patch('requests.post', side_effect=ConnectionError('Connection failed')):
            chatbot = ChatbotService()
            result = chatbot.generate_response('Hello')
            
            assert result['success'] is False
            assert 'Connection Error' in result['error']