"""
Integration tests for AI Chatbot application
Tests complete workflows and interactions between components
"""

import pytest
import json
from models import User, ChatSession, ChatMessage, db


class TestUserWorkflow:
    """Test complete user workflow from registration to chat"""
    
    def test_complete_user_journey(self, client, mock_ollama_api):
        """Test complete user journey: register -> login -> chat -> logout"""
        
        # Step 1: Register new user
        register_data = {
            'username': 'journeyuser',
            'email': 'journey@example.com',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(register_data),
                             content_type='application/json')
        
        assert response.status_code == 201
        register_response = response.get_json()
        token = register_response['token']
        user_id = register_response['user']['id']
        
        # Step 2: Verify user can access protected endpoints
        headers = {'Authorization': f'Bearer {token}'}
        
        response = client.get('/api/auth/me', headers=headers)
        assert response.status_code == 200
        
        # Step 3: Create a chat session
        session_data = {'name': 'My First Chat'}
        response = client.post('/api/chat/sessions',
                             data=json.dumps(session_data),
                             content_type='application/json',
                             headers=headers)
        
        assert response.status_code == 201
        session_response = response.get_json()
        session_id = session_response['session_id']
        
        # Step 4: Send chat messages
        chat_data = {
            'message': 'Hello, this is my first message!',
            'session_id': session_id
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(chat_data),
                             content_type='application/json',
                             headers=headers)
        
        assert response.status_code == 200
        chat_response = response.get_json()
        assert chat_response['success'] is True
        
        # Step 5: Send follow-up message
        chat_data = {
            'message': 'Can you tell me more?',
            'session_id': session_id
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(chat_data),
                             content_type='application/json',
                             headers=headers)
        
        assert response.status_code == 200
        
        # Step 6: Get chat history
        response = client.get(f'/api/chat/sessions/{session_id}', headers=headers)
        assert response.status_code == 200
        
        history_response = response.get_json()
        messages = history_response['messages']
        
        # Should have 4 messages: 2 user + 2 bot responses
        assert len(messages) == 4
        assert messages[0]['type'] == 'user'
        assert messages[1]['type'] == 'bot'
        assert messages[2]['type'] == 'user'
        assert messages[3]['type'] == 'bot'
        
        # Step 7: Get all sessions
        response = client.get('/api/chat/sessions', headers=headers)
        assert response.status_code == 200
        
        sessions_response = response.get_json()
        assert len(sessions_response['sessions']) == 1
        assert sessions_response['sessions'][0]['id'] == session_id
        
        # Step 8: Logout
        response = client.post('/api/auth/logout', headers=headers)
        assert response.status_code == 200
    
    def test_multiple_sessions_workflow(self, client, mock_ollama_api):
        """Test user creating and managing multiple chat sessions"""
        
        # Register and login
        register_data = {
            'username': 'multisession',
            'email': 'multi@example.com',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(register_data),
                             content_type='application/json')
        
        assert response.status_code == 201
        token = response.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}
        
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            session_data = {'name': f'Chat Session {i+1}'}
            response = client.post('/api/chat/sessions',
                                 data=json.dumps(session_data),
                                 content_type='application/json',
                                 headers=headers)
            
            assert response.status_code == 201
            session_ids.append(response.get_json()['session_id'])
        
        # Send messages to different sessions
        for i, session_id in enumerate(session_ids):
            chat_data = {
                'message': f'Message for session {i+1}',
                'session_id': session_id
            }
            
            response = client.post('/api/chat',
                                 data=json.dumps(chat_data),
                                 content_type='application/json',
                                 headers=headers)
            
            assert response.status_code == 200
        
        # Verify all sessions exist
        response = client.get('/api/chat/sessions', headers=headers)
        assert response.status_code == 200
        
        sessions = response.get_json()['sessions']
        assert len(sessions) == 3
        
        # Delete one session
        response = client.delete(f'/api/chat/sessions/{session_ids[0]}', headers=headers)
        assert response.status_code == 200
        
        # Verify session was deleted
        response = client.get('/api/chat/sessions', headers=headers)
        assert response.status_code == 200
        
        sessions = response.get_json()['sessions']
        assert len(sessions) == 2
        
        remaining_ids = [s['id'] for s in sessions]
        assert session_ids[0] not in remaining_ids
        assert session_ids[1] in remaining_ids
        assert session_ids[2] in remaining_ids


class TestErrorRecovery:
    """Test error recovery and edge cases"""
    
    def test_chat_with_ai_service_down(self, client, auth_headers):
        """Test chat functionality when AI service is down"""
        
        # Mock AI service failure
        with pytest.raises(Exception):
            # This should trigger the error handling in the chat endpoint
            chat_data = {
                'message': 'Hello, are you there?',
                'session_id': None
            }
            
            response = client.post('/api/chat',
                                 data=json.dumps(chat_data),
                                 content_type='application/json',
                                 headers=auth_headers)
            
            # Should still return a response, but with error
            assert response.status_code == 200
            response_data = response.get_json()
            assert response_data['success'] is False
            assert 'error' in response_data
    
    def test_session_isolation(self, client, mock_ollama_api):
        """Test that different users' sessions are properly isolated"""
        
        # Create two users
        users_data = [
            {'username': 'user1', 'email': 'user1@example.com', 'password': 'password123'},
            {'username': 'user2', 'email': 'user2@example.com', 'password': 'password123'}
        ]
        
        tokens = []
        for user_data in users_data:
            response = client.post('/api/auth/register',
                                 data=json.dumps(user_data),
                                 content_type='application/json')
            
            assert response.status_code == 201
            tokens.append(response.get_json()['token'])
        
        # Create sessions for each user
        session_ids = []
        for i, token in enumerate(tokens):
            headers = {'Authorization': f'Bearer {token}'}
            session_data = {'name': f'User {i+1} Session'}
            
            response = client.post('/api/chat/sessions',
                                 data=json.dumps(session_data),
                                 content_type='application/json',
                                 headers=headers)
            
            assert response.status_code == 201
            session_ids.append(response.get_json()['session_id'])
        
        # User 1 should not be able to access User 2's session
        headers1 = {'Authorization': f'Bearer {tokens[0]}'}
        response = client.get(f'/api/chat/sessions/{session_ids[1]}', headers=headers1)
        assert response.status_code == 404
        
        # User 2 should not be able to access User 1's session
        headers2 = {'Authorization': f'Bearer {tokens[1]}'}
        response = client.get(f'/api/chat/sessions/{session_ids[0]}', headers=headers2)
        assert response.status_code == 404
        
        # Each user should only see their own sessions
        response = client.get('/api/chat/sessions', headers=headers1)
        assert response.status_code == 200
        user1_sessions = response.get_json()['sessions']
        assert len(user1_sessions) == 1
        assert user1_sessions[0]['id'] == session_ids[0]
        
        response = client.get('/api/chat/sessions', headers=headers2)
        assert response.status_code == 200
        user2_sessions = response.get_json()['sessions']
        assert len(user2_sessions) == 1
        assert user2_sessions[0]['id'] == session_ids[1]


class TestDataConsistency:
    """Test data consistency and database integrity"""
    
    def test_cascade_deletion(self, client, auth_headers, app):
        """Test that deleting a session also deletes its messages"""
        
        # Create session and send messages
        session_data = {'name': 'Test Cascade Session'}
        response = client.post('/api/chat/sessions',
                             data=json.dumps(session_data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 201
        session_id = response.get_json()['session_id']
        
        # Send multiple messages
        for i in range(3):
            chat_data = {
                'message': f'Test message {i+1}',
                'session_id': session_id
            }
            
            with pytest.raises(Exception):
                # This will fail due to AI service, but messages should still be saved
                response = client.post('/api/chat',
                                     data=json.dumps(chat_data),
                                     content_type='application/json',
                                     headers=auth_headers)
        
        # Verify messages exist
        with app.app_context():
            messages_before = ChatMessage.query.filter_by(session_id=session_id).count()
            # At least user messages should be saved
            assert messages_before > 0
        
        # Delete session
        response = client.delete(f'/api/chat/sessions/{session_id}', headers=auth_headers)
        assert response.status_code == 200
        
        # Verify messages were also deleted
        with app.app_context():
            messages_after = ChatMessage.query.filter_by(session_id=session_id).count()
            assert messages_after == 0
    
    def test_user_deletion_cascade(self, client, app):
        """Test that deleting a user cascades to sessions and messages"""
        
        # Register user
        register_data = {
            'username': 'cascadeuser',
            'email': 'cascade@example.com',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(register_data),
                             content_type='application/json')
        
        assert response.status_code == 201
        user_id = response.get_json()['user']['id']
        token = response.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}
        
        # Create session
        session_data = {'name': 'Cascade Test Session'}
        response = client.post('/api/chat/sessions',
                             data=json.dumps(session_data),
                             content_type='application/json',
                             headers=headers)
        
        assert response.status_code == 201
        session_id = response.get_json()['session_id']
        
        # Verify data exists
        with app.app_context():
            user = User.query.get(user_id)
            assert user is not None
            
            session = ChatSession.query.get(session_id)
            assert session is not None
            assert session.user_id == user_id
        
        # Delete user (simulated - would need admin endpoint)
        with app.app_context():
            user = User.query.get(user_id)
            db.session.delete(user)
            db.session.commit()
            
            # Verify cascade deletion
            session = ChatSession.query.get(session_id)
            assert session is None  # Should be deleted due to cascade


class TestPerformance:
    """Test performance-related scenarios"""
    
    def test_large_message_content(self, client, auth_headers, mock_ollama_api):
        """Test handling of large message content"""
        
        # Create a large message (but within reasonable limits)
        large_message = "This is a test message. " * 100  # ~2500 characters
        
        chat_data = {
            'message': large_message,
            'session_id': None
        }
        
        response = client.post('/api/chat',
                             data=json.dumps(chat_data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        assert response_data['success'] is True
    
    def test_many_sessions(self, client, auth_headers):
        """Test creating many sessions"""
        
        # Create many sessions
        session_count = 20
        for i in range(session_count):
            session_data = {'name': f'Performance Test Session {i+1}'}
            response = client.post('/api/chat/sessions',
                                 data=json.dumps(session_data),
                                 content_type='application/json',
                                 headers=auth_headers)
            
            assert response.status_code == 201
        
        # Get all sessions
        response = client.get('/api/chat/sessions', headers=auth_headers)
        assert response.status_code == 200
        
        sessions = response.get_json()['sessions']
        # Should return up to the limit (50 by default)
        assert len(sessions) == min(session_count, 50)
    
    def test_session_with_many_messages(self, client, auth_headers, mock_ollama_api):
        """Test session with many messages"""
        
        # Create session
        session_data = {'name': 'High Volume Session'}
        response = client.post('/api/chat/sessions',
                             data=json.dumps(session_data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 201
        session_id = response.get_json()['session_id']
        
        # Send many messages
        message_count = 10
        for i in range(message_count):
            chat_data = {
                'message': f'Message number {i+1}',
                'session_id': session_id
            }
            
            response = client.post('/api/chat',
                                 data=json.dumps(chat_data),
                                 content_type='application/json',
                                 headers=auth_headers)
            
            assert response.status_code == 200
        
        # Get session messages
        response = client.get(f'/api/chat/sessions/{session_id}', headers=auth_headers)
        assert response.status_code == 200
        
        messages = response.get_json()['messages']
        # Should have user messages + bot responses
        assert len(messages) == message_count * 2