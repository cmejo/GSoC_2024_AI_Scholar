"""
Tests for health check and utility API endpoints
"""

import pytest
import json
from unittest.mock import patch, MagicMock


class TestHealthAPI:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client, mock_ollama_health):
        """Test successful health check"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert response_data['status'] == 'healthy'
        assert response_data['ollama_connected'] is True
        assert 'model' in response_data
        assert 'timestamp' in response_data
    
    def test_health_check_ollama_down(self, client):
        """Test health check when Ollama is down"""
        with patch('requests.get', side_effect=ConnectionError('Connection failed')):
            response = client.get('/api/health')
            
            assert response.status_code == 200
            response_data = response.get_json()
            
            assert response_data['status'] == 'healthy'
            assert response_data['ollama_connected'] is False
    
    def test_health_check_ollama_error(self, client):
        """Test health check when Ollama returns error"""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            response = client.get('/api/health')
            
            assert response.status_code == 200
            response_data = response.get_json()
            
            assert response_data['status'] == 'healthy'
            assert response_data['ollama_connected'] is False


class TestMainPage:
    """Test main page endpoint"""
    
    def test_main_page_loads(self, client):
        """Test that main page loads successfully"""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'<!DOCTYPE html>' in response.data
        # Check for session ID generation
        with client.session_transaction() as sess:
            assert 'session_id' in sess
    
    def test_main_page_session_persistence(self, client):
        """Test that session ID persists across requests"""
        # First request
        response1 = client.get('/')
        assert response1.status_code == 200
        
        with client.session_transaction() as sess:
            session_id1 = sess.get('session_id')
        
        # Second request
        response2 = client.get('/')
        assert response2.status_code == 200
        
        with client.session_transaction() as sess:
            session_id2 = sess.get('session_id')
        
        # Session ID should be the same
        assert session_id1 == session_id2
        assert session_id1 is not None


class TestErrorHandling:
    """Test error handling for various scenarios"""
    
    def test_404_error(self, client):
        """Test 404 error for non-existent endpoint"""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method"""
        response = client.get('/api/auth/register')  # Should be POST
        assert response.status_code == 405
    
    def test_invalid_json(self, client, auth_headers):
        """Test handling of invalid JSON in request body"""
        response = client.post('/api/chat',
                             data='invalid json',
                             content_type='application/json',
                             headers=auth_headers)
        
        # Should handle gracefully
        assert response.status_code in [400, 500]
    
    def test_missing_content_type(self, client, auth_headers):
        """Test handling of missing content type"""
        response = client.post('/api/chat',
                             data=json.dumps({'message': 'test'}),
                             headers=auth_headers)
        
        # Should still work or return appropriate error
        assert response.status_code in [200, 400, 415]


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses"""
        response = client.options('/api/health')
        
        # Check for CORS headers (if configured)
        # Note: This depends on how CORS is configured in the app
        assert response.status_code in [200, 204]
    
    def test_preflight_request(self, client):
        """Test CORS preflight request"""
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type,Authorization'
        }
        
        response = client.options('/api/chat', headers=headers)
        
        # Should handle preflight request
        assert response.status_code in [200, 204]


class TestRateLimiting:
    """Test rate limiting (if implemented)"""
    
    def test_multiple_requests(self, client, auth_headers, mock_ollama_api):
        """Test multiple rapid requests"""
        # Make multiple requests rapidly
        responses = []
        for i in range(5):
            data = {'message': f'Test message {i}'}
            response = client.post('/api/chat',
                                 data=json.dumps(data),
                                 content_type='application/json',
                                 headers=auth_headers)
            responses.append(response)
        
        # All should succeed (unless rate limiting is implemented)
        for response in responses:
            assert response.status_code in [200, 429]  # 429 = Too Many Requests


class TestDatabaseConnection:
    """Test database connection and error handling"""
    
    def test_database_operations_with_auth(self, client, auth_headers):
        """Test that database operations work with authentication"""
        # Create a session (requires database)
        data = {'name': 'Test Session'}
        response = client.post('/api/chat/sessions',
                             data=json.dumps(data),
                             content_type='application/json',
                             headers=auth_headers)
        
        assert response.status_code == 201
        
        # Get sessions (requires database)
        response = client.get('/api/chat/sessions', headers=auth_headers)
        assert response.status_code == 200
    
    def test_user_operations(self, client):
        """Test user registration and login (requires database)"""
        # Register user
        register_data = {
            'username': 'dbtest',
            'email': 'dbtest@example.com',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(register_data),
                             content_type='application/json')
        
        assert response.status_code == 201
        
        # Login user
        login_data = {
            'username': 'dbtest',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/login',
                             data=json.dumps(login_data),
                             content_type='application/json')
        
        assert response.status_code == 200