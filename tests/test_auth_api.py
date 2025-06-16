"""
Tests for authentication API endpoints
"""

import pytest
import json
import jwt
from datetime import datetime, timedelta
from models import User, UserSession, db


class TestAuthRegistration:
    """Test user registration endpoint"""
    
    def test_register_success(self, client):
        """Test successful user registration"""
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register', 
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 201
        response_data = response.get_json()
        
        assert response_data['message'] == 'User registered successfully'
        assert 'token' in response_data
        assert 'user' in response_data
        assert response_data['user']['username'] == 'newuser'
        assert response_data['user']['email'] == 'newuser@example.com'
        
        # Verify user was created in database
        user = User.query.filter_by(username='newuser').first()
        assert user is not None
        assert user.email == 'newuser@example.com'
    
    def test_register_missing_fields(self, client):
        """Test registration with missing required fields"""
        test_cases = [
            {},  # All fields missing
            {'username': 'test'},  # Missing email and password
            {'email': 'test@example.com'},  # Missing username and password
            {'password': 'password123'},  # Missing username and email
            {'username': 'test', 'email': 'test@example.com'},  # Missing password
        ]
        
        for data in test_cases:
            response = client.post('/api/auth/register',
                                 data=json.dumps(data),
                                 content_type='application/json')
            
            assert response.status_code == 400
            response_data = response.get_json()
            assert 'Username, email, and password are required' in response_data['message']
    
    def test_register_short_password(self, client):
        """Test registration with password too short"""
        data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': '123'  # Too short
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = response.get_json()
        assert 'Password must be at least 6 characters long' in response_data['message']
    
    def test_register_duplicate_username(self, client, sample_user):
        """Test registration with existing username"""
        data = {
            'username': sample_user.username,  # Duplicate username
            'email': 'different@example.com',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = response.get_json()
        assert 'Username or email already exists' in response_data['message']
    
    def test_register_duplicate_email(self, client, sample_user):
        """Test registration with existing email"""
        data = {
            'username': 'differentuser',
            'email': sample_user.email,  # Duplicate email
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = response.get_json()
        assert 'Username or email already exists' in response_data['message']


class TestAuthLogin:
    """Test user login endpoint"""
    
    def test_login_success_with_username(self, client, sample_user):
        """Test successful login with username"""
        data = {
            'username': sample_user.username,
            'password': 'testpassword123'
        }
        
        response = client.post('/api/auth/login',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert response_data['message'] == 'Login successful'
        assert 'token' in response_data
        assert 'user' in response_data
        assert response_data['user']['username'] == sample_user.username
    
    def test_login_success_with_email(self, client, sample_user):
        """Test successful login with email"""
        data = {
            'username': sample_user.email,  # Using email as username
            'password': 'testpassword123'
        }
        
        response = client.post('/api/auth/login',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert response_data['message'] == 'Login successful'
        assert 'token' in response_data
        assert response_data['user']['email'] == sample_user.email
    
    def test_login_invalid_credentials(self, client, sample_user):
        """Test login with invalid credentials"""
        test_cases = [
            {'username': sample_user.username, 'password': 'wrongpassword'},
            {'username': 'nonexistentuser', 'password': 'testpassword123'},
            {'username': sample_user.email, 'password': 'wrongpassword'},
        ]
        
        for data in test_cases:
            response = client.post('/api/auth/login',
                                 data=json.dumps(data),
                                 content_type='application/json')
            
            assert response.status_code == 401
            response_data = response.get_json()
            assert response_data['message'] == 'Invalid credentials'
    
    def test_login_missing_fields(self, client):
        """Test login with missing required fields"""
        test_cases = [
            {},  # Both fields missing
            {'username': 'testuser'},  # Missing password
            {'password': 'password123'},  # Missing username
        ]
        
        for data in test_cases:
            response = client.post('/api/auth/login',
                                 data=json.dumps(data),
                                 content_type='application/json')
            
            assert response.status_code == 400
            response_data = response.get_json()
            assert 'Username and password are required' in response_data['message']
    
    def test_login_updates_last_login(self, client, sample_user, app):
        """Test that login updates the last_login timestamp"""
        original_last_login = sample_user.last_login
        
        data = {
            'username': sample_user.username,
            'password': 'testpassword123'
        }
        
        response = client.post('/api/auth/login',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 200
        
        # Check that last_login was updated
        with app.app_context():
            updated_user = User.query.get(sample_user.id)
            assert updated_user.last_login != original_last_login
            assert updated_user.last_login is not None


class TestAuthTokenValidation:
    """Test JWT token validation and protected endpoints"""
    
    def test_get_current_user_success(self, client, auth_headers, sample_user):
        """Test getting current user with valid token"""
        response = client.get('/api/auth/me', headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        
        assert 'user' in response_data
        assert response_data['user']['id'] == sample_user.id
        assert response_data['user']['username'] == sample_user.username
    
    def test_get_current_user_no_token(self, client):
        """Test getting current user without token"""
        response = client.get('/api/auth/me')
        
        assert response.status_code == 401
        response_data = response.get_json()
        assert response_data['message'] == 'Token is missing'
    
    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token"""
        headers = {'Authorization': 'Bearer invalid-token'}
        response = client.get('/api/auth/me', headers=headers)
        
        assert response.status_code == 401
        response_data = response.get_json()
        assert response_data['message'] == 'Token is invalid'
    
    def test_get_current_user_expired_token(self, client, app, sample_user):
        """Test getting current user with expired token"""
        with app.app_context():
            # Create expired token
            expired_token = jwt.encode({
                'user_id': sample_user.id,
                'username': sample_user.username,
                'exp': datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
            }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        headers = {'Authorization': f'Bearer {expired_token}'}
        response = client.get('/api/auth/me', headers=headers)
        
        assert response.status_code == 401
        response_data = response.get_json()
        assert response_data['message'] == 'Token has expired'
    
    def test_logout_success(self, client, auth_headers):
        """Test successful logout"""
        response = client.post('/api/auth/logout', headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.get_json()
        assert response_data['message'] == 'Logout successful'
    
    def test_logout_no_token(self, client):
        """Test logout without token"""
        response = client.post('/api/auth/logout')
        
        assert response.status_code == 401
        response_data = response.get_json()
        assert response_data['message'] == 'Token is missing'


class TestUserSessions:
    """Test user session management"""
    
    def test_session_created_on_login(self, client, sample_user, app):
        """Test that user session is created on login"""
        data = {
            'username': sample_user.username,
            'password': 'testpassword123'
        }
        
        response = client.post('/api/auth/login',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 200
        
        # Check that user session was created
        with app.app_context():
            session = UserSession.query.filter_by(user_id=sample_user.id).first()
            assert session is not None
            assert session.is_active is True
            assert session.expires_at > datetime.utcnow()
    
    def test_session_created_on_registration(self, client, app):
        """Test that user session is created on registration"""
        data = {
            'username': 'newsessionuser',
            'email': 'newsession@example.com',
            'password': 'password123'
        }
        
        response = client.post('/api/auth/register',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 201
        
        # Check that user session was created
        with app.app_context():
            user = User.query.filter_by(username='newsessionuser').first()
            session = UserSession.query.filter_by(user_id=user.id).first()
            assert session is not None
            assert session.is_active is True