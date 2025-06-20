#!/usr/bin/env python3
"""
Enhanced Backend Test Script
Tests the AI Chatbot backend functionality
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing Health Check...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            
            # Print service status
            services = data.get('services', {})
            for service, info in services.items():
                status = info.get('status', 'unknown')
                print(f"   - {service}: {status}")
            
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_user_registration():
    """Test user registration"""
    print("\n👤 Testing User Registration...")
    
    test_user = {
        "username": f"testuser_{int(time.time())}",
        "email": f"test_{int(time.time())}@example.com",
        "password": "testpassword123",
        "firstName": "Test",
        "lastName": "User"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json=test_user,
            timeout=10
        )
        
        if response.status_code == 201:
            data = response.json()
            print(f"✅ User Registration: Success")
            print(f"   - User ID: {data['user']['id']}")
            print(f"   - Username: {data['user']['username']}")
            
            # Return token for further tests
            return data['token']
        else:
            print(f"❌ User Registration Failed: {response.status_code}")
            print(f"   - Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ User Registration Error: {e}")
        return None

def test_models_endpoint(token):
    """Test models endpoint"""
    print("\n🤖 Testing Models Endpoint...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Models Endpoint: Found {len(models)} models")
            
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model['name']} ({model.get('size', 'unknown size')})")
            
            return len(models) > 0
        else:
            print(f"❌ Models Endpoint Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Models Endpoint Error: {e}")
        return False

def test_chat_session_creation(token):
    """Test chat session creation"""
    print("\n💬 Testing Chat Session Creation...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    session_data = {
        "name": f"Test Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat/sessions",
            headers=headers,
            json=session_data,
            timeout=10
        )
        
        if response.status_code == 201:
            data = response.json()
            session_id = data['session_id']
            print(f"✅ Chat Session Created: ID {session_id}")
            return session_id
        else:
            print(f"❌ Chat Session Creation Failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Chat Session Creation Error: {e}")
        return None

def test_chat_message(token, session_id):
    """Test sending a chat message"""
    print("\n📝 Testing Chat Message...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    message_data = {
        "message": "Hello! This is a test message. Please respond briefly.",
        "session_id": session_id
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            headers=headers,
            json=message_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Chat Message: Success")
                print(f"   - Response: {data['response'][:100]}...")
                return True
            else:
                print(f"❌ Chat Message Failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Chat Message Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat Message Error: {e}")
        return False

def test_system_status(token):
    """Test system status endpoint"""
    print("\n📊 Testing System Status...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/system/status",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System Status: Success")
            print(f"   - Ollama Available: {data.get('ollama_available', 'unknown')}")
            print(f"   - Total Models: {data.get('total_models', 'unknown')}")
            print(f"   - Monitoring Active: {data.get('monitoring_active', 'unknown')}")
            return True
        else:
            print(f"❌ System Status Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ System Status Error: {e}")
        return False

def test_model_recommendations(token):
    """Test model recommendations endpoint"""
    print("\n🎯 Testing Model Recommendations...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/models/recommendations?use_case=general_chat",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model Recommendations: Success")
            print(f"   - Use Case: {data.get('use_case', 'unknown')}")
            print(f"   - Optimal Model: {data.get('optimal_model', 'none')}")
            print(f"   - Recommendations: {len(data.get('recommendations', []))}")
            return True
        else:
            print(f"❌ Model Recommendations Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model Recommendations Error: {e}")
        return False

def run_all_tests():
    """Run all backend tests"""
    print("🧪 Enhanced Backend Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Health Check
    if test_health_check():
        tests_passed += 1
    
    # Test 2: User Registration
    token = test_user_registration()
    if token:
        tests_passed += 1
        
        # Test 3: Models Endpoint
        if test_models_endpoint(token):
            tests_passed += 1
        
        # Test 4: Chat Session Creation
        session_id = test_chat_session_creation(token)
        if session_id:
            tests_passed += 1
            
            # Test 5: Chat Message
            if test_chat_message(token, session_id):
                tests_passed += 1
        
        # Test 6: System Status
        if test_system_status(token):
            tests_passed += 1
        
        # Test 7: Model Recommendations
        if test_model_recommendations(token):
            tests_passed += 1
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Backend is working correctly.")
        return True
    elif tests_passed >= total_tests * 0.7:  # 70% pass rate
        print("⚠️ Most tests passed. Some features may need attention.")
        return True
    else:
        print("❌ Many tests failed. Please check the backend configuration.")
        return False

def main():
    """Main test function"""
    print("Starting Enhanced Backend Tests...")
    print("Make sure the backend server is running on http://localhost:5000")
    print()
    
    # Wait a moment for user to read
    time.sleep(2)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ Backend testing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Backend testing failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()