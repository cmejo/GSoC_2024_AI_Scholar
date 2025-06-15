#!/usr/bin/env python3
"""
Test script for AI Chatbot Web GUI
Verifies that all components are working correctly
"""

import requests
import json
import time
import sys
from app import ChatbotService

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_chat_endpoint():
    """Test the chat API endpoint"""
    print("🔍 Testing chat endpoint...")
    try:
        payload = {
            "message": "Hello, this is a test message",
            "history": []
        }
        response = requests.post(
            'http://localhost:5000/api/chat',
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Chat API working: {data.get('response', '')[:100]}...")
                return True
            else:
                print(f"❌ Chat API error: {data.get('error')}")
                return False
        else:
            print(f"❌ Chat API failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Chat API error: {e}")
        return False

def test_chatbot_service():
    """Test the ChatbotService class directly"""
    print("🔍 Testing ChatbotService...")
    try:
        chatbot = ChatbotService()
        result = chatbot.generate_response("Hello, this is a test")
        
        if result.get('success'):
            print(f"✅ ChatbotService working: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ ChatbotService error: {result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ ChatbotService error: {e}")
        return False

def test_static_files():
    """Test that static files are accessible"""
    print("🔍 Testing static files...")
    static_files = [
        '/static/css/style.css',
        '/static/js/app.js',
        '/static/manifest.json'
    ]
    
    all_passed = True
    for file_path in static_files:
        try:
            response = requests.get(f'http://localhost:5000{file_path}', timeout=5)
            if response.status_code == 200:
                print(f"✅ {file_path} accessible")
            else:
                print(f"❌ {file_path} failed: {response.status_code}")
                all_passed = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {file_path} error: {e}")
            all_passed = False
    
    return all_passed

def test_main_page():
    """Test that the main page loads"""
    print("🔍 Testing main page...")
    try:
        response = requests.get('http://localhost:5000/', timeout=5)
        if response.status_code == 200 and 'AI Chatbot' in response.text:
            print("✅ Main page loads correctly")
            return True
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Main page error: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 AI Chatbot Web GUI Test Suite")
    print("=================================")
    print()
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    tests = [
        ("Main Page", test_main_page),
        ("Static Files", test_static_files),
        ("Health Endpoint", test_health_endpoint),
        ("ChatbotService", test_chatbot_service),
        ("Chat Endpoint", test_chat_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your chatbot is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())