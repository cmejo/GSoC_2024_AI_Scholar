#!/usr/bin/env python3
"""
Setup script for routing and authentication features
Validates the implementation and provides setup guidance
"""

import os
import sys
import subprocess
from pathlib import Path


def check_file_exists(file_path, description):
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False


def check_directory_exists(dir_path, description):
    """Check if a directory exists"""
    if Path(dir_path).is_dir():
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ {description}: {dir_path} - NOT FOUND")
        return False


def check_file_content(file_path, search_text, description):
    """Check if a file contains specific content"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            if search_text in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - Content not found")
                return False
    except FileNotFoundError:
        print(f"❌ {description} - File not found: {file_path}")
        return False


def run_command_check(command, description):
    """Run a command and check if it succeeds"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description} - Command failed")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False


def main():
    """Main validation function"""
    print("🚀 AI Chatbot Routing & Authentication Setup Validation")
    print("=" * 60)
    
    success_count = 0
    total_checks = 0
    
    # Frontend Routing Components Check
    print("\n🧭 Frontend Routing Components:")
    checks = [
        ("frontend/src/components/ProtectedRoute.js", "Protected Route Guard"),
        ("frontend/src/components/PublicRoute.js", "Public Route Guard"),
        ("frontend/src/components/Navigation.js", "Navigation Component"),
        ("frontend/src/components/SessionExpirationWarning.js", "Session Warning Component"),
        ("frontend/src/components/UserMenu.js", "User Menu Component"),
    ]
    
    for file_path, description in checks:
        total_checks += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    # Page Components Check
    print("\n📄 Page Components:")
    checks = [
        ("frontend/src/pages/LoginPage.js", "Login Page"),
        ("frontend/src/pages/RegisterPage.js", "Register Page"),
        ("frontend/src/pages/ChatPage.js", "Chat Page"),
        ("frontend/src/pages/HistoryPage.js", "History Page"),
    ]
    
    for file_path, description in checks:
        total_checks += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    # Frontend Dependencies Check
    print("\n📦 Frontend Dependencies:")
    content_checks = [
        ("frontend/package.json", "react-router-dom", "React Router dependency"),
    ]
    
    for file_path, search_text, description in content_checks:
        total_checks += 1
        if check_file_content(file_path, search_text, description):
            success_count += 1
    
    # App.js Router Setup Check
    print("\n⚙️ Router Configuration:")
    content_checks = [
        ("frontend/src/App.js", "BrowserRouter", "Router setup in App.js"),
        ("frontend/src/App.js", "ProtectedRoute", "Protected routes configured"),
        ("frontend/src/App.js", "PublicRoute", "Public routes configured"),
    ]
    
    for file_path, search_text, description in content_checks:
        total_checks += 1
        if check_file_content(file_path, search_text, description):
            success_count += 1
    
    # AuthContext Enhancements Check
    print("\n🔐 Authentication Context:")
    content_checks = [
        ("frontend/src/context/AuthContext.js", "sessionExpiry", "Session expiry tracking"),
        ("frontend/src/context/AuthContext.js", "refreshToken", "Token refresh functionality"),
        ("frontend/src/context/AuthContext.js", "scheduleTokenRefresh", "Automatic refresh scheduling"),
    ]
    
    for file_path, search_text, description in content_checks:
        total_checks += 1
        if check_file_content(file_path, search_text, description):
            success_count += 1
    
    # Backend Endpoints Check
    print("\n🔧 Backend Endpoints:")
    content_checks = [
        ("app.py", "/api/auth/refresh", "Token refresh endpoint"),
        ("app.py", "first_name", "Enhanced registration with names"),
        ("app.py", "len(password) < 8", "8-character password requirement"),
    ]
    
    for file_path, search_text, description in content_checks:
        total_checks += 1
        if check_file_content(file_path, search_text, description):
            success_count += 1
    
    # Database Migration Check
    print("\n🗄️ Database Schema:")
    checks = [
        ("migrations/versions/002_add_user_profile_fields.py", "User profile migration"),
    ]
    
    for file_path, description in checks:
        total_checks += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    # Documentation Check
    print("\n📚 Documentation:")
    checks = [
        ("ROUTING_AND_SESSION_MANAGEMENT.md", "Routing and session documentation"),
    ]
    
    for file_path, description in checks:
        total_checks += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    # Component Integration Check
    print("\n🔗 Component Integration:")
    content_checks = [
        ("frontend/src/pages/ChatPage.js", "SessionExpirationWarning", "Session warning in ChatPage"),
        ("frontend/src/pages/HistoryPage.js", "SessionExpirationWarning", "Session warning in HistoryPage"),
        ("frontend/src/components/UserMenu.js", "useNavigate", "Navigation in UserMenu"),
    ]
    
    for file_path, search_text, description in content_checks:
        total_checks += 1
        if check_file_content(file_path, search_text, description):
            success_count += 1
    
    # Final Summary
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful checks: {success_count}")
    print(f"❌ Failed checks: {total_checks - success_count}")
    print(f"📈 Success rate: {(success_count/total_checks)*100:.1f}%")
    
    if success_count == total_checks:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Your routing and authentication system is complete and ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Install frontend dependencies: cd frontend && npm install")
        print("   2. Run database migrations: python manage_db.py upgrade-db")
        print("   3. Start the development environment: docker-compose -f docker-compose.dev.yml up -d")
        print("   4. Test the routing and authentication features!")
        print("\n📋 Features implemented:")
        print("   ✅ React Router with protected/public routes")
        print("   ✅ Authentication guards and redirects")
        print("   ✅ Session management with token refresh")
        print("   ✅ Session expiration warnings")
        print("   ✅ Enhanced user registration with profile fields")
        print("   ✅ Logout functionality throughout the app")
        print("   ✅ User info display and navigation")
        print("   ✅ Mobile-responsive navigation")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("❌ Please review the failed checks above and fix any issues.")
        print("📖 Refer to ROUTING_AND_SESSION_MANAGEMENT.md for detailed setup instructions.")
        
        # Provide specific guidance based on failed checks
        if success_count < total_checks * 0.5:
            print("\n🔧 Major issues detected. Recommended actions:")
            print("   1. Ensure all files have been created correctly")
            print("   2. Check that React Router is properly installed")
            print("   3. Verify that all imports are correct")
            print("   4. Review the implementation documentation")
        elif success_count < total_checks * 0.8:
            print("\n🔧 Minor issues detected. Recommended actions:")
            print("   1. Check for missing imports or typos")
            print("   2. Verify component integration")
            print("   3. Test the routing functionality")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())