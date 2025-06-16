#!/usr/bin/env python3
"""
Setup validation script for AI Chatbot PostgreSQL and API testing implementation
Validates that all components are properly configured and working
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
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False


def main():
    """Main validation function"""
    print("🤖 AI Chatbot PostgreSQL & API Testing Setup Validation")
    print("=" * 60)
    
    success_count = 0
    total_checks = 0
    
    # Core Files Check
    print("\n📁 Core Files:")
    checks = [
        ("app.py", "Main application file"),
        ("models.py", "Database models"),
        ("requirements.txt", "Python dependencies"),
        ("manage_db.py", "Database management script"),
        ("run_tests.py", "Test runner script"),
    ]
    
    for file_path, description in checks:
        total_checks += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    # Docker Configuration Check
    print("\n🐳 Docker Configuration:")
    checks = [
        ("docker-compose.yml", "Production Docker Compose"),
        ("docker-compose.dev.yml", "Development Docker Compose"),
        ("Dockerfile", "Docker build file"),
        ("init-db.sql", "Database initialization script"),
    ]
    
    for file_path, description in checks:
        total_checks += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    # Database Migration Files Check
    print("\n🗄️ Database Migrations:")
    checks = [
        ("migrations", "Migrations directory"),
        ("migrations/versions", "Migration versions directory"),
        ("migrations/alembic.ini", "Alembic configuration"),
        ("migrations/env.py", "Migration environment"),
    ]
    
    for dir_path, description in checks:
        total_checks += 1
        if Path(dir_path).exists():
            if Path(dir_path).is_dir():
                if check_directory_exists(dir_path, description):
                    success_count += 1
            else:
                if check_file_exists(dir_path, description):
                    success_count += 1
        else:
            print(f"❌ {description}: {dir_path} - NOT FOUND")
    
    # Test Files Check
    print("\n🧪 Test Files:")
    checks = [
        ("tests", "Tests directory"),
        ("tests/__init__.py", "Test package init"),
        ("tests/conftest.py", "Pytest configuration"),
        ("tests/test_auth_api.py", "Authentication API tests"),
        ("tests/test_chat_api.py", "Chat API tests"),
        ("tests/test_models.py", "Database model tests"),
        ("tests/test_health_api.py", "Health API tests"),
        ("tests/test_integration.py", "Integration tests"),
    ]
    
    for path, description in checks:
        total_checks += 1
        if Path(path).is_dir():
            if check_directory_exists(path, description):
                success_count += 1
        else:
            if check_file_exists(path, description):
                success_count += 1
    
    # Documentation Check
    print("\n📚 Documentation:")
    checks = [
        ("DATABASE.md", "Database documentation"),
        ("API_TESTING.md", "API testing documentation"),
        (".env.example", "Environment configuration example"),
    ]
    
    for file_path, description in checks:
        total_checks += 1
        if check_file_exists(file_path, description):
            success_count += 1
    
    # Configuration Content Check
    print("\n⚙️ Configuration Content:")
    content_checks = [
        ("docker-compose.yml", "postgres:", "PostgreSQL service in production compose"),
        ("docker-compose.dev.yml", "postgres-dev:", "PostgreSQL dev service in dev compose"),
        ("docker-compose.dev.yml", "postgres-test:", "PostgreSQL test service in dev compose"),
        ("requirements.txt", "psycopg2-binary", "PostgreSQL driver in requirements"),
        ("requirements.txt", "flask-sqlalchemy", "SQLAlchemy in requirements"),
        ("requirements.txt", "flask-migrate", "Flask-Migrate in requirements"),
        ("requirements.txt", "pytest", "Pytest in requirements"),
        (".env.example", "DB_HOST", "Database configuration in env example"),
        ("app.py", "postgresql://", "PostgreSQL connection in app"),
        ("models.py", "db.Model", "SQLAlchemy models"),
    ]
    
    for file_path, search_text, description in content_checks:
        total_checks += 1
        if check_file_content(file_path, search_text, description):
            success_count += 1
    
    # Python Syntax Check
    print("\n🐍 Python Syntax Check:")
    python_files = [
        "app.py",
        "models.py",
        "manage_db.py",
        "run_tests.py",
        "tests/conftest.py",
        "tests/test_auth_api.py",
        "tests/test_chat_api.py",
        "tests/test_models.py",
    ]
    
    # Try both python and python3
    python_cmd = "python3" if run_command_check("python3 --version", "Python3 availability") else "python"
    
    for file_path in python_files:
        if Path(file_path).exists():
            total_checks += 1
            if run_command_check(f"{python_cmd} -m py_compile {file_path}", f"Syntax check: {file_path}"):
                success_count += 1
    
    # Dependencies Check (skip if no Python available)
    print("\n📦 Dependencies Check:")
    if run_command_check(f"{python_cmd} --version", "Python availability"):
        total_checks += 1
        if run_command_check(f"{python_cmd} -c 'import sys; print(\"Python available\")'", "Python interpreter working"):
            success_count += 1
    else:
        print("⚠️  Python not available - skipping dependency checks")
    
    # Final Summary
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful checks: {success_count}")
    print(f"❌ Failed checks: {total_checks - success_count}")
    print(f"📈 Success rate: {(success_count/total_checks)*100:.1f}%")
    
    if success_count == total_checks:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Your PostgreSQL database schema and API testing setup is complete and ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Start the development environment: docker-compose -f docker-compose.dev.yml up -d")
        print("   2. Run database migrations: python manage_db.py upgrade-db")
        print("   3. Run the test suite: python run_tests.py --type all --coverage")
        print("   4. Start developing your chatbot features!")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("❌ Please review the failed checks above and fix any issues.")
        print("📖 Refer to DATABASE.md and API_TESTING.md for detailed setup instructions.")
        return 1


if __name__ == '__main__':
    sys.exit(main())