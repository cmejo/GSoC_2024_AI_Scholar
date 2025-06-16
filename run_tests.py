#!/usr/bin/env python3
"""
Test runner script for AI Chatbot application
Provides different test execution options and reporting
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result"""
    print(f"\n🔍 {description}")
    print(f"Command: {' '.join(command)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def setup_test_environment():
    """Set up test environment variables"""
    os.environ.setdefault('FLASK_ENV', 'testing')
    os.environ.setdefault('TESTING', 'true')
    os.environ.setdefault('DB_NAME', 'chatbot_test_db')
    os.environ.setdefault('DB_USER', 'chatbot_test_user')
    os.environ.setdefault('DB_PASSWORD', 'chatbot_test_password')
    
    # Load .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Loaded environment variables from .env file")
        except ImportError:
            print("⚠️  python-dotenv not installed, skipping .env file loading")


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests"""
    command = ['python', '-m', 'pytest', 'tests/test_models.py']
    
    if verbose:
        command.append('-v')
    
    if coverage:
        command.extend(['--cov=models', '--cov-report=term-missing'])
    
    return run_command(command, "Unit Tests (Models)")


def run_api_tests(verbose=False, coverage=False):
    """Run API tests"""
    test_files = [
        'tests/test_auth_api.py',
        'tests/test_chat_api.py',
        'tests/test_health_api.py'
    ]
    
    command = ['python', '-m', 'pytest'] + test_files
    
    if verbose:
        command.append('-v')
    
    if coverage:
        command.extend(['--cov=app', '--cov-report=term-missing'])
    
    return run_command(command, "API Tests")


def run_integration_tests(verbose=False, coverage=False):
    """Run integration tests"""
    command = ['python', '-m', 'pytest', 'tests/test_integration.py']
    
    if verbose:
        command.append('-v')
    
    if coverage:
        command.extend(['--cov=app', '--cov=models', '--cov-report=term-missing'])
    
    return run_command(command, "Integration Tests")


def run_all_tests(verbose=False, coverage=False, html_report=False):
    """Run all tests"""
    command = ['python', '-m', 'pytest', 'tests/']
    
    if verbose:
        command.append('-v')
    
    if coverage:
        command.extend([
            '--cov=app',
            '--cov=models',
            '--cov-report=term-missing'
        ])
        
        if html_report:
            command.append('--cov-report=html:htmlcov')
    
    return run_command(command, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function"""
    command = ['python', '-m', 'pytest', test_path]
    
    if verbose:
        command.append('-v')
    
    return run_command(command, f"Specific Test: {test_path}")


def run_linting():
    """Run code linting"""
    success = True
    
    # Check if flake8 is available
    try:
        result = run_command(['flake8', '--version'], "Checking flake8 availability")
        if result:
            success &= run_command(
                ['flake8', 'app.py', 'models.py', 'tests/', '--max-line-length=100'],
                "Code Linting (flake8)"
            )
    except FileNotFoundError:
        print("⚠️  flake8 not found, skipping linting")
    
    return success


def run_security_tests():
    """Run security tests"""
    try:
        # Check if bandit is available
        result = run_command(['bandit', '--version'], "Checking bandit availability")
        if result:
            return run_command(
                ['bandit', '-r', 'app.py', 'models.py', '-f', 'json'],
                "Security Tests (bandit)"
            )
    except FileNotFoundError:
        print("⚠️  bandit not found, skipping security tests")
        return True
    
    return True


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("📊 GENERATING COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    # Run tests with coverage and generate HTML report
    command = [
        'python', '-m', 'pytest', 'tests/',
        '--cov=app',
        '--cov=models',
        '--cov-report=html:htmlcov',
        '--cov-report=xml:coverage.xml',
        '--cov-report=term-missing',
        '--junit-xml=test-results.xml',
        '-v'
    ]
    
    success = run_command(command, "Comprehensive Test Report Generation")
    
    if success:
        print("\n📋 Test Report Files Generated:")
        print("  - htmlcov/index.html (HTML coverage report)")
        print("  - coverage.xml (XML coverage report)")
        print("  - test-results.xml (JUnit test results)")
        print("\n🌐 Open htmlcov/index.html in your browser to view the coverage report")
    
    return success


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='AI Chatbot Test Runner')
    parser.add_argument('--type', choices=['unit', 'api', 'integration', 'all', 'specific'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--test-path', help='Specific test file or function to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true', help='Generate coverage report')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--lint', action='store_true', help='Run code linting')
    parser.add_argument('--security', action='store_true', help='Run security tests')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive test report')
    
    args = parser.parse_args()
    
    print("🤖 AI Chatbot Test Runner")
    print("=" * 40)
    
    # Setup test environment
    setup_test_environment()
    
    success = True
    
    # Run linting if requested
    if args.lint:
        success &= run_linting()
    
    # Run security tests if requested
    if args.security:
        success &= run_security_tests()
    
    # Run tests based on type
    if args.type == 'unit':
        success &= run_unit_tests(args.verbose, args.coverage)
    elif args.type == 'api':
        success &= run_api_tests(args.verbose, args.coverage)
    elif args.type == 'integration':
        success &= run_integration_tests(args.verbose, args.coverage)
    elif args.type == 'all':
        success &= run_all_tests(args.verbose, args.coverage, args.html_report)
    elif args.type == 'specific':
        if not args.test_path:
            print("❌ --test-path is required when using --type specific")
            return 1
        success &= run_specific_test(args.test_path, args.verbose)
    
    # Generate comprehensive report if requested
    if args.report:
        success &= generate_test_report()
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Your AI Chatbot application is ready for deployment")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Please review the test output and fix any issues")
    print("="*60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())