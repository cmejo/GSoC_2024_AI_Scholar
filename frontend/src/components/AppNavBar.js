import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import UserMenu from './UserMenu';
import TokenStatusIndicator from './TokenStatusIndicator';

function AppNavBar({ onShowToast }) {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const isActive = (path) => {
    return location.pathname === path;
  };

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
      if (onShowToast) {
        onShowToast('Logged out successfully', 'success');
      }
    } catch (error) {
      if (onShowToast) {
        onShowToast('Logout failed', 'error');
      }
    }
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Left side - Logo and Navigation */}
          <div className="flex">
            {/* Logo */}
            <div className="flex-shrink-0 flex items-center">
              <Link to="/chat" className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                  <i className="fas fa-robot text-white text-sm"></i>
                </div>
                <div className="hidden sm:block">
                  <span className="text-xl font-bold text-gray-900 dark:text-gray-100">
                    AI Chatbot
                  </span>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Powered by Advanced AI
                  </div>
                </div>
              </Link>
            </div>

            {/* Desktop Navigation Links */}
            <div className="hidden sm:ml-8 sm:flex sm:space-x-1">
              <Link
                to="/chat"
                className={`inline-flex items-center px-4 py-2 border-b-2 text-sm font-medium transition-colors ${
                  isActive('/chat')
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/20'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                <i className="fas fa-comments mr-2"></i>
                Chat
                {isActive('/chat') && (
                  <div className="ml-2 w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
                )}
              </Link>
              
              <Link
                to="/history"
                className={`inline-flex items-center px-4 py-2 border-b-2 text-sm font-medium transition-colors ${
                  isActive('/history')
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/20'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                <i className="fas fa-history mr-2"></i>
                History
                {isActive('/history') && (
                  <div className="ml-2 w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
                )}
              </Link>
            </div>
          </div>

          {/* Right side - User info and actions */}
          <div className="flex items-center space-x-4">
            {/* Session Status Indicator */}
            <div className="hidden md:block">
              <TokenStatusIndicator showDetails={true} />
            </div>

            {/* User Info (hidden on mobile) */}
            <div className="hidden lg:flex items-center space-x-3">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {user?.firstName ? `${user.firstName} ${user.lastName || ''}`.trim() : user?.username || 'User'}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {user?.email}
                </p>
              </div>
            </div>

            {/* User Menu */}
            <div className="hidden sm:block">
              <UserMenu onShowToast={onShowToast} />
            </div>

            {/* Quick Logout Button (desktop) */}
            <button
              onClick={handleLogout}
              className="hidden sm:flex items-center px-3 py-2 rounded-lg text-sm font-medium text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
              title="Logout"
            >
              <i className="fas fa-sign-out-alt mr-2"></i>
              <span className="hidden lg:inline">Logout</span>
            </button>

            {/* Mobile menu button */}
            <button
              type="button"
              onClick={toggleMobileMenu}
              className="sm:hidden inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 transition-colors"
            >
              <i className={`fas ${isMobileMenuOpen ? 'fa-times' : 'fa-bars'} text-lg`}></i>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isMobileMenuOpen && (
        <div className="sm:hidden border-t border-gray-200 dark:border-gray-700">
          <div className="pt-2 pb-3 space-y-1">
            {/* User Info Mobile */}
            <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center text-white font-medium">
                  {user?.firstName?.charAt(0) || user?.username?.charAt(0) || 'U'}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {user?.firstName ? `${user.firstName} ${user.lastName || ''}`.trim() : user?.username || 'User'}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {user?.email}
                  </p>
                </div>
                <div className="text-right">
                  <TokenStatusIndicator />
                </div>
              </div>
            </div>

            {/* Navigation Links Mobile */}
            <Link
              to="/chat"
              onClick={() => setIsMobileMenuOpen(false)}
              className={`flex items-center pl-4 pr-4 py-3 border-l-4 text-base font-medium transition-colors ${
                isActive('/chat')
                  ? 'bg-primary-50 dark:bg-primary-900/20 border-primary-500 text-primary-700 dark:text-primary-300'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300'
              }`}
            >
              <i className="fas fa-comments mr-3 w-5"></i>
              <span className="flex-1">Chat</span>
              {isActive('/chat') && (
                <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
              )}
            </Link>
            
            <Link
              to="/history"
              onClick={() => setIsMobileMenuOpen(false)}
              className={`flex items-center pl-4 pr-4 py-3 border-l-4 text-base font-medium transition-colors ${
                isActive('/history')
                  ? 'bg-primary-50 dark:bg-primary-900/20 border-primary-500 text-primary-700 dark:text-primary-300'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300'
              }`}
            >
              <i className="fas fa-history mr-3 w-5"></i>
              <span className="flex-1">History</span>
              {isActive('/history') && (
                <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
              )}
            </Link>

            {/* Mobile Actions */}
            <div className="pt-4 pb-3 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => {
                  setIsMobileMenuOpen(false);
                  // Open user profile or settings
                }}
                className="flex items-center w-full pl-4 pr-4 py-3 text-base font-medium text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <i className="fas fa-user mr-3 w-5"></i>
                Profile
              </button>
              
              <button
                onClick={() => {
                  setIsMobileMenuOpen(false);
                  handleLogout();
                }}
                className="flex items-center w-full pl-4 pr-4 py-3 text-base font-medium text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20"
              >
                <i className="fas fa-sign-out-alt mr-3 w-5"></i>
                Logout
              </button>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}

export default AppNavBar;