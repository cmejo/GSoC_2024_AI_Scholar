import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import AppNavBar from '../components/AppNavBar';
import ChatHistory from '../components/ChatHistory';
import SessionExpirationWarning from '../components/SessionExpirationWarning';
import Toast from '../components/Toast';

function HistoryPage() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 5000);
  };

  const handleSelectSession = (sessionId) => {
    navigate(`/chat?session=${sessionId}`);
  };

  const handleNewSession = (sessionId) => {
    navigate(`/chat?session=${sessionId}`);
  };

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
      showToast('Logged out successfully', 'success');
    } catch (error) {
      showToast('Logout failed', 'error');
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation Bar */}
      <AppNavBar onShowToast={showToast} />

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 h-[calc(100vh-120px)]">
          <div className="flex flex-col h-full">
            {/* History Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center">
                  <i className="fas fa-history text-white text-lg"></i>
                </div>
                <div>
                  <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                    Chat History
                  </h1>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Manage and organize your conversation history
                  </p>
                </div>
              </div>
              
              <button
                onClick={() => navigate('/chat')}
                className="flex items-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
              >
                <i className="fas fa-comments mr-2"></i>
                Back to Chat
              </button>
            </div>

            {/* Chat History Component */}
            <div className="flex-1 overflow-hidden">
              <ChatHistory
                currentSessionId={null}
                onSelectSession={handleSelectSession}
                onNewSession={handleNewSession}
                onShowToast={showToast}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Session Expiration Warning */}
      <SessionExpirationWarning />

      {/* Toast Notifications */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}

export default HistoryPage;