import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import AppNavBar from '../components/AppNavBar';
import ChatContainer from '../components/ChatContainer';
import ChatHistory from '../components/ChatHistory';
import SettingsModal from '../components/SettingsModal';
import SessionExpirationWarning from '../components/SessionExpirationWarning';
import Toast from '../components/Toast';

function ChatPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { user, logout } = useAuth();
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [toast, setToast] = useState(null);

  // Check for session parameter in URL
  useEffect(() => {
    const sessionParam = searchParams.get('session');
    if (sessionParam) {
      setCurrentSessionId(parseInt(sessionParam));
    }
  }, [searchParams]);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 5000);
  };

  const handleSelectSession = (sessionId) => {
    setCurrentSessionId(sessionId);
    setShowHistory(false); // Close history on mobile after selection
  };

  const handleNewSession = (sessionId) => {
    setCurrentSessionId(sessionId);
    setShowHistory(false);
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
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation Bar */}
      <AppNavBar onShowToast={showToast} />

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Chat History */}
        <div className={`${
          showHistory ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-30 w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transition-transform duration-300 ease-in-out lg:transition-none`}>
          <div className="flex flex-col h-full">
            {/* Sidebar Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 lg:hidden">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Chat History
              </h2>
              <button
                onClick={() => setShowHistory(false)}
                className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <i className="fas fa-times text-lg"></i>
              </button>
            </div>

            {/* Chat History Component */}
            <div className="flex-1 overflow-hidden">
              <ChatHistory
                currentSessionId={currentSessionId}
                onSelectSession={handleSelectSession}
                onNewSession={handleNewSession}
                onShowToast={showToast}
              />
            </div>
          </div>
        </div>

        {/* Overlay for mobile */}
        {showHistory && (
          <div 
            className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-20"
            onClick={() => setShowHistory(false)}
          />
        )}

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Mobile Header for Chat */}
          <div className="lg:hidden bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3">
            <div className="flex items-center justify-between">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <i className="fas fa-bars text-lg"></i>
              </button>
              
              <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Chat
              </h1>
              
              <button
                onClick={() => setShowSettings(true)}
                className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                title="Settings"
              >
                <i className="fas fa-cog text-lg"></i>
              </button>
            </div>
          </div>

          {/* Chat Container */}
          <div className="flex-1 overflow-hidden">
            <ChatContainer
              sessionId={currentSessionId}
              onSessionChange={setCurrentSessionId}
              onShowToast={showToast}
            />
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onShowToast={showToast}
        />
      )}

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

export default ChatPage;