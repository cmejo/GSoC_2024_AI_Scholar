import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { tokenManager } from '../utils/tokenManager';

function SessionExpirationWarning() {
  const { refreshToken, logout, getTokenStatus } = useAuth();
  const [showWarning, setShowWarning] = useState(false);
  const [timeLeft, setTimeLeft] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [tokenStatus, setTokenStatus] = useState(null);

  useEffect(() => {
    const checkSessionExpiry = () => {
      const status = tokenManager.getTokenStatus();
      setTokenStatus(status);
      
      if (status.status === 'missing' || status.status === 'invalid') {
        setShowWarning(false);
        return;
      }
      
      if (status.status === 'expired') {
        setShowWarning(false);
        logout();
        return;
      }
      
      if (status.status === 'refresh_needed') {
        const timeUntilExpiry = tokenManager.getTimeUntilExpiry();
        const warningTime = 10 * 60 * 1000; // 10 minutes in milliseconds
        
        if (timeUntilExpiry <= warningTime && timeUntilExpiry > 0) {
          setShowWarning(true);
          setTimeLeft(Math.floor(timeUntilExpiry / 1000)); // Convert to seconds
        } else {
          setShowWarning(false);
        }
      } else {
        setShowWarning(false);
      }
    };

    // Check immediately
    checkSessionExpiry();

    // Check every 30 seconds
    const interval = setInterval(checkSessionExpiry, 30000);

    return () => clearInterval(interval);
  }, [logout]);

  useEffect(() => {
    if (!showWarning || timeLeft <= 0) return;

    // Update countdown every second
    const countdown = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          setShowWarning(false);
          logout();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(countdown);
  }, [showWarning, timeLeft, logout]);

  const handleRefreshSession = async () => {
    setIsRefreshing(true);
    try {
      await refreshToken();
      setShowWarning(false);
    } catch (error) {
      console.error('Failed to refresh session:', error);
      // Let the session expire naturally
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleLogoutNow = () => {
    setShowWarning(false);
    logout();
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  if (!showWarning) return null;

  return (
    <div className="fixed top-4 right-4 z-50 max-w-sm">
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg shadow-lg p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <i className="fas fa-exclamation-triangle text-yellow-600 dark:text-yellow-400 text-lg"></i>
          </div>
          <div className="ml-3 flex-1">
            <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
              Session Expiring Soon
            </h3>
            <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
              Your session will expire in {formatTime(timeLeft)}. Would you like to extend it?
            </p>
            {tokenStatus && (
              <p className="mt-1 text-xs text-yellow-600 dark:text-yellow-400">
                Status: {tokenStatus.message}
              </p>
            )}
            <div className="mt-3 flex space-x-2">
              <button
                onClick={handleRefreshSession}
                disabled={isRefreshing}
                className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded text-yellow-800 dark:text-yellow-200 bg-yellow-100 dark:bg-yellow-900/40 hover:bg-yellow-200 dark:hover:bg-yellow-900/60 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isRefreshing ? (
                  <>
                    <div className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin mr-1"></div>
                    Extending...
                  </>
                ) : (
                  'Extend Session'
                )}
              </button>
              <button
                onClick={handleLogoutNow}
                className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded text-yellow-800 dark:text-yellow-200 bg-transparent hover:bg-yellow-100 dark:hover:bg-yellow-900/40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
              >
                Logout Now
              </button>
            </div>
          </div>
          <div className="ml-4 flex-shrink-0">
            <button
              onClick={() => setShowWarning(false)}
              className="inline-flex text-yellow-400 dark:text-yellow-500 hover:text-yellow-500 dark:hover:text-yellow-400 focus:outline-none focus:text-yellow-500 dark:focus:text-yellow-400"
            >
              <i className="fas fa-times text-sm"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SessionExpirationWarning;