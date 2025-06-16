import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { tokenManager } from '../utils/tokenManager';

function TokenStatusIndicator({ showDetails = false, className = '' }) {
  const { getTokenStatus, refreshToken } = useAuth();
  const [tokenStatus, setTokenStatus] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    const updateStatus = () => {
      const status = tokenManager.getTokenStatus();
      setTokenStatus(status);
    };

    // Update immediately
    updateStatus();

    // Update every 30 seconds
    const interval = setInterval(updateStatus, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = async () => {
    if (isRefreshing) return;
    
    setIsRefreshing(true);
    try {
      await refreshToken();
    } catch (error) {
      console.error('Manual refresh failed:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  if (!tokenStatus) return null;

  const getStatusIcon = () => {
    switch (tokenStatus.status) {
      case 'valid':
        return <i className="fas fa-check-circle text-green-500"></i>;
      case 'refresh_needed':
        return <i className="fas fa-exclamation-triangle text-yellow-500"></i>;
      case 'expired':
        return <i className="fas fa-times-circle text-red-500"></i>;
      case 'invalid':
        return <i className="fas fa-ban text-red-500"></i>;
      case 'missing':
        return <i className="fas fa-question-circle text-gray-500"></i>;
      default:
        return <i className="fas fa-circle text-gray-400"></i>;
    }
  };

  const getStatusColor = () => {
    switch (tokenStatus.status) {
      case 'valid':
        return 'text-green-600 dark:text-green-400';
      case 'refresh_needed':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'expired':
      case 'invalid':
        return 'text-red-600 dark:text-red-400';
      case 'missing':
        return 'text-gray-500 dark:text-gray-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getStatusText = () => {
    switch (tokenStatus.status) {
      case 'valid':
        return showDetails ? `Valid (${tokenManager.formatTimeRemaining()})` : 'Active';
      case 'refresh_needed':
        return showDetails ? `Expires in ${tokenManager.formatTimeRemaining()}` : 'Expires Soon';
      case 'expired':
        return 'Expired';
      case 'invalid':
        return 'Invalid';
      case 'missing':
        return 'No Token';
      default:
        return 'Unknown';
    }
  };

  const shouldShowRefreshButton = () => {
    return tokenStatus.status === 'refresh_needed' && !isRefreshing;
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {/* Status Icon */}
      <div className="flex items-center space-x-1">
        {getStatusIcon()}
        <span className={`text-sm font-medium ${getStatusColor()}`}>
          {getStatusText()}
        </span>
      </div>

      {/* Refresh Button */}
      {shouldShowRefreshButton() && (
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="px-2 py-1 text-xs bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300 hover:bg-yellow-200 dark:hover:bg-yellow-900/40 rounded transition-colors disabled:opacity-50"
          title="Refresh token"
        >
          {isRefreshing ? (
            <div className="flex items-center">
              <div className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin mr-1"></div>
              Refreshing...
            </div>
          ) : (
            'Extend'
          )}
        </button>
      )}

      {/* Detailed Status Tooltip */}
      {showDetails && tokenStatus.message && (
        <div className="hidden md:block">
          <div className="text-xs text-gray-500 dark:text-gray-400 max-w-xs truncate" title={tokenStatus.message}>
            {tokenStatus.message}
          </div>
        </div>
      )}
    </div>
  );
}

export default TokenStatusIndicator;