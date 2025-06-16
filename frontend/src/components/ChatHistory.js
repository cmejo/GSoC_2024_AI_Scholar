import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { chatHistoryService } from '../services/chatHistoryService';
import { formatTime } from '../utils/messageUtils';

function ChatHistory({ currentSessionId, onSelectSession, onNewSession, onShowToast }) {
  const [sessions, setSessions] = useState([]);
  const [filteredSessions, setFilteredSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('updated'); // 'updated', 'created', 'name', 'messages'
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc', 'desc'
  const [selectedSessions, setSelectedSessions] = useState(new Set());
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [showRenameModal, setShowRenameModal] = useState(null);
  const [newSessionName, setNewSessionName] = useState('');
  const [isRenaming, setIsRenaming] = useState(false);
  const [viewMode, setViewMode] = useState('list'); // 'list', 'grid'
  const searchInputRef = useRef(null);
  const { token } = useAuth();

  useEffect(() => {
    loadSessions();
  }, []);

  // Filter and sort sessions when search query, sort options, or sessions change
  useEffect(() => {
    let filtered = [...sessions];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(session => 
        session.name.toLowerCase().includes(query) ||
        session.id.toString().includes(query)
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'created':
          aValue = new Date(a.created_at);
          bValue = new Date(b.created_at);
          break;
        case 'messages':
          aValue = a.message_count;
          bValue = b.message_count;
          break;
        case 'updated':
        default:
          aValue = new Date(a.last_message_time || a.updated_at);
          bValue = new Date(b.last_message_time || b.updated_at);
          break;
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    setFilteredSessions(filtered);
  }, [sessions, searchQuery, sortBy, sortOrder]);

  const loadSessions = async () => {
    try {
      setIsLoading(true);
      const sessionsData = await chatHistoryService.getSessions(token);
      setSessions(sessionsData);
    } catch (error) {
      console.error('Failed to load sessions:', error);
      onShowToast('Failed to load chat history', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSession = async () => {
    try {
      setIsCreating(true);
      const response = await chatHistoryService.createSession(token);
      await loadSessions(); // Refresh the list
      onNewSession(response.session_id);
      onShowToast('New chat session created', 'success');
    } catch (error) {
      console.error('Failed to create session:', error);
      onShowToast('Failed to create new session', 'error');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteSession = async (sessionId, sessionName) => {
    if (!window.confirm(`Are you sure you want to delete "${sessionName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      await chatHistoryService.deleteSession(sessionId, token);
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      
      // If the deleted session was the current one, create a new session
      if (sessionId === currentSessionId) {
        handleNewSession();
      }
      
      onShowToast('Chat session deleted', 'success');
    } catch (error) {
      console.error('Failed to delete session:', error);
      onShowToast('Failed to delete session', 'error');
    }
  };

  const handleRenameSession = async (sessionId, newName) => {
    if (!newName.trim()) {
      onShowToast('Session name cannot be empty', 'error');
      return;
    }

    try {
      setIsRenaming(true);
      await chatHistoryService.renameSession(sessionId, newName.trim(), token);
      setSessions(prev => prev.map(s => 
        s.id === sessionId ? { ...s, name: newName.trim() } : s
      ));
      setShowRenameModal(null);
      setNewSessionName('');
      onShowToast('Session renamed successfully', 'success');
    } catch (error) {
      console.error('Failed to rename session:', error);
      onShowToast('Failed to rename session', 'error');
    } finally {
      setIsRenaming(false);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedSessions.size === 0) return;
    
    const sessionNames = Array.from(selectedSessions).map(id => 
      sessions.find(s => s.id === id)?.name
    ).join(', ');
    
    if (!window.confirm(`Are you sure you want to delete ${selectedSessions.size} session(s): ${sessionNames}? This action cannot be undone.`)) {
      return;
    }

    try {
      const deletePromises = Array.from(selectedSessions).map(sessionId =>
        chatHistoryService.deleteSession(sessionId, token)
      );
      
      await Promise.all(deletePromises);
      
      setSessions(prev => prev.filter(s => !selectedSessions.has(s.id)));
      setSelectedSessions(new Set());
      setIsSelectionMode(false);
      
      // If current session was deleted, create new one
      if (selectedSessions.has(currentSessionId)) {
        handleNewSession();
      }
      
      onShowToast(`${selectedSessions.size} session(s) deleted successfully`, 'success');
    } catch (error) {
      console.error('Failed to delete sessions:', error);
      onShowToast('Failed to delete some sessions', 'error');
    }
  };

  const toggleSessionSelection = (sessionId) => {
    setSelectedSessions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sessionId)) {
        newSet.delete(sessionId);
      } else {
        newSet.add(sessionId);
      }
      return newSet;
    });
  };

  const selectAllSessions = () => {
    if (selectedSessions.size === filteredSessions.length) {
      setSelectedSessions(new Set());
    } else {
      setSelectedSessions(new Set(filteredSessions.map(s => s.id)));
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    if (searchInputRef.current) {
      searchInputRef.current.focus();
    }
  };

  const toggleSortOrder = () => {
    setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <div className="animate-pulse space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Chat History
          </h3>
          <div className="flex items-center space-x-2">
            {/* View Mode Toggle */}
            <div className="flex items-center bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
              <button
                onClick={() => setViewMode('list')}
                className={`p-1.5 rounded ${viewMode === 'list' 
                  ? 'bg-white dark:bg-gray-600 text-primary-600 dark:text-primary-400 shadow-sm' 
                  : 'text-gray-500 dark:text-gray-400'}`}
                title="List view"
              >
                <i className="fas fa-list text-sm"></i>
              </button>
              <button
                onClick={() => setViewMode('grid')}
                className={`p-1.5 rounded ${viewMode === 'grid' 
                  ? 'bg-white dark:bg-gray-600 text-primary-600 dark:text-primary-400 shadow-sm' 
                  : 'text-gray-500 dark:text-gray-400'}`}
                title="Grid view"
              >
                <i className="fas fa-th text-sm"></i>
              </button>
            </div>

            {/* Selection Mode Toggle */}
            <button
              onClick={() => {
                setIsSelectionMode(!isSelectionMode);
                setSelectedSessions(new Set());
              }}
              className={`p-2 rounded-lg transition-colors ${isSelectionMode 
                ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400' 
                : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
              title="Selection mode"
            >
              <i className="fas fa-check-square text-sm"></i>
            </button>

            {/* New Chat Button */}
            <button
              onClick={handleNewSession}
              disabled={isCreating}
              className="p-2 text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-lg transition-colors"
              title="New Chat"
            >
              {isCreating ? (
                <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <i className="fas fa-plus text-sm"></i>
              )}
            </button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="space-y-3">
          {/* Search Bar */}
          <div className="relative">
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search sessions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
            />
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <i className="fas fa-search text-gray-400 text-sm"></i>
            </div>
            {searchQuery && (
              <button
                onClick={clearSearch}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <i className="fas fa-times text-sm"></i>
              </button>
            )}
          </div>

          {/* Sort and Filter Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="text-xs px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-primary-500"
              >
                <option value="updated">Last Updated</option>
                <option value="created">Created Date</option>
                <option value="name">Name</option>
                <option value="messages">Message Count</option>
              </select>
              
              <button
                onClick={toggleSortOrder}
                className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
                title={`Sort ${sortOrder === 'asc' ? 'descending' : 'ascending'}`}
              >
                <i className={`fas fa-sort-${sortOrder === 'asc' ? 'up' : 'down'} text-xs`}></i>
              </button>
            </div>

            {/* Session Count */}
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {filteredSessions.length} of {sessions.length} sessions
            </span>
          </div>

          {/* Selection Mode Controls */}
          {isSelectionMode && (
            <div className="flex items-center justify-between p-2 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
              <div className="flex items-center space-x-3">
                <button
                  onClick={selectAllSessions}
                  className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
                >
                  {selectedSessions.size === filteredSessions.length ? 'Deselect All' : 'Select All'}
                </button>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {selectedSessions.size} selected
                </span>
              </div>
              
              {selectedSessions.size > 0 && (
                <button
                  onClick={handleBulkDelete}
                  className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                >
                  <i className="fas fa-trash mr-1"></i>
                  Delete Selected
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto">
        {filteredSessions.length === 0 ? (
          <div className="p-4 text-center">
            {sessions.length === 0 ? (
              <>
                <div className="text-gray-400 dark:text-gray-500 mb-4">
                  <i className="fas fa-comments text-4xl"></i>
                </div>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  No chat sessions yet
                </p>
                <button
                  onClick={handleNewSession}
                  disabled={isCreating}
                  className="btn-primary"
                >
                  Start Your First Chat
                </button>
              </>
            ) : (
              <>
                <div className="text-gray-400 dark:text-gray-500 mb-4">
                  <i className="fas fa-search text-4xl"></i>
                </div>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  No sessions match your search
                </p>
                <button
                  onClick={clearSearch}
                  className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
                >
                  Clear search
                </button>
              </>
            )}
          </div>
        ) : (
          <div className={`p-2 ${viewMode === 'grid' ? 'grid grid-cols-1 gap-2' : 'space-y-1'}`}>
            {filteredSessions.map((session) => (
              <div
                key={session.id}
                className={`group relative rounded-lg cursor-pointer transition-all duration-200 ${
                  session.id === currentSessionId
                    ? 'bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 shadow-sm'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-700 border border-transparent'
                } ${isSelectionMode ? 'pr-10' : ''} ${viewMode === 'grid' ? 'p-4' : 'p-3'}`}
                onClick={() => {
                  if (isSelectionMode) {
                    toggleSessionSelection(session.id);
                  } else {
                    onSelectSession(session.id);
                  }
                }}
              >
                {/* Selection Checkbox */}
                {isSelectionMode && (
                  <div className="absolute top-3 right-3">
                    <input
                      type="checkbox"
                      checked={selectedSessions.has(session.id)}
                      onChange={() => toggleSessionSelection(session.id)}
                      className="w-4 h-4 text-primary-600 bg-gray-100 border-gray-300 rounded focus:ring-primary-500 dark:focus:ring-primary-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                    />
                  </div>
                )}

                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h4 className={`font-medium text-gray-900 dark:text-gray-100 truncate ${
                      viewMode === 'grid' ? 'text-base mb-2' : 'text-sm'
                    }`}>
                      {session.name}
                    </h4>
                    
                    <div className={`flex items-center text-gray-500 dark:text-gray-400 ${
                      viewMode === 'grid' ? 'text-sm space-x-4' : 'text-xs mt-1'
                    }`}>
                      <div className="flex items-center">
                        <i className="fas fa-comment-dots mr-1"></i>
                        <span>{session.message_count} messages</span>
                      </div>
                      {viewMode === 'grid' && (
                        <div className="flex items-center">
                          <i className="fas fa-clock mr-1"></i>
                          <span>{formatTime(session.last_message_time || session.updated_at)}</span>
                        </div>
                      )}
                      {viewMode === 'list' && (
                        <>
                          <span className="mx-1">•</span>
                          <span>{formatTime(session.last_message_time || session.updated_at)}</span>
                        </>
                      )}
                    </div>

                    {viewMode === 'grid' && (
                      <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
                        Created {formatTime(session.created_at)}
                      </div>
                    )}
                  </div>
                  
                  {/* Action Buttons */}
                  {!isSelectionMode && (
                    <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowRenameModal(session.id);
                          setNewSessionName(session.name);
                        }}
                        className="p-1.5 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 rounded transition-colors"
                        title="Rename session"
                      >
                        <i className="fas fa-edit text-xs"></i>
                      </button>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteSession(session.id, session.name);
                        }}
                        className="p-1.5 text-gray-400 hover:text-red-600 dark:hover:text-red-400 rounded transition-colors"
                        title="Delete session"
                      >
                        <i className="fas fa-trash text-xs"></i>
                      </button>
                    </div>
                  )}
                </div>

                {/* Active Indicator */}
                {session.id === currentSessionId && (
                  <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-1 h-8 bg-primary-600 rounded-r"></div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Rename Modal */}
      {showRenameModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Rename Session
              </h3>
              
              <input
                type="text"
                value={newSessionName}
                onChange={(e) => setNewSessionName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="Enter new session name"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleRenameSession(showRenameModal, newSessionName);
                  } else if (e.key === 'Escape') {
                    setShowRenameModal(null);
                    setNewSessionName('');
                  }
                }}
              />
              
              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={() => {
                    setShowRenameModal(null);
                    setNewSessionName('');
                  }}
                  className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
                  disabled={isRenaming}
                >
                  Cancel
                </button>
                
                <button
                  onClick={() => handleRenameSession(showRenameModal, newSessionName)}
                  disabled={isRenaming || !newSessionName.trim()}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isRenaming ? (
                    <div className="flex items-center">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Renaming...
                    </div>
                  ) : (
                    'Rename'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatHistory;