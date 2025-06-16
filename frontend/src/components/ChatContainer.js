import React, { useState } from 'react';
import ChatHeader from './ChatHeader';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import ChatHistory from './ChatHistory';
import { useChat } from '../context/ChatContext';

function ChatContainer({ onOpenSettings, onShowToast }) {
  const [showHistory, setShowHistory] = useState(false);
  const { sendMessage, clearMessages, createNewSession, loadSession, currentSessionId } = useChat();

  const handleSendMessage = async (message) => {
    try {
      await sendMessage(message);
    } catch (error) {
      onShowToast('Failed to send message. Please try again.', 'error');
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear the current chat?')) {
      clearMessages();
      createNewSession();
      onShowToast('New chat started', 'success');
    }
  };

  const handleSelectSession = async (sessionId) => {
    try {
      await loadSession(sessionId);
      setShowHistory(false); // Close sidebar on mobile
      onShowToast('Chat session loaded', 'success');
    } catch (error) {
      onShowToast('Failed to load chat session', 'error');
    }
  };

  const handleNewSession = (sessionId) => {
    setShowHistory(false); // Close sidebar on mobile
  };

  return (
    <div className="flex h-full">
      {/* Chat History Sidebar */}
      <div className={`${
        showHistory ? 'block' : 'hidden'
      } lg:block w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex-shrink-0`}>
        <ChatHistory 
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          onShowToast={onShowToast}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-h-0">
        <ChatHeader 
          onOpenSettings={onOpenSettings}
          onClearChat={handleClearChat}
          onToggleHistory={() => setShowHistory(!showHistory)}
          showHistory={showHistory}
        />
        
        <div className="flex-1 flex flex-col min-h-0">
          <ChatMessages />
          <ChatInput onSendMessage={handleSendMessage} />
        </div>
      </div>

      {/* Mobile Overlay */}
      {showHistory && (
        <div 
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setShowHistory(false)}
        />
      )}
    </div>
  );
}

export default ChatContainer;