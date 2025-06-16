import React, { useState, useRef, useEffect } from 'react';
import { useSettings } from '../hooks/useSettings';
import { chatService } from '../services/chatService';

function ChatInput({ onSendMessage }) {
  const [message, setMessage] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const textareaRef = useRef(null);
  const { settings } = useSettings();
  const maxLength = 2000;

  useEffect(() => {
    // Auto-resize textarea
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  }, [message]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const trimmedMessage = message.trim();
    if (!trimmedMessage || trimmedMessage.length > maxLength || isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    
    try {
      // Play send sound
      if (settings.sound) {
        chatService.playSound('send');
      }
      
      await onSendMessage(trimmedMessage);
      setMessage('');
      
      // Play receive sound after a delay
      if (settings.sound) {
        setTimeout(() => {
          chatService.playSound('receive');
        }, 1000);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsSubmitting(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleChange = (e) => {
    setMessage(e.target.value);
  };

  const isDisabled = !message.trim() || message.length > maxLength || isSubmitting;

  return (
    <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4">
      <form onSubmit={handleSubmit} className="flex flex-col space-y-2">
        <div className="flex items-end space-x-2">
          {/* Attach Button */}
          <button
            type="button"
            className="flex-shrink-0 p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Attach file"
          >
            <i className="fas fa-paperclip text-lg"></i>
          </button>
          
          {/* Message Input */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-2xl bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
              rows={1}
              maxLength={maxLength}
              disabled={isSubmitting}
            />
            
            {/* Emoji Button */}
            <button
              type="button"
              className="absolute right-3 bottom-3 p-1 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
              title="Add emoji"
            >
              <i className="fas fa-smile"></i>
            </button>
          </div>
          
          {/* Send Button */}
          <button
            type="submit"
            disabled={isDisabled}
            className={`flex-shrink-0 p-3 rounded-full transition-all duration-200 ${
              isDisabled
                ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                : 'bg-primary-600 hover:bg-primary-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
            }`}
            title="Send message"
          >
            {isSubmitting ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <i className="fas fa-paper-plane text-lg"></i>
            )}
          </button>
        </div>
        
        {/* Footer */}
        <div className="flex justify-between items-center text-xs text-gray-500 dark:text-gray-400">
          <div>
            <span className={message.length > maxLength * 0.9 ? 'text-red-500' : ''}>
              {message.length}
            </span>
            /{maxLength}
          </div>
          <div className="text-right">
            Press Enter to send, Shift+Enter for new line
          </div>
        </div>
      </form>
    </div>
  );
}

export default ChatInput;