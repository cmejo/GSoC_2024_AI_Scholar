import React from 'react';
import { formatTime, formatMessage } from '../utils/messageUtils';

function Message({ message }) {
  const { content, type, timestamp } = message;
  const isUser = type === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-slide-up`}>
      <div className={`flex max-w-xs sm:max-w-md lg:max-w-lg xl:max-w-xl ${
        isUser ? 'flex-row-reverse' : 'flex-row'
      } space-x-2`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-primary-600 text-white' 
            : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
        }`}>
          <i className={`fas ${isUser ? 'fa-user' : 'fa-robot'} text-sm`}></i>
        </div>
        
        {/* Message Content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <div className={`px-4 py-2 rounded-2xl message-bubble ${
            isUser
              ? 'bg-primary-600 text-white rounded-br-md'
              : 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-600 rounded-bl-md'
          }`}>
            <div 
              className="text-sm leading-relaxed"
              dangerouslySetInnerHTML={{ __html: formatMessage(content) }}
            />
          </div>
          
          {/* Timestamp */}
          <div className={`text-xs text-gray-500 dark:text-gray-400 mt-1 ${
            isUser ? 'text-right' : 'text-left'
          }`}>
            {formatTime(timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Message;