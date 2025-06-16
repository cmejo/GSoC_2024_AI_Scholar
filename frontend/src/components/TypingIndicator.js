import React from 'react';

function TypingIndicator() {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex space-x-2">
        {/* Avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center">
          <i className="fas fa-robot text-sm text-gray-700 dark:text-gray-300"></i>
        </div>
        
        {/* Typing Bubble */}
        <div className="bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-2xl rounded-bl-md px-4 py-3">
          <div className="flex space-x-1 typing-dots">
            <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full"></span>
            <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full"></span>
            <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full"></span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TypingIndicator;