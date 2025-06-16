import React from 'react';

function LoadingScreen() {
  return (
    <div className="fixed inset-0 bg-white dark:bg-gray-900 flex items-center justify-center z-50">
      <div className="text-center">
        <div className="relative w-16 h-16 mx-auto mb-6">
          <div className="absolute inset-0 border-4 border-primary-200 dark:border-primary-800 rounded-full"></div>
          <div className="absolute inset-0 border-4 border-primary-600 rounded-full border-t-transparent animate-spin"></div>
        </div>
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
          AI Chatbot
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Initializing...
        </p>
      </div>
    </div>
  );
}

export default LoadingScreen;