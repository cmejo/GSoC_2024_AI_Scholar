import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { chatService } from '../services/chatService';
import { chatHistoryService } from '../services/chatHistoryService';
import { useAuth } from './AuthContext';

const ChatContext = createContext();

const initialState = {
  messages: [],
  currentSessionId: null,
  isTyping: false,
  isConnected: false,
  connectionStatus: 'disconnected',
  botStatus: 'offline',
  isLoadingSession: false
};

function chatReducer(state, action) {
  switch (action.type) {
    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.payload]
      };
    case 'SET_MESSAGES':
      return {
        ...state,
        messages: action.payload
      };
    case 'SET_CURRENT_SESSION':
      return {
        ...state,
        currentSessionId: action.payload
      };
    case 'SET_TYPING':
      return {
        ...state,
        isTyping: action.payload
      };
    case 'SET_CONNECTION_STATUS':
      return {
        ...state,
        isConnected: action.payload,
        connectionStatus: action.payload ? 'connected' : 'disconnected'
      };
    case 'SET_BOT_STATUS':
      return {
        ...state,
        botStatus: action.payload ? 'online' : 'offline'
      };
    case 'SET_LOADING_SESSION':
      return {
        ...state,
        isLoadingSession: action.payload
      };
    case 'CLEAR_MESSAGES':
      return {
        ...state,
        messages: []
      };
    default:
      return state;
  }
}

export function ChatProvider({ children }) {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const { token, isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      // Initialize chat service
      chatService.initialize({
        onConnectionChange: (connected) => {
          dispatch({ type: 'SET_CONNECTION_STATUS', payload: connected });
        },
        onTypingChange: (typing) => {
          dispatch({ type: 'SET_TYPING', payload: typing });
        },
        onMessage: (message) => {
          dispatch({ type: 'ADD_MESSAGE', payload: message });
        }
      });

      // Check health status
      chatService.checkHealth().then((healthy) => {
        dispatch({ type: 'SET_BOT_STATUS', payload: healthy });
      });

      // Create initial session if none exists
      if (!state.currentSessionId) {
        createNewSession();
      }
    }

    return () => {
      if (isAuthenticated) {
        chatService.disconnect();
      }
    };
  }, [isAuthenticated]);

  const createNewSession = async () => {
    try {
      const response = await chatHistoryService.createSession(token);
      dispatch({ type: 'SET_CURRENT_SESSION', payload: response.session_id });
      dispatch({ type: 'CLEAR_MESSAGES' });
      
      // Add welcome message
      const welcomeMessage = {
        id: Date.now(),
        content: "👋 Hello! I'm your AI assistant. How can I help you today?",
        type: 'bot',
        timestamp: new Date().toISOString()
      };
      dispatch({ type: 'ADD_MESSAGE', payload: welcomeMessage });
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const loadSession = async (sessionId) => {
    try {
      dispatch({ type: 'SET_LOADING_SESSION', payload: true });
      const messages = await chatHistoryService.getSessionMessages(sessionId, token);
      
      dispatch({ type: 'SET_CURRENT_SESSION', payload: sessionId });
      dispatch({ type: 'SET_MESSAGES', payload: messages });
    } catch (error) {
      console.error('Failed to load session:', error);
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING_SESSION', payload: false });
    }
  };

  const sendMessage = async (content) => {
    const userMessage = {
      id: Date.now(),
      content,
      type: 'user',
      timestamp: new Date().toISOString()
    };

    dispatch({ type: 'ADD_MESSAGE', payload: userMessage });
    
    try {
      const response = await chatService.sendMessage(content, state.currentSessionId, token);
      
      if (response.success) {
        const botMessage = {
          id: Date.now() + 1,
          content: response.response,
          type: 'bot',
          timestamp: response.timestamp || new Date().toISOString()
        };
        dispatch({ type: 'ADD_MESSAGE', payload: botMessage });
        
        // Update session ID if it was created
        if (response.session_id && response.session_id !== state.currentSessionId) {
          dispatch({ type: 'SET_CURRENT_SESSION', payload: response.session_id });
        }
      } else {
        throw new Error(response.error || 'Failed to get response');
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        content: 'Sorry, I encountered an error. Please try again.',
        type: 'bot',
        timestamp: new Date().toISOString()
      };
      dispatch({ type: 'ADD_MESSAGE', payload: errorMessage });
      throw error;
    }
  };

  const clearMessages = () => {
    dispatch({ type: 'CLEAR_MESSAGES' });
  };

  const value = {
    ...state,
    sendMessage,
    clearMessages,
    createNewSession,
    loadSession
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}