import React, { createContext, useContext, useReducer, useEffect } from 'react';

const SettingsContext = createContext();

const defaultSettings = {
  theme: 'auto',
  fontSize: 'medium',
  sound: true,
  notifications: true
};

function settingsReducer(state, action) {
  switch (action.type) {
    case 'SET_THEME':
      return { ...state, theme: action.payload };
    case 'SET_FONT_SIZE':
      return { ...state, fontSize: action.payload };
    case 'SET_SOUND':
      return { ...state, sound: action.payload };
    case 'SET_NOTIFICATIONS':
      return { ...state, notifications: action.payload };
    case 'LOAD_SETTINGS':
      return { ...state, ...action.payload };
    default:
      return state;
  }
}

export function SettingsProvider({ children }) {
  const [settings, dispatch] = useReducer(settingsReducer, defaultSettings);

  useEffect(() => {
    // Load settings from localStorage
    try {
      const saved = localStorage.getItem('chatbot-settings');
      if (saved) {
        dispatch({ type: 'LOAD_SETTINGS', payload: JSON.parse(saved) });
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }, []);

  useEffect(() => {
    // Save settings to localStorage
    try {
      localStorage.setItem('chatbot-settings', JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save settings:', error);
    }
  }, [settings]);

  const updateSettings = (key, value) => {
    switch (key) {
      case 'theme':
        dispatch({ type: 'SET_THEME', payload: value });
        break;
      case 'fontSize':
        dispatch({ type: 'SET_FONT_SIZE', payload: value });
        break;
      case 'sound':
        dispatch({ type: 'SET_SOUND', payload: value });
        break;
      case 'notifications':
        dispatch({ type: 'SET_NOTIFICATIONS', payload: value });
        break;
      default:
        break;
    }
  };

  const value = {
    settings,
    updateSettings
  };

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}