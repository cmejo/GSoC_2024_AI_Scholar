import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ChatProvider } from './context/ChatContext';
import { SettingsProvider } from './context/SettingsContext';
import { useSettings } from './hooks/useSettings';
import ProtectedRoute from './components/ProtectedRoute';
import PublicRoute from './components/PublicRoute';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ChatPage from './pages/ChatPage';
import HistoryPage from './pages/HistoryPage';

function ThemeProvider({ children }) {
  const { settings } = useSettings();

  useEffect(() => {
    // Apply theme
    const root = document.documentElement;
    if (settings.theme === 'dark' || (settings.theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [settings.theme]);

  useEffect(() => {
    // Apply font size
    const root = document.documentElement;
    const sizes = {
      small: '14px',
      medium: '16px',
      large: '18px'
    };
    root.style.fontSize = sizes[settings.fontSize] || sizes.medium;
  }, [settings.fontSize]);

  return children;
}

function App() {
  return (
    <AuthProvider>
      <SettingsProvider>
        <ThemeProvider>
          <ChatProvider>
            <Router>
              <Routes>
                {/* Public Routes */}
                <Route 
                  path="/login" 
                  element={
                    <PublicRoute>
                      <LoginPage />
                    </PublicRoute>
                  } 
                />
                <Route 
                  path="/register" 
                  element={
                    <PublicRoute>
                      <RegisterPage />
                    </PublicRoute>
                  } 
                />

                {/* Protected Routes */}
                <Route 
                  path="/chat" 
                  element={
                    <ProtectedRoute>
                      <ChatPage />
                    </ProtectedRoute>
                  } 
                />
                <Route 
                  path="/history" 
                  element={
                    <ProtectedRoute>
                      <HistoryPage />
                    </ProtectedRoute>
                  } 
                />

                {/* Default redirect */}
                <Route path="/" element={<Navigate to="/chat" replace />} />
                
                {/* Catch all route */}
                <Route path="*" element={<Navigate to="/chat" replace />} />
              </Routes>
            </Router>
          </ChatProvider>
        </ThemeProvider>
      </SettingsProvider>
    </AuthProvider>
  );
}

export default App;