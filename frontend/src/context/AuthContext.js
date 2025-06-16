import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react';
import { authService } from '../services/authService';
import { tokenManager } from '../utils/tokenManager';

const AuthContext = createContext();

const initialState = {
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
  sessionExpiry: null,
  refreshTimer: null
};

function authReducer(state, action) {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        sessionExpiry: action.payload.sessionExpiry,
        isAuthenticated: true,
        isLoading: false,
        error: null
      };
    case 'TOKEN_REFRESHED':
      return {
        ...state,
        token: action.payload.token,
        sessionExpiry: action.payload.sessionExpiry,
        error: null
      };
    case 'UPDATE_USER':
      return {
        ...state,
        user: { ...state.user, ...action.payload }
      };
    case 'SET_REFRESH_TIMER':
      return {
        ...state,
        refreshTimer: action.payload
      };
    case 'LOGOUT':
      return {
        ...state,
        user: null,
        token: null,
        sessionExpiry: null,
        refreshTimer: null,
        isAuthenticated: false,
        isLoading: false,
        error: null
      };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    default:
      return state;
  }
}

export function AuthProvider({ children }) {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Token refresh function with enhanced error handling
  const refreshToken = useCallback(async (force = false) => {
    try {
      // Prevent multiple simultaneous refresh attempts
      if (state.refreshTimer && !force) {
        console.log('Refresh already in progress');
        return state.token;
      }

      const currentToken = tokenManager.getToken();
      if (!currentToken) {
        throw new Error('No token available for refresh');
      }

      // Check if token is still valid for refresh
      if (tokenManager.isTokenExpired() && !force) {
        throw new Error('Token is expired and cannot be refreshed');
      }

      console.log('Refreshing token...');
      const response = await authService.refreshToken(currentToken);
      const newToken = response.token;
      
      // Use token manager to store new token
      const sessionExpiry = tokenManager.setToken(newToken);
      
      dispatch({
        type: 'TOKEN_REFRESHED',
        payload: { token: newToken, sessionExpiry }
      });

      // Schedule next refresh
      scheduleTokenRefresh(sessionExpiry);
      
      console.log('Token refreshed successfully');
      return newToken;
    } catch (error) {
      console.error('Token refresh failed:', error);
      
      // Clear invalid token
      tokenManager.clearToken();
      
      // If refresh fails, logout user
      dispatch({ type: 'LOGOUT' });
      throw error;
    }
  }, [state.refreshTimer]);

  // Schedule token refresh with enhanced timing
  const scheduleTokenRefresh = useCallback((sessionExpiry) => {
    // Clear existing timer
    if (state.refreshTimer) {
      clearTimeout(state.refreshTimer);
      dispatch({ type: 'SET_REFRESH_TIMER', payload: null });
    }

    if (!sessionExpiry) return;

    const timeUntilRefresh = tokenManager.getTimeUntilRefresh();
    
    if (timeUntilRefresh > 0) {
      console.log(`Scheduling token refresh in ${Math.round(timeUntilRefresh / 1000)} seconds`);
      
      const timer = setTimeout(() => {
        if (tokenManager.needsRefresh() && !tokenManager.isTokenExpired()) {
          refreshToken();
        }
      }, timeUntilRefresh);

      dispatch({
        type: 'SET_REFRESH_TIMER',
        payload: timer
      });
    } else if (tokenManager.needsRefresh()) {
      // Immediate refresh needed
      console.log('Immediate token refresh needed');
      refreshToken();
    }
  }, [state.refreshTimer, refreshToken]);

  // Check if session is expired using token manager
  const isSessionExpired = useCallback(() => {
    return tokenManager.isTokenExpired();
  }, []);

  // Get token status
  const getTokenStatus = useCallback(() => {
    return tokenManager.getTokenStatus();
  }, []);

  // Check if token needs refresh
  const needsTokenRefresh = useCallback(() => {
    return tokenManager.needsRefresh();
  }, []);

  useEffect(() => {
    // Check for existing token on app start
    const initializeAuth = async () => {
      const token = tokenManager.getToken();
      const sessionExpiry = tokenManager.getTokenExpiry();
      
      if (token) {
        try {
          // Check token validity
          if (!tokenManager.isValidToken()) {
            console.log('Invalid token found, clearing...');
            tokenManager.clearToken();
            dispatch({ type: 'SET_LOADING', payload: false });
            return;
          }

          // Check if session is expired
          if (tokenManager.isTokenExpired()) {
            console.log('Expired token found, attempting refresh...');
            try {
              await refreshToken(true); // Force refresh
              return;
            } catch (error) {
              console.log('Token refresh failed during initialization');
              tokenManager.clearToken();
              dispatch({ type: 'SET_LOADING', payload: false });
              return;
            }
          }

          // Validate token with server
          const user = await authService.getCurrentUser(token);
          
          dispatch({
            type: 'LOGIN_SUCCESS',
            payload: { user, token, sessionExpiry }
          });

          // Schedule token refresh if needed
          if (sessionExpiry) {
            scheduleTokenRefresh(sessionExpiry);
          }
        } catch (error) {
          console.error('Token validation failed:', error);
          tokenManager.clearToken();
          dispatch({ type: 'SET_LOADING', payload: false });
        }
      } else {
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    };

    initializeAuth();
  }, [refreshToken, scheduleTokenRefresh]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (state.refreshTimer) {
        clearTimeout(state.refreshTimer);
      }
    };
  }, [state.refreshTimer]);

  const login = async (username, password) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERROR' });
      
      const response = await authService.login(username, password);
      
      // Use token manager to store token
      const sessionExpiry = tokenManager.setToken(response.token);
      
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: response.user,
          token: response.token,
          sessionExpiry
        }
      });

      // Schedule token refresh
      scheduleTokenRefresh(sessionExpiry);
      
      console.log('Login successful, token expires at:', sessionExpiry);
      return response;
    } catch (error) {
      dispatch({
        type: 'SET_ERROR',
        payload: error.message || 'Login failed'
      });
      throw error;
    }
  };

  const register = async (username, email, password, additionalData = {}) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERROR' });
      
      const response = await authService.register(username, email, password, additionalData);
      
      // Use token manager to store token
      const sessionExpiry = tokenManager.setToken(response.token);
      
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: response.user,
          token: response.token,
          sessionExpiry
        }
      });

      // Schedule token refresh
      scheduleTokenRefresh(sessionExpiry);
      
      console.log('Registration successful, token expires at:', sessionExpiry);
      return response;
    } catch (error) {
      dispatch({
        type: 'SET_ERROR',
        payload: error.message || 'Registration failed'
      });
      throw error;
    }
  };

  const logout = async () => {
    try {
      // Clear refresh timer
      if (state.refreshTimer) {
        clearTimeout(state.refreshTimer);
      }

      const token = tokenManager.getToken();
      if (token) {
        await authService.logout(token);
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Always clear local state and storage using token manager
      tokenManager.clearToken();
      dispatch({ type: 'LOGOUT' });
      console.log('Logout completed');
    }
  };

  // Update user information
  const updateUser = (userData) => {
    dispatch({
      type: 'UPDATE_USER',
      payload: userData
    });
  };

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const value = {
    ...state,
    login,
    register,
    logout,
    updateUser,
    refreshToken,
    isSessionExpired,
    getTokenStatus,
    needsTokenRefresh,
    clearError
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}