/**
 * Token Manager Utility
 * Handles JWT token lifecycle, expiration checking, and refresh logic
 */

class TokenManager {
  constructor() {
    this.TOKEN_KEY = 'chatbot_token';
    this.EXPIRY_KEY = 'chatbot_session_expiry';
    this.REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes in milliseconds
    this.refreshPromise = null;
  }

  /**
   * Store token and expiration time
   */
  setToken(token, expiresIn = 24 * 60 * 60 * 1000) { // Default 24 hours
    const expiryTime = new Date(Date.now() + expiresIn);
    
    localStorage.setItem(this.TOKEN_KEY, token);
    localStorage.setItem(this.EXPIRY_KEY, expiryTime.toISOString());
    
    return expiryTime;
  }

  /**
   * Get current token
   */
  getToken() {
    return localStorage.getItem(this.TOKEN_KEY);
  }

  /**
   * Get token expiration time
   */
  getTokenExpiry() {
    const expiry = localStorage.getItem(this.EXPIRY_KEY);
    return expiry ? new Date(expiry) : null;
  }

  /**
   * Check if token exists
   */
  hasToken() {
    return !!this.getToken();
  }

  /**
   * Check if token is expired
   */
  isTokenExpired() {
    const expiry = this.getTokenExpiry();
    if (!expiry) return true;
    
    return new Date() >= expiry;
  }

  /**
   * Check if token needs refresh (within threshold)
   */
  needsRefresh() {
    const expiry = this.getTokenExpiry();
    if (!expiry) return false;
    
    const timeUntilExpiry = expiry.getTime() - Date.now();
    return timeUntilExpiry <= this.REFRESH_THRESHOLD && timeUntilExpiry > 0;
  }

  /**
   * Get time until token expires (in milliseconds)
   */
  getTimeUntilExpiry() {
    const expiry = this.getTokenExpiry();
    if (!expiry) return 0;
    
    return Math.max(0, expiry.getTime() - Date.now());
  }

  /**
   * Get time until refresh needed (in milliseconds)
   */
  getTimeUntilRefresh() {
    const expiry = this.getTokenExpiry();
    if (!expiry) return 0;
    
    const refreshTime = expiry.getTime() - this.REFRESH_THRESHOLD;
    return Math.max(0, refreshTime - Date.now());
  }

  /**
   * Format time remaining as human-readable string
   */
  formatTimeRemaining() {
    const timeLeft = this.getTimeUntilExpiry();
    if (timeLeft <= 0) return 'Expired';
    
    const hours = Math.floor(timeLeft / (60 * 60 * 1000));
    const minutes = Math.floor((timeLeft % (60 * 60 * 1000)) / (60 * 1000));
    const seconds = Math.floor((timeLeft % (60 * 1000)) / 1000);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  }

  /**
   * Clear token and expiration
   */
  clearToken() {
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.EXPIRY_KEY);
    
    // Clear any pending refresh
    if (this.refreshPromise) {
      this.refreshPromise = null;
    }
  }

  /**
   * Decode JWT token payload (without verification)
   */
  decodeToken(token = null) {
    const tokenToUse = token || this.getToken();
    if (!tokenToUse) return null;
    
    try {
      const parts = tokenToUse.split('.');
      if (parts.length !== 3) return null;
      
      const payload = JSON.parse(atob(parts[1]));
      return payload;
    } catch (error) {
      console.error('Error decoding token:', error);
      return null;
    }
  }

  /**
   * Get token expiration from JWT payload
   */
  getTokenExpiryFromJWT(token = null) {
    const payload = this.decodeToken(token);
    if (!payload || !payload.exp) return null;
    
    return new Date(payload.exp * 1000); // JWT exp is in seconds
  }

  /**
   * Validate token format and expiration
   */
  isValidToken(token = null) {
    const tokenToUse = token || this.getToken();
    if (!tokenToUse) return false;
    
    const payload = this.decodeToken(tokenToUse);
    if (!payload) return false;
    
    // Check if token is expired based on JWT payload
    if (payload.exp && new Date(payload.exp * 1000) <= new Date()) {
      return false;
    }
    
    return true;
  }

  /**
   * Get user info from token
   */
  getUserFromToken(token = null) {
    const payload = this.decodeToken(token);
    if (!payload) return null;
    
    return {
      id: payload.user_id,
      username: payload.username,
      exp: payload.exp
    };
  }

  /**
   * Schedule automatic token refresh
   */
  scheduleRefresh(refreshCallback) {
    const timeUntilRefresh = this.getTimeUntilRefresh();
    
    if (timeUntilRefresh > 0) {
      return setTimeout(() => {
        if (this.needsRefresh() && !this.isTokenExpired()) {
          refreshCallback();
        }
      }, timeUntilRefresh);
    }
    
    return null;
  }

  /**
   * Get token status summary
   */
  getTokenStatus() {
    const token = this.getToken();
    const expiry = this.getTokenExpiry();
    
    if (!token) {
      return { status: 'missing', message: 'No token found' };
    }
    
    if (!this.isValidToken()) {
      return { status: 'invalid', message: 'Token is invalid or malformed' };
    }
    
    if (this.isTokenExpired()) {
      return { status: 'expired', message: 'Token has expired' };
    }
    
    if (this.needsRefresh()) {
      return { 
        status: 'refresh_needed', 
        message: `Token expires in ${this.formatTimeRemaining()}`,
        timeRemaining: this.getTimeUntilExpiry()
      };
    }
    
    return { 
      status: 'valid', 
      message: `Token valid for ${this.formatTimeRemaining()}`,
      timeRemaining: this.getTimeUntilExpiry()
    };
  }
}

// Export singleton instance
export const tokenManager = new TokenManager();
export default tokenManager;