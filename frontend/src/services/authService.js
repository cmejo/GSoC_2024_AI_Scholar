import axios from 'axios';

class AuthService {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || '';
  }

  async login(username, password) {
    try {
      const response = await axios.post(`${this.baseURL}/api/auth/login`, {
        username,
        password
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Login failed');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Login request failed');
      }
    }
  }

  async register(username, email, password, additionalData = {}) {
    try {
      const response = await axios.post(`${this.baseURL}/api/auth/register`, {
        username,
        email,
        password,
        ...additionalData
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Registration failed');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Registration request failed');
      }
    }
  }

  async getCurrentUser(token) {
    try {
      const response = await axios.get(`${this.baseURL}/api/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data.user;
    } catch (error) {
      if (error.response?.status === 401) {
        throw new Error('Token expired');
      }
      throw new Error('Failed to get user information');
    }
  }

  async logout(token) {
    try {
      await axios.post(`${this.baseURL}/api/auth/logout`, {}, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
    } catch (error) {
      // Logout errors are not critical
      console.error('Logout error:', error);
    }
  }

  async refreshToken(token) {
    try {
      const response = await axios.post(`${this.baseURL}/api/auth/refresh`, {}, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Token refresh failed');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Token refresh request failed');
      }
    }
  }

  async updateProfile(profileData, token) {
    try {
      const response = await axios.put(`${this.baseURL}/api/auth/profile`, profileData, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to update profile');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Profile update request failed');
      }
    }
  }

  async changePassword(passwordData, token) {
    try {
      const response = await axios.put(`${this.baseURL}/api/auth/password`, passwordData, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to change password');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Password change request failed');
      }
    }
  }

  async getUserStats(token) {
    try {
      const response = await axios.get(`${this.baseURL}/api/auth/stats`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to get user stats');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Stats request failed');
      }
    }
  }

  async getUserSessions(token) {
    try {
      const response = await axios.get(`${this.baseURL}/api/auth/sessions`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data.sessions;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to get user sessions');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Sessions request failed');
      }
    }
  }

  async logoutAllSessions(token) {
    try {
      const response = await axios.post(`${this.baseURL}/api/auth/logout-all`, {}, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to logout from all sessions');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Logout all request failed');
      }
    }
  }

  async deleteAccount(token) {
    try {
      const response = await axios.delete(`${this.baseURL}/api/auth/account`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to delete account');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Account deletion request failed');
      }
    }
  }

  async checkUsernameAvailability(username) {
    try {
      const response = await axios.get(`${this.baseURL}/api/auth/check-username/${username}`);
      return response.data.available;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to check username availability');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Username check request failed');
      }
    }
  }

  async checkEmailAvailability(email) {
    try {
      const response = await axios.get(`${this.baseURL}/api/auth/check-email/${email}`);
      return response.data.available;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to check email availability');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Email check request failed');
      }
    }
  }

  // Helper method to get auth headers
  getAuthHeaders(token) {
    return {
      Authorization: `Bearer ${token}`
    };
  }
}

export const authService = new AuthService();