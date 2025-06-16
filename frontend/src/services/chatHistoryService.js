import axios from 'axios';

class ChatHistoryService {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || '';
  }

  async getSessions(token) {
    try {
      const response = await axios.get(`${this.baseURL}/api/chat/sessions`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data.sessions;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to get sessions');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Request failed');
      }
    }
  }

  async createSession(token, name = '') {
    try {
      const response = await axios.post(`${this.baseURL}/api/chat/sessions`, 
        { name },
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to create session');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Request failed');
      }
    }
  }

  async getSessionMessages(sessionId, token) {
    try {
      const response = await axios.get(`${this.baseURL}/api/chat/sessions/${sessionId}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data.messages;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to get messages');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Request failed');
      }
    }
  }

  async deleteSession(sessionId, token) {
    try {
      const response = await axios.delete(`${this.baseURL}/api/chat/sessions/${sessionId}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to delete session');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Request failed');
      }
    }
  }

  async renameSession(sessionId, newName, token) {
    try {
      const response = await axios.put(`${this.baseURL}/api/chat/sessions/${sessionId}`, 
        { name: newName },
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.message || 'Failed to rename session');
      } else if (error.request) {
        throw new Error('Network error - please check your connection');
      } else {
        throw new Error('Request failed');
      }
    }
  }
}

export const chatHistoryService = new ChatHistoryService();