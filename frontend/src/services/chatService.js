import axios from 'axios';
import { io } from 'socket.io-client';

class ChatService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.callbacks = {};
    
    // Configure axios defaults
    axios.defaults.baseURL = process.env.REACT_APP_API_URL || '';
    axios.defaults.timeout = 30000;
  }

  initialize(callbacks) {
    this.callbacks = callbacks;
    this.connectSocket();
  }

  connectSocket() {
    try {
      this.socket = io(process.env.REACT_APP_API_URL || '', {
        transports: ['websocket', 'polling'],
        timeout: 20000,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 5
      });

      this.socket.on('connect', () => {
        console.log('🔌 Connected to server');
        this.isConnected = true;
        this.callbacks.onConnectionChange?.(true);
      });

      this.socket.on('disconnect', () => {
        console.log('🔌 Disconnected from server');
        this.isConnected = false;
        this.callbacks.onConnectionChange?.(false);
      });

      this.socket.on('chat_response', (data) => {
        this.callbacks.onTypingChange?.(false);
        if (data.success) {
          const message = {
            id: Date.now(),
            content: data.response,
            type: 'bot',
            timestamp: data.timestamp || new Date().toISOString()
          };
          this.callbacks.onMessage?.(message);
        }
      });

      this.socket.on('typing', (data) => {
        this.callbacks.onTypingChange?.(data.typing);
      });

      this.socket.on('error', (data) => {
        console.error('Socket error:', data);
        this.callbacks.onTypingChange?.(false);
      });

    } catch (error) {
      console.error('Failed to initialize socket:', error);
      this.isConnected = false;
      this.callbacks.onConnectionChange?.(false);
    }
  }

  async sendMessage(message, sessionId = null, token = null) {
    // Try WebSocket first if connected
    if (this.isConnected && this.socket) {
      return this.sendMessageViaSocket(message, sessionId, token);
    } else {
      // Fallback to REST API
      return this.sendMessageViaAPI(message, sessionId, token);
    }
  }

  sendMessageViaSocket(message, sessionId, token) {
    return new Promise((resolve, reject) => {
      if (!this.socket || !this.isConnected) {
        reject(new Error('Socket not connected'));
        return;
      }

      // Set up one-time listeners for this message
      const timeout = setTimeout(() => {
        reject(new Error('Message timeout'));
      }, 30000);

      const onResponse = (data) => {
        clearTimeout(timeout);
        this.socket.off('chat_response', onResponse);
        this.socket.off('error', onError);
        resolve(data);
      };

      const onError = (error) => {
        clearTimeout(timeout);
        this.socket.off('chat_response', onResponse);
        this.socket.off('error', onError);
        reject(new Error(error.message || 'Socket error'));
      };

      this.socket.on('chat_response', onResponse);
      this.socket.on('error', onError);

      // Send the message with authentication
      this.socket.emit('chat_message', {
        message,
        session_id: sessionId,
        token
      });

      // Show typing indicator
      this.callbacks.onTypingChange?.(true);
    });
  }

  async sendMessageViaAPI(message, sessionId, token) {
    try {
      this.callbacks.onTypingChange?.(true);
      
      const headers = {};
      if (token) {
        headers.Authorization = `Bearer ${token}`;
      }
      
      const response = await axios.post('/api/chat', {
        message,
        session_id: sessionId
      }, { headers });

      this.callbacks.onTypingChange?.(false);
      return response.data;
    } catch (error) {
      this.callbacks.onTypingChange?.(false);
      console.error('API request failed:', error);
      
      if (error.response) {
        return {
          success: false,
          error: error.response.data?.error || 'Server error'
        };
      } else if (error.request) {
        return {
          success: false,
          error: 'Network error - please check your connection'
        };
      } else {
        return {
          success: false,
          error: 'Request failed'
        };
      }
    }
  }

  async checkHealth() {
    try {
      const response = await axios.get('/api/health');
      return response.data.status === 'healthy' && response.data.ollama_connected;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnected = false;
  }

  playSound(type) {
    // Create audio context for sound effects
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      if (type === 'send') {
        oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.1);
      } else if (type === 'receive') {
        oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 0.1);
      }
      
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
      
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.1);
      
    } catch (error) {
      // Silently fail if audio context is not supported
    }
  }
}

export const chatService = new ChatService();