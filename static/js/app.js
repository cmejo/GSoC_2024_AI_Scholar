/**
 * AI Chatbot Web Application
 * Responsive, mobile-ready JavaScript frontend
 */

class ChatbotApp {
    constructor() {
        this.config = window.CHATBOT_CONFIG || {};
        this.socket = null;
        this.conversationHistory = [];
        this.isTyping = false;
        this.settings = this.loadSettings();
        this.messageQueue = [];
        this.isConnected = false;
        
        // DOM elements
        this.elements = {};
        
        // Initialize app
        this.init();
    }
    
    /**
     * Initialize the application
     */
    async init() {
        try {
            // Cache DOM elements
            this.cacheElements();
            
            // Apply saved settings
            this.applySettings();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize WebSocket connection
            this.initializeSocket();
            
            // Check health status
            await this.checkHealth();
            
            // Hide loading screen
            this.hideLoadingScreen();
            
            // Focus input
            this.elements.messageInput.focus();
            
            console.log('🤖 Chatbot app initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize app:', error);
            this.showError('Failed to initialize application');
            this.hideLoadingScreen();
        }
    }
    
    /**
     * Cache DOM elements for better performance
     */
    cacheElements() {
        this.elements = {
            // Main containers
            loadingScreen: document.getElementById('loading-screen'),
            app: document.getElementById('app'),
            chatMessages: document.getElementById('chat-messages'),
            
            // Header elements
            botStatus: document.getElementById('bot-status'),
            settingsBtn: document.getElementById('settings-btn'),
            clearChatBtn: document.getElementById('clear-chat-btn'),
            menuBtn: document.getElementById('menu-btn'),
            
            // Input elements
            messageInput: document.getElementById('message-input'),
            sendBtn: document.getElementById('send-btn'),
            attachBtn: document.getElementById('attach-btn'),
            emojiBtn: document.getElementById('emoji-btn'),
            charCount: document.getElementById('char-count'),
            connectionStatus: document.getElementById('connection-status'),
            
            // Modal elements
            settingsModal: document.getElementById('settings-modal'),
            settingsClose: document.getElementById('settings-close'),
            themeSelect: document.getElementById('theme-select'),
            fontSizeSelect: document.getElementById('font-size-select'),
            soundToggle: document.getElementById('sound-toggle'),
            notificationsToggle: document.getElementById('notifications-toggle'),
            
            // Toast elements
            errorToast: document.getElementById('error-toast'),
            successToast: document.getElementById('success-toast'),
            
            // Other elements
            typingIndicator: document.getElementById('typing-indicator')
        };
    }
    
    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Input events
        this.elements.messageInput.addEventListener('input', this.handleInputChange.bind(this));
        this.elements.messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        this.elements.sendBtn.addEventListener('click', this.sendMessage.bind(this));
        
        // Header button events
        this.elements.settingsBtn.addEventListener('click', this.openSettings.bind(this));
        this.elements.clearChatBtn.addEventListener('click', this.clearChat.bind(this));
        this.elements.menuBtn.addEventListener('click', this.toggleMenu.bind(this));
        
        // Modal events
        this.elements.settingsClose.addEventListener('click', this.closeSettings.bind(this));
        this.elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.settingsModal) {
                this.closeSettings();
            }
        });
        
        // Settings events
        this.elements.themeSelect.addEventListener('change', this.handleThemeChange.bind(this));
        this.elements.fontSizeSelect.addEventListener('change', this.handleFontSizeChange.bind(this));
        this.elements.soundToggle.addEventListener('change', this.handleSoundToggle.bind(this));
        this.elements.notificationsToggle.addEventListener('change', this.handleNotificationsToggle.bind(this));
        
        // Toast close events
        document.querySelectorAll('.toast-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.toast').classList.remove('show');
            });
        });
        
        // Window events
        window.addEventListener('resize', this.handleResize.bind(this));
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleGlobalKeyDown.bind(this));
        
        // Touch events for mobile
        this.setupTouchEvents();
    }
    
    /**
     * Setup touch events for mobile devices
     */
    setupTouchEvents() {
        // Prevent zoom on double tap for input
        this.elements.messageInput.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.target.focus();
        });
        
        // Handle swipe gestures
        let touchStartY = 0;
        let touchStartX = 0;
        
        this.elements.chatMessages.addEventListener('touchstart', (e) => {
            touchStartY = e.touches[0].clientY;
            touchStartX = e.touches[0].clientX;
        });
        
        this.elements.chatMessages.addEventListener('touchmove', (e) => {
            const touchY = e.touches[0].clientY;
            const touchX = e.touches[0].clientX;
            const deltaY = touchStartY - touchY;
            const deltaX = touchStartX - touchX;
            
            // Prevent horizontal scrolling
            if (Math.abs(deltaX) > Math.abs(deltaY)) {
                e.preventDefault();
            }
        });
    }
    
    /**
     * Initialize WebSocket connection
     */
    initializeSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('🔌 Connected to server');
                this.isConnected = true;
                this.updateConnectionStatus(true);
                this.processMessageQueue();
            });
            
            this.socket.on('disconnect', () => {
                console.log('🔌 Disconnected from server');
                this.isConnected = false;
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('chat_response', (data) => {
                this.handleChatResponse(data);
            });
            
            this.socket.on('typing', (data) => {
                this.handleTypingIndicator(data.typing);
            });
            
            this.socket.on('error', (data) => {
                this.showError(data.message);
            });
            
            this.socket.on('status', (data) => {
                console.log('📡 Server status:', data.message);
            });
            
        } catch (error) {
            console.error('❌ Failed to initialize socket:', error);
            this.isConnected = false;
            this.updateConnectionStatus(false);
        }
    }
    
    /**
     * Handle input change events
     */
    handleInputChange(e) {
        const value = e.target.value;
        const length = value.length;
        
        // Update character count
        this.elements.charCount.textContent = length;
        
        // Auto-resize textarea
        this.autoResizeTextarea(e.target);
        
        // Enable/disable send button
        this.elements.sendBtn.disabled = length === 0 || length > this.config.maxMessageLength;
        
        // Show character limit warning
        if (length > this.config.maxMessageLength * 0.9) {
            this.elements.charCount.style.color = 'var(--error-color)';
        } else {
            this.elements.charCount.style.color = 'var(--text-tertiary)';
        }
    }
    
    /**
     * Auto-resize textarea based on content
     */
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    /**
     * Handle key down events
     */
    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }
    
    /**
     * Handle global keyboard shortcuts
     */
    handleGlobalKeyDown(e) {
        // Escape key to close modals
        if (e.key === 'Escape') {
            this.closeSettings();
        }
        
        // Ctrl/Cmd + K to focus input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            this.elements.messageInput.focus();
        }
        
        // Ctrl/Cmd + L to clear chat
        if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
            e.preventDefault();
            this.clearChat();
        }
    }
    
    /**
     * Send message to chatbot
     */
    async sendMessage() {
        const message = this.elements.messageInput.value.trim();
        
        if (!message || message.length > this.config.maxMessageLength) {
            return;
        }
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.elements.messageInput.value = '';
        this.elements.charCount.textContent = '0';
        this.elements.sendBtn.disabled = true;
        this.autoResizeTextarea(this.elements.messageInput);
        
        // Add to conversation history
        this.conversationHistory.push({
            type: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });
        
        // Send via WebSocket if connected, otherwise queue
        if (this.isConnected && this.socket) {
            this.socket.emit('chat_message', {
                message: message,
                history: this.conversationHistory.slice(-10) // Last 10 messages
            });
        } else {
            // Fallback to REST API
            this.sendMessageViaAPI(message);
        }
        
        // Play send sound
        this.playSound('send');
        
        // Focus input
        this.elements.messageInput.focus();
    }
    
    /**
     * Send message via REST API (fallback)
     */
    async sendMessageViaAPI(message) {
        try {
            this.showTypingIndicator();
            
            const response = await fetch(this.config.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    history: this.conversationHistory.slice(-10)
                })
            });
            
            const data = await response.json();
            this.handleChatResponse(data);
            
        } catch (error) {
            console.error('❌ API request failed:', error);
            this.hideTypingIndicator();
            this.showError('Failed to send message. Please try again.');
        }
    }
    
    /**
     * Handle chat response from server
     */
    handleChatResponse(data) {
        this.hideTypingIndicator();
        
        if (data.success) {
            // Add bot response to chat
            this.addMessage(data.response, 'bot');
            
            // Add to conversation history
            this.conversationHistory.push({
                type: 'bot',
                content: data.response,
                timestamp: data.timestamp || new Date().toISOString()
            });
            
            // Play receive sound
            this.playSound('receive');
            
            // Show notification if page is not visible
            if (document.hidden && this.settings.notifications) {
                this.showNotification('New message from AI Assistant', data.response);
            }
            
        } else {
            this.showError(data.error || 'Failed to get response from AI');
        }
    }
    
    /**
     * Add message to chat interface
     */
    addMessage(content, type, timestamp = null) {
        const messageGroup = document.createElement('div');
        messageGroup.className = `message-group ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        // Format message content
        bubble.innerHTML = this.formatMessage(content);
        
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = this.formatTime(timestamp || new Date());
        
        messageContent.appendChild(bubble);
        messageContent.appendChild(time);
        messageGroup.appendChild(avatar);
        messageGroup.appendChild(messageContent);
        
        // Insert before typing indicator
        const typingIndicator = this.elements.typingIndicator;
        this.elements.chatMessages.insertBefore(messageGroup, typingIndicator);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Animate message
        requestAnimationFrame(() => {
            messageGroup.style.opacity = '0';
            messageGroup.style.transform = 'translateY(20px)';
            requestAnimationFrame(() => {
                messageGroup.style.transition = 'all 0.3s ease-out';
                messageGroup.style.opacity = '1';
                messageGroup.style.transform = 'translateY(0)';
            });
        });
    }
    
    /**
     * Format message content (basic markdown support)
     */
    formatMessage(content) {
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
    }
    
    /**
     * Format timestamp
     */
    formatTime(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) { // Less than 1 minute
            return 'Just now';
        } else if (diff < 3600000) { // Less than 1 hour
            const minutes = Math.floor(diff / 60000);
            return `${minutes}m ago`;
        } else if (diff < 86400000) { // Less than 1 day
            const hours = Math.floor(diff / 3600000);
            return `${hours}h ago`;
        } else {
            return date.toLocaleDateString();
        }
    }
    
    /**
     * Show/hide typing indicator
     */
    showTypingIndicator() {
        this.isTyping = true;
        this.elements.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        this.elements.typingIndicator.style.display = 'none';
    }
    
    handleTypingIndicator(typing) {
        if (typing) {
            this.showTypingIndicator();
        } else {
            this.hideTypingIndicator();
        }
    }
    
    /**
     * Scroll chat to bottom
     */
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        });
    }
    
    /**
     * Clear chat history
     */
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Remove all messages except welcome message
            const messages = this.elements.chatMessages.querySelectorAll('.message-group');
            messages.forEach((message, index) => {
                if (index > 0) { // Keep welcome message
                    message.remove();
                }
            });
            
            // Clear conversation history
            this.conversationHistory = [];
            
            this.showSuccess('Chat history cleared');
        }
    }
    
    /**
     * Settings management
     */
    openSettings() {
        this.elements.settingsModal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }
    
    closeSettings() {
        this.elements.settingsModal.classList.remove('show');
        document.body.style.overflow = '';
    }
    
    handleThemeChange(e) {
        this.settings.theme = e.target.value;
        this.applyTheme();
        this.saveSettings();
    }
    
    handleFontSizeChange(e) {
        this.settings.fontSize = e.target.value;
        this.applyFontSize();
        this.saveSettings();
    }
    
    handleSoundToggle(e) {
        this.settings.sound = e.target.checked;
        this.saveSettings();
    }
    
    handleNotificationsToggle(e) {
        this.settings.notifications = e.target.checked;
        this.saveSettings();
        
        if (e.target.checked) {
            this.requestNotificationPermission();
        }
    }
    
    /**
     * Apply saved settings
     */
    applySettings() {
        // Apply theme
        this.applyTheme();
        
        // Apply font size
        this.applyFontSize();
        
        // Update UI controls
        this.elements.themeSelect.value = this.settings.theme;
        this.elements.fontSizeSelect.value = this.settings.fontSize;
        this.elements.soundToggle.checked = this.settings.sound;
        this.elements.notificationsToggle.checked = this.settings.notifications;
    }
    
    applyTheme() {
        const theme = this.settings.theme;
        
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        } else {
            document.documentElement.setAttribute('data-theme', theme);
        }
    }
    
    applyFontSize() {
        const fontSize = this.settings.fontSize;
        const sizes = {
            small: '14px',
            medium: '16px',
            large: '18px'
        };
        
        document.documentElement.style.fontSize = sizes[fontSize] || sizes.medium;
    }
    
    /**
     * Load settings from localStorage
     */
    loadSettings() {
        const defaultSettings = {
            theme: 'auto',
            fontSize: 'medium',
            sound: true,
            notifications: true
        };
        
        try {
            const saved = localStorage.getItem('chatbot-settings');
            return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
        } catch (error) {
            console.error('Failed to load settings:', error);
            return defaultSettings;
        }
    }
    
    /**
     * Save settings to localStorage
     */
    saveSettings() {
        try {
            localStorage.setItem('chatbot-settings', JSON.stringify(this.settings));
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }
    
    /**
     * Connection status management
     */
    updateConnectionStatus(connected) {
        const statusElement = this.elements.connectionStatus;
        const statusText = statusElement.querySelector('span');
        const statusIcon = statusElement.querySelector('i');
        
        if (connected) {
            statusText.textContent = 'Connected';
            statusIcon.className = 'fas fa-wifi';
            statusElement.style.color = 'var(--success-color)';
        } else {
            statusText.textContent = 'Disconnected';
            statusIcon.className = 'fas fa-wifi-slash';
            statusElement.style.color = 'var(--error-color)';
        }
    }
    
    /**
     * Health check
     */
    async checkHealth() {
        try {
            const response = await fetch(this.config.healthEndpoint);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                console.log('✅ Health check passed');
                this.updateBotStatus(data.ollama_connected);
            } else {
                console.warn('⚠️ Health check failed');
                this.updateBotStatus(false);
            }
        } catch (error) {
            console.error('❌ Health check error:', error);
            this.updateBotStatus(false);
        }
    }
    
    updateBotStatus(online) {
        const statusElement = this.elements.botStatus;
        const indicator = statusElement.querySelector('.status-indicator');
        const text = statusElement.querySelector('.status-text');
        
        if (online) {
            text.textContent = 'Online';
            indicator.style.backgroundColor = 'var(--success-color)';
        } else {
            text.textContent = 'Offline';
            indicator.style.backgroundColor = 'var(--error-color)';
        }
    }
    
    /**
     * Notification management
     */
    async requestNotificationPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            const permission = await Notification.requestPermission();
            if (permission !== 'granted') {
                this.settings.notifications = false;
                this.elements.notificationsToggle.checked = false;
                this.saveSettings();
            }
        }
    }
    
    showNotification(title, body) {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(title, {
                body: body.substring(0, 100) + (body.length > 100 ? '...' : ''),
                icon: '/static/images/icon-192.png',
                badge: '/static/images/icon-192.png',
                tag: 'chatbot-message'
            });
        }
    }
    
    /**
     * Sound effects
     */
    playSound(type) {
        if (!this.settings.sound) return;
        
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
    
    /**
     * Toast notifications
     */
    showError(message) {
        this.showToast(message, 'error');
    }
    
    showSuccess(message) {
        this.showToast(message, 'success');
    }
    
    showToast(message, type) {
        const toast = type === 'error' ? this.elements.errorToast : this.elements.successToast;
        const messageElement = toast.querySelector('.toast-message');
        
        messageElement.textContent = message;
        toast.classList.add('show');
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
        }, 5000);
    }
    
    /**
     * Utility methods
     */
    hideLoadingScreen() {
        setTimeout(() => {
            this.elements.loadingScreen.style.display = 'none';
            this.elements.app.style.display = 'flex';
        }, 500);
    }
    
    handleResize() {
        // Adjust chat height on mobile keyboard
        if (window.innerHeight < 500) {
            document.body.classList.add('keyboard-open');
        } else {
            document.body.classList.remove('keyboard-open');
        }
    }
    
    handleBeforeUnload(e) {
        if (this.conversationHistory.length > 1) {
            e.preventDefault();
            e.returnValue = '';
        }
    }
    
    toggleMenu() {
        // Mobile menu functionality can be added here
        console.log('Menu toggled');
    }
    
    processMessageQueue() {
        // Process any queued messages when connection is restored
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.sendMessageViaAPI(message);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatbotApp = new ChatbotApp();
});

// Handle theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (window.chatbotApp && window.chatbotApp.settings.theme === 'auto') {
        window.chatbotApp.applyTheme();
    }
});