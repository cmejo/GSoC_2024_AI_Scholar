# AI Chatbot Web GUI

A responsive, mobile-ready web frontend for AI chatbot services built with Flask, JavaScript, and modern web technologies.

## Features

### 🎨 **Modern UI/UX**
- Clean, intuitive interface with smooth animations
- Responsive design that works on desktop, tablet, and mobile
- Dark/light theme support with auto-detection
- Customizable font sizes and accessibility options

### 📱 **Mobile-First Design**
- Touch-optimized interactions
- Mobile keyboard handling
- Swipe gestures support
- Progressive Web App (PWA) capabilities
- Offline functionality with service worker

### 🚀 **Real-time Communication**
- WebSocket support for instant messaging
- Typing indicators
- Connection status monitoring
- Auto-reconnection on network issues

### 🤖 **AI Integration**
- Compatible with Ollama and other AI services
- Conversation history management
- Message formatting with basic markdown support
- Error handling and fallback mechanisms

### 🔧 **Advanced Features**
- Settings panel with theme and preferences
- Sound effects and notifications
- Chat history management
- Keyboard shortcuts
- Health monitoring
- Toast notifications

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama (or compatible AI service)
- Modern web browser

### Installation

1. **Clone or download the project files**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   # Create .env file
   echo "OLLAMA_BASE_URL=http://localhost:11434" > .env
   echo "DEFAULT_MODEL=llama2" >> .env
   echo "SECRET_KEY=your-secret-key-here" >> .env
   ```

4. **Start Ollama** (if using Ollama)
   ```bash
   ollama serve
   ```

5. **Pull an AI model** (if using Ollama)
   ```bash
   ollama pull llama2
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open your browser**
   Navigate to `http://localhost:5000`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `DEFAULT_MODEL` | `llama2` | Default AI model to use |
| `SECRET_KEY` | `your-secret-key-here` | Flask session secret key |

### Customization

#### Themes
The application supports three themes:
- **Light**: Clean, bright interface
- **Dark**: Easy on the eyes for low-light environments  
- **Auto**: Automatically switches based on system preference

#### Font Sizes
- **Small**: 14px base font size
- **Medium**: 16px base font size (default)
- **Large**: 18px base font size

## API Endpoints

### REST API

#### `POST /api/chat`
Send a message to the chatbot
```json
{
  "message": "Hello, how are you?",
  "history": [...]
}
```

#### `GET /api/health`
Check service health status
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "model": "llama2",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### WebSocket Events

#### Client → Server
- `chat_message`: Send a chat message
- `connect`: Client connection established
- `disconnect`: Client disconnected

#### Server → Client
- `chat_response`: AI response to user message
- `typing`: Typing indicator status
- `status`: Connection status updates
- `error`: Error messages

## Mobile Features

### Progressive Web App (PWA)
- Install as native app on mobile devices
- Offline functionality with service worker
- App-like experience with splash screen
- Background sync for offline messages

### Touch Interactions
- Optimized touch targets (minimum 44px)
- Swipe gestures for navigation
- Pull-to-refresh support
- Haptic feedback on supported devices

### Keyboard Handling
- Auto-resize text input
- Smart keyboard avoidance
- Enter to send, Shift+Enter for new line
- Character count and limits

## Browser Support

### Desktop
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Mobile
- iOS Safari 14+
- Chrome Mobile 90+
- Samsung Internet 14+
- Firefox Mobile 88+

## Development

### Project Structure
```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Responsive styles
│   ├── js/
│   │   └── app.js        # Main JavaScript application
│   ├── manifest.json     # PWA manifest
│   └── sw.js            # Service worker
└── README.md             # This file
```

### Key Components

#### Frontend (JavaScript)
- `ChatbotApp`: Main application class
- WebSocket management with auto-reconnection
- Message handling and UI updates
- Settings management with localStorage
- PWA features and offline support

#### Backend (Python/Flask)
- `ChatbotService`: AI integration layer
- WebSocket server with Flask-SocketIO
- REST API endpoints
- Health monitoring
- Session management

### Customization

#### Adding New Themes
1. Add CSS variables in `:root` and `[data-theme="new-theme"]`
2. Update theme selector in settings
3. Add theme logic in JavaScript

#### Integrating Different AI Services
1. Modify `ChatbotService` class in `app.py`
2. Update API endpoints and request format
3. Adjust response parsing logic

#### Adding New Features
1. Update HTML template with new UI elements
2. Add CSS styles for responsive design
3. Implement JavaScript functionality
4. Add backend API endpoints if needed

## Troubleshooting

### Common Issues

#### Connection Problems
- Check if Ollama is running: `ollama serve`
- Verify the correct port (default: 11434)
- Check firewall settings
- Ensure model is downloaded: `ollama pull llama2`

#### Mobile Issues
- Clear browser cache and reload
- Check if JavaScript is enabled
- Verify WebSocket support
- Try different network connection

#### Performance Issues
- Reduce conversation history length
- Disable animations in settings
- Use smaller AI model
- Check browser developer tools for errors

### Debug Mode
Run with debug enabled:
```bash
FLASK_DEBUG=1 python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on multiple devices/browsers
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with Flask and modern web technologies
- Icons from Font Awesome
- Fonts from Google Fonts
- AI integration with Ollama
- Responsive design principles from modern web standards