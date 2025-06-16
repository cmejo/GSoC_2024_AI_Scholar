# AI Chatbot Frontend

A modern React.js frontend for the AI Chatbot application, built with Tailwind CSS and axios.

## Features

- 🎨 Modern, responsive UI with Tailwind CSS
- 🌙 Dark/Light theme support
- 📱 Mobile-friendly design
- 🔄 Real-time messaging with WebSocket fallback
- 🔊 Sound effects and notifications
- ⚙️ Customizable settings
- 🚀 PWA support
- 🐳 Docker containerized

## Tech Stack

- **React 18** - UI framework
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client
- **Socket.IO Client** - Real-time communication
- **React Hooks** - State management
- **Context API** - Global state

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Development

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm start
```

3. Open [http://localhost:3000](http://localhost:3000)

### Production Build

```bash
npm run build
```

### Docker

Build and run with Docker:

```bash
# Development
docker-compose -f docker-compose.dev.yml up --build

# Production
docker-compose up --build
```

## Environment Variables

- `REACT_APP_API_URL` - Backend API URL (default: http://localhost:5000)

## Project Structure

```
src/
├── components/          # React components
│   ├── ChatContainer.js
│   ├── ChatHeader.js
│   ├── ChatMessages.js
│   ├── ChatInput.js
│   ├── Message.js
│   ├── TypingIndicator.js
│   ├── SettingsModal.js
│   ├── LoadingScreen.js
│   └── Toast.js
├── context/            # React context providers
│   ├── ChatContext.js
│   └── SettingsContext.js
├── hooks/              # Custom React hooks
│   ├── useSettings.js
│   └── useNotifications.js
├── services/           # API services
│   └── chatService.js
├── utils/              # Utility functions
│   └── messageUtils.js
├── App.js              # Main app component
├── index.js            # Entry point
└── index.css           # Global styles
```

## Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

## Features

### Responsive Design
- Mobile-first approach
- Touch-friendly interface
- Adaptive layouts

### Theming
- Light/Dark/Auto themes
- System preference detection
- Persistent settings

### Real-time Communication
- WebSocket for real-time messaging
- Automatic fallback to HTTP
- Connection status indicators

### Accessibility
- Keyboard navigation
- Screen reader support
- High contrast support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.