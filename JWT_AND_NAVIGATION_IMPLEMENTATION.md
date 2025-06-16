# JWT-based Expiration & Refresh Support + Navigation Bar Implementation

This document describes the comprehensive JWT token management system and navigation bar implementation for seamless switching between Chat and History views.

## Table of Contents

- [JWT Token Management](#jwt-token-management)
- [Navigation Bar Implementation](#navigation-bar-implementation)
- [Token Status Monitoring](#token-status-monitoring)
- [Session Expiration Handling](#session-expiration-handling)
- [Frontend Architecture](#frontend-architecture)
- [Backend Enhancements](#backend-enhancements)
- [Security Features](#security-features)
- [Usage Examples](#usage-examples)

## JWT Token Management

### Enhanced Token Manager Utility

#### **TokenManager Class Features**
```javascript
class TokenManager {
  // Token storage and retrieval
  setToken(token, expiresIn)     // Store token with expiration
  getToken()                     // Retrieve current token
  clearToken()                   // Clear token and expiration
  
  // Token validation
  isTokenExpired()               // Check if token is expired
  isValidToken()                 // Validate token format and expiration
  needsRefresh()                 // Check if token needs refresh
  
  // Time calculations
  getTimeUntilExpiry()           // Time until token expires (ms)
  getTimeUntilRefresh()          // Time until refresh needed (ms)
  formatTimeRemaining()          // Human-readable time format
  
  // JWT payload handling
  decodeToken()                  // Decode JWT payload
  getUserFromToken()             // Extract user info from token
  
  // Status and monitoring
  getTokenStatus()               // Comprehensive token status
  scheduleRefresh(callback)      // Schedule automatic refresh
}
```

#### **Token Lifecycle Management**
- **Automatic Storage**: Secure localStorage management with expiration tracking
- **Validation**: Format checking and expiration verification
- **Refresh Scheduling**: Automatic refresh 5 minutes before expiry
- **Status Monitoring**: Real-time token status with detailed information

### Enhanced AuthContext Integration

#### **Improved Token Refresh Logic**
```javascript
const refreshToken = useCallback(async (force = false) => {
  try {
    // Prevent multiple simultaneous refresh attempts
    if (state.refreshTimer && !force) {
      return state.token;
    }

    // Validate current token
    if (tokenManager.isTokenExpired() && !force) {
      throw new Error('Token is expired and cannot be refreshed');
    }

    // Refresh token via API
    const response = await authService.refreshToken(currentToken);
    const newToken = response.token;
    
    // Store new token using token manager
    const sessionExpiry = tokenManager.setToken(newToken);
    
    // Update context state
    dispatch({
      type: 'TOKEN_REFRESHED',
      payload: { token: newToken, sessionExpiry }
    });

    // Schedule next refresh
    scheduleTokenRefresh(sessionExpiry);
    
    return newToken;
  } catch (error) {
    tokenManager.clearToken();
    dispatch({ type: 'LOGOUT' });
    throw error;
  }
}, [state.refreshTimer]);
```

#### **Smart Refresh Scheduling**
- **Automatic Timing**: Calculates optimal refresh time based on token expiration
- **Error Handling**: Graceful fallback to logout on refresh failure
- **Multiple Prevention**: Prevents simultaneous refresh attempts
- **Activity Tracking**: Updates session activity on successful refresh

## Navigation Bar Implementation

### AppNavBar Component Features

#### **Responsive Navigation Design**
```jsx
function AppNavBar({ onShowToast }) {
  return (
    <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-40">
      {/* Desktop Navigation */}
      <div className="hidden sm:ml-8 sm:flex sm:space-x-1">
        <Link to="/chat" className={`${isActive('/chat') ? 'active-styles' : 'inactive-styles'}`}>
          <i className="fas fa-comments mr-2"></i>
          Chat
          {isActive('/chat') && <div className="animate-pulse indicator"></div>}
        </Link>
        
        <Link to="/history" className={`${isActive('/history') ? 'active-styles' : 'inactive-styles'}`}>
          <i className="fas fa-history mr-2"></i>
          History
          {isActive('/history') && <div className="animate-pulse indicator"></div>}
        </Link>
      </div>
      
      {/* Mobile Navigation */}
      {isMobileMenuOpen && (
        <div className="sm:hidden">
          {/* Mobile menu items */}
        </div>
      )}
    </nav>
  );
}
```

#### **Navigation Features**
- **Active Route Highlighting**: Visual indication of current page
- **Animated Indicators**: Pulsing dots for active routes
- **Mobile-Responsive**: Collapsible hamburger menu for mobile
- **User Information**: Display user details and session status
- **Quick Actions**: Logout and profile access
- **Token Status**: Real-time session status indicator

### Navigation State Management

#### **Route-Aware Components**
```javascript
const isActive = (path) => {
  return location.pathname === path;
};

// Dynamic styling based on active route
const getLinkClassName = (path) => {
  return `inline-flex items-center px-4 py-2 border-b-2 text-sm font-medium transition-colors ${
    isActive(path)
      ? 'border-primary-500 text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/20'
      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
  }`;
};
```

#### **Seamless Navigation Flow**
- **URL Preservation**: Maintains session parameters across navigation
- **State Persistence**: Preserves chat state when switching views
- **Loading States**: Smooth transitions between routes
- **Error Handling**: Graceful fallback for navigation errors

## Token Status Monitoring

### TokenStatusIndicator Component

#### **Real-time Status Display**
```jsx
function TokenStatusIndicator({ showDetails = false }) {
  const [tokenStatus, setTokenStatus] = useState(null);
  
  useEffect(() => {
    const updateStatus = () => {
      const status = tokenManager.getTokenStatus();
      setTokenStatus(status);
    };

    updateStatus();
    const interval = setInterval(updateStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = () => {
    switch (tokenStatus.status) {
      case 'valid': return <i className="fas fa-check-circle text-green-500"></i>;
      case 'refresh_needed': return <i className="fas fa-exclamation-triangle text-yellow-500"></i>;
      case 'expired': return <i className="fas fa-times-circle text-red-500"></i>;
      default: return <i className="fas fa-question-circle text-gray-500"></i>;
    }
  };

  return (
    <div className="flex items-center space-x-2">
      {getStatusIcon()}
      <span className={getStatusColor()}>{getStatusText()}</span>
      {shouldShowRefreshButton() && (
        <button onClick={handleRefresh}>Extend</button>
      )}
    </div>
  );
}
```

#### **Status Types and Indicators**
- **Valid**: Green checkmark - session is active and healthy
- **Refresh Needed**: Yellow warning - token expires soon
- **Expired**: Red X - token has expired
- **Invalid**: Red ban - token is malformed
- **Missing**: Gray question - no token found

### Session Expiration Warning

#### **Enhanced Warning System**
```javascript
function SessionExpirationWarning() {
  const { refreshToken, logout } = useAuth();
  const [showWarning, setShowWarning] = useState(false);
  const [timeLeft, setTimeLeft] = useState(0);

  useEffect(() => {
    const checkSessionExpiry = () => {
      const status = tokenManager.getTokenStatus();
      
      if (status.status === 'refresh_needed') {
        const timeUntilExpiry = tokenManager.getTimeUntilExpiry();
        const warningTime = 10 * 60 * 1000; // 10 minutes
        
        if (timeUntilExpiry <= warningTime && timeUntilExpiry > 0) {
          setShowWarning(true);
          setTimeLeft(Math.floor(timeUntilExpiry / 1000));
        }
      }
    };

    const interval = setInterval(checkSessionExpiry, 30000);
    return () => clearInterval(interval);
  }, []);

  return showWarning ? (
    <div className="session-warning-modal">
      <h3>Session Expiring Soon</h3>
      <p>Your session will expire in {formatTime(timeLeft)}</p>
      <button onClick={handleRefreshSession}>Extend Session</button>
      <button onClick={logout}>Logout Now</button>
    </div>
  ) : null;
}
```

#### **Warning Features**
- **10-Minute Alert**: Shows warning 10 minutes before expiry
- **Real-time Countdown**: Updates every second with time remaining
- **Action Options**: Extend session or logout immediately
- **Auto-dismiss**: Hides after successful refresh
- **Status Integration**: Uses TokenManager for accurate timing

## Frontend Architecture

### Enhanced Page Structure

#### **ChatPage with Navigation**
```jsx
function ChatPage() {
  return (
    <div className="flex flex-col h-screen">
      {/* Navigation Bar */}
      <AppNavBar onShowToast={showToast} />
      
      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Chat History */}
        <div className="w-80 bg-white dark:bg-gray-800">
          <ChatHistory />
        </div>
        
        {/* Chat Container */}
        <div className="flex-1">
          <ChatContainer />
        </div>
      </div>
      
      {/* Session Warning */}
      <SessionExpirationWarning />
    </div>
  );
}
```

#### **HistoryPage with Navigation**
```jsx
function HistoryPage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Navigation Bar */}
      <AppNavBar onShowToast={showToast} />
      
      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
          <div className="flex flex-col h-full">
            {/* History Header */}
            <div className="flex items-center justify-between p-6">
              <h1>Chat History</h1>
              <button onClick={() => navigate('/chat')}>
                Back to Chat
              </button>
            </div>
            
            {/* Chat History Component */}
            <div className="flex-1 overflow-hidden">
              <ChatHistory />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
```

### Component Integration

#### **Unified Token Management**
- **TokenManager**: Centralized token lifecycle management
- **AuthContext**: Enhanced with token manager integration
- **TokenStatusIndicator**: Real-time status display
- **SessionExpirationWarning**: Proactive session management

#### **Navigation Integration**
- **AppNavBar**: Unified navigation across all pages
- **Route Awareness**: Active state management
- **Mobile Optimization**: Responsive design patterns
- **User Experience**: Seamless transitions

## Backend Enhancements

### JWT Configuration

#### **Configurable Token Expiration**
```python
# Enhanced JWT configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-string-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=int(os.environ.get('JWT_EXPIRES_HOURS', 24)))
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=int(os.environ.get('JWT_REFRESH_EXPIRES_DAYS', 30)))
```

#### **Environment Variables**
```env
# JWT Configuration
JWT_SECRET_KEY=your-secure-jwt-secret-key
JWT_EXPIRES_HOURS=24
JWT_REFRESH_EXPIRES_DAYS=30
```

### Enhanced Token Refresh Endpoint

#### **Robust Refresh Logic**
```python
@app.route('/api/auth/refresh', methods=['POST'])
@token_required
def refresh_token(current_user_id):
    try:
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        # Generate new JWT token
        new_token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        # Update user session with new token
        current_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        current_token_hash = hashlib.sha256(current_token.encode()).hexdigest()
        
        user_session = UserSession.query.filter_by(
            user_id=current_user_id,
            token_hash=current_token_hash,
            is_active=True
        ).first()
        
        if user_session:
            # Update existing session
            new_token_hash = hashlib.sha256(new_token.encode()).hexdigest()
            user_session.token_hash = new_token_hash
            user_session.expires_at = datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
            user_session.last_activity = datetime.utcnow()
            db.session.commit()
        
        return jsonify({
            'message': 'Token refreshed successfully',
            'token': new_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Token refresh failed: {str(e)}'}), 500
```

## Security Features

### Token Security

#### **Enhanced Security Measures**
- **Secure Storage**: Token hash storage instead of plain tokens
- **Expiration Enforcement**: Strict expiration checking
- **Automatic Cleanup**: Expired session removal
- **Activity Tracking**: Session activity monitoring

#### **Session Management**
- **Multiple Session Support**: Track multiple active sessions
- **Session Isolation**: User-specific session management
- **Logout All**: Ability to terminate all sessions
- **Current Session Identification**: Mark current session

### Frontend Security

#### **Token Handling**
- **Secure Storage**: localStorage with expiration tracking
- **Automatic Cleanup**: Clear tokens on logout/expiry
- **Validation**: Client-side token format validation
- **Error Handling**: Graceful handling of token errors

#### **Route Protection**
- **Authentication Guards**: Protect sensitive routes
- **Automatic Redirects**: Redirect to login when needed
- **State Preservation**: Maintain intended destination

## Usage Examples

### Setting Up JWT Token Management

```javascript
// Initialize token manager
import { tokenManager } from '../utils/tokenManager';

// Store token after login
const sessionExpiry = tokenManager.setToken(response.token);

// Check token status
const status = tokenManager.getTokenStatus();
console.log('Token status:', status.message);

// Schedule automatic refresh
const timer = tokenManager.scheduleRefresh(() => {
  refreshToken();
});
```

### Using Navigation Bar

```jsx
import AppNavBar from '../components/AppNavBar';

function MyPage() {
  const showToast = (message, type) => {
    // Toast implementation
  };

  return (
    <div className="flex flex-col h-screen">
      <AppNavBar onShowToast={showToast} />
      {/* Page content */}
    </div>
  );
}
```

### Implementing Token Status Monitoring

```jsx
import TokenStatusIndicator from '../components/TokenStatusIndicator';

function Header() {
  return (
    <header className="flex items-center justify-between">
      <h1>My App</h1>
      <div className="flex items-center space-x-4">
        <TokenStatusIndicator showDetails={true} />
        <UserMenu />
      </div>
    </header>
  );
}
```

### Session Expiration Handling

```jsx
import SessionExpirationWarning from '../components/SessionExpirationWarning';

function App() {
  return (
    <div className="app">
      {/* App content */}
      <SessionExpirationWarning />
    </div>
  );
}
```

## Configuration

### Environment Setup

```env
# JWT Configuration
JWT_SECRET_KEY=your-secure-jwt-secret-key-change-in-production
JWT_EXPIRES_HOURS=24
JWT_REFRESH_EXPIRES_DAYS=30

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chatbot_db
DB_USER=chatbot_user
DB_PASSWORD=chatbot_password
```

### Frontend Configuration

```javascript
// Token manager configuration
const tokenManager = new TokenManager();
tokenManager.REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes

// Navigation configuration
const routes = [
  { path: '/chat', component: ChatPage, protected: true },
  { path: '/history', component: HistoryPage, protected: true },
  { path: '/login', component: LoginPage, public: true },
];
```

## Best Practices

### Token Management

1. **Secure Storage**
   - Use token manager for all token operations
   - Never store plain tokens in localStorage
   - Clear tokens on logout/expiry

2. **Refresh Strategy**
   - Schedule automatic refresh before expiry
   - Handle refresh failures gracefully
   - Prevent multiple simultaneous refreshes

3. **Status Monitoring**
   - Display real-time token status
   - Provide user feedback on session state
   - Offer manual refresh options

### Navigation Design

1. **User Experience**
   - Provide clear visual indicators for active routes
   - Ensure smooth transitions between pages
   - Maintain responsive design across devices

2. **State Management**
   - Preserve application state during navigation
   - Handle route parameters appropriately
   - Implement proper error boundaries

3. **Performance**
   - Optimize navigation components for re-renders
   - Use proper memoization techniques
   - Implement lazy loading where appropriate

## Troubleshooting

### Common Issues

#### **Token Refresh Problems**
- Check network connectivity
- Verify JWT secret key configuration
- Ensure token format is correct
- Check server-side refresh endpoint

#### **Navigation Issues**
- Verify route configuration
- Check authentication guards
- Ensure proper imports
- Test responsive behavior

#### **Session Expiration**
- Check token expiration times
- Verify refresh scheduling
- Test warning display
- Ensure proper cleanup

### Debug Information

#### **Token Status Debugging**
```javascript
// Check token status
const status = tokenManager.getTokenStatus();
console.log('Token Status:', status);

// Check time remaining
const timeLeft = tokenManager.getTimeUntilExpiry();
console.log('Time until expiry:', timeLeft);

// Check refresh timing
const refreshTime = tokenManager.getTimeUntilRefresh();
console.log('Time until refresh:', refreshTime);
```

#### **Navigation Debugging**
```javascript
// Check current route
console.log('Current route:', location.pathname);

// Check authentication state
console.log('Is authenticated:', isAuthenticated);

// Check route guards
console.log('Route protection:', isProtectedRoute);
```

For additional support, refer to the main documentation or contact the development team.