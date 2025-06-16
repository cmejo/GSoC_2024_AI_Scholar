# Routing and Session Management Implementation

This document describes the comprehensive routing system, authentication guards, session management, and token refresh functionality implemented for the AI Chatbot application.

## Table of Contents

- [Routing System](#routing-system)
- [Authentication Guards](#authentication-guards)
- [Session Management](#session-management)
- [Token Refresh](#token-refresh)
- [User Interface Enhancements](#user-interface-enhancements)
- [Backend Improvements](#backend-improvements)
- [Security Features](#security-features)
- [Usage Examples](#usage-examples)

## Routing System

### React Router Implementation

The application now uses React Router v6 for client-side routing with the following structure:

```
/login          - Public route for user authentication
/register       - Public route for user registration
/chat           - Protected route for main chat interface
/history        - Protected route for chat history management
/               - Redirects to /chat
/*              - Catch-all redirects to /chat
```

### Route Components

#### **Public Routes**
- **LoginPage**: User authentication with return URL support
- **RegisterPage**: Multi-step user registration

#### **Protected Routes**
- **ChatPage**: Main chat interface with session management
- **HistoryPage**: Dedicated chat history management

#### **Route Guards**
- **ProtectedRoute**: Ensures user authentication before access
- **PublicRoute**: Redirects authenticated users away from auth pages

### Navigation Features

#### **Persistent Navigation Bar**
- Logo and branding
- Active route highlighting
- User information display
- Quick action buttons
- Mobile-responsive design

#### **Mobile Navigation**
- Collapsible hamburger menu
- Touch-friendly interface
- User profile in mobile menu
- Quick logout access

## Authentication Guards

### ProtectedRoute Component

```jsx
function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) return <LoadingScreen />;
  
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}
```

**Features:**
- Automatic redirect to login for unauthenticated users
- Preserves intended destination URL
- Loading state handling
- Seamless user experience

### PublicRoute Component

```jsx
function PublicRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) return <LoadingScreen />;
  
  if (isAuthenticated) {
    return <Navigate to="/chat" replace />;
  }

  return children;
}
```

**Features:**
- Prevents authenticated users from accessing auth pages
- Automatic redirect to main application
- Consistent user flow

## Session Management

### Enhanced AuthContext

#### **Session State Management**
```javascript
const initialState = {
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
  sessionExpiry: null,
  refreshTimer: null
};
```

#### **Session Tracking**
- **Session Expiry**: Tracks when the current session expires
- **Refresh Timer**: Automatically schedules token refresh
- **Activity Monitoring**: Updates session activity timestamps

### Session Persistence

#### **Local Storage Management**
- `chatbot_token`: JWT authentication token
- `chatbot_session_expiry`: Session expiration timestamp

#### **Session Restoration**
- Automatic session restoration on app reload
- Token validation and refresh on startup
- Graceful handling of expired sessions

## Token Refresh

### Automatic Token Refresh

#### **Refresh Strategy**
- Tokens refresh 5 minutes before expiry
- Background refresh without user interruption
- Fallback to manual refresh if automatic fails

#### **Refresh Implementation**
```javascript
const refreshToken = useCallback(async () => {
  try {
    const response = await authService.refreshToken(currentToken);
    const newToken = response.token;
    const sessionExpiry = new Date(Date.now() + (24 * 60 * 60 * 1000));

    localStorage.setItem('chatbot_token', newToken);
    
    dispatch({
      type: 'TOKEN_REFRESHED',
      payload: { token: newToken, sessionExpiry }
    });

    scheduleTokenRefresh(sessionExpiry);
    return newToken;
  } catch (error) {
    logout();
    throw error;
  }
}, []);
```

### Session Expiration Warning

#### **SessionExpirationWarning Component**
- Shows warning 10 minutes before expiry
- Real-time countdown display
- Options to extend session or logout
- Automatic logout on expiry

#### **Warning Features**
- **Visual Countdown**: Shows time remaining in MM:SS format
- **Action Buttons**: Extend session or logout immediately
- **Auto-dismiss**: Hides after successful refresh
- **Responsive Design**: Works on all screen sizes

## User Interface Enhancements

### User Information Display

#### **Header User Info**
```jsx
<div className="hidden md:flex items-center space-x-3">
  <div className="text-right">
    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
      Welcome back, {user?.firstName || user?.username || 'User'}!
    </p>
    <p className="text-xs text-gray-500 dark:text-gray-400">
      {user?.email}
    </p>
  </div>
</div>
```

#### **User Menu Enhancements**
- Profile picture/avatar display
- Quick navigation links
- User statistics
- Session management options

### Logout Functionality

#### **Multiple Logout Options**
1. **Header Logout Button**: Prominent logout in main navigation
2. **User Menu Logout**: Dropdown menu option
3. **Mobile Menu Logout**: Touch-friendly mobile option
4. **Session Warning Logout**: Emergency logout from expiration warning

#### **Logout Implementation**
```javascript
const handleLogout = async () => {
  try {
    await logout();
    navigate('/login');
    onShowToast('Logged out successfully', 'success');
  } catch (error) {
    onShowToast('Logout failed', 'error');
  }
};
```

## Backend Improvements

### Enhanced Registration Endpoint

#### **Extended User Data**
```python
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    first_name = data.get('firstName', '').strip()
    last_name = data.get('lastName', '').strip()
    
    # Enhanced validation
    if len(password) < 8:
        return jsonify({'message': 'Password must be at least 8 characters long'}), 400
    
    # Create user with additional fields
    user = User(username=username, email=email, password=password)
    if first_name:
        user.first_name = first_name
    if last_name:
        user.last_name = last_name
```

#### **Improved Validation**
- Minimum 8-character passwords
- Email format validation
- Username uniqueness checking
- Detailed error messages

### Token Refresh Endpoint

#### **Refresh Implementation**
```python
@app.route('/api/auth/refresh', methods=['POST'])
@token_required
def refresh_token(current_user_id):
    # Generate new JWT token
    new_token = jwt.encode({
        'user_id': user.id,
        'username': user.username,
        'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
    }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
    
    # Update user session
    user_session.token_hash = new_token_hash
    user_session.expires_at = datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
    user_session.last_activity = datetime.utcnow()
```

#### **Session Tracking**
- Token hash storage for security
- Activity timestamp updates
- Session expiration management
- IP address and user agent tracking

## Security Features

### Token Security

#### **JWT Implementation**
- Secure token generation with expiration
- Token hash storage (not plain tokens)
- Automatic token rotation
- Secure logout with token invalidation

#### **Session Security**
- Session expiration enforcement
- Automatic cleanup of expired sessions
- IP address and user agent tracking
- Protection against session hijacking

### Authentication Flow

#### **Secure Registration**
1. Client-side validation
2. Server-side validation
3. Password hashing
4. Token generation
5. Session creation
6. Automatic login

#### **Secure Login**
1. Credential validation
2. Password verification
3. Token generation
4. Session tracking
5. Activity logging

### Data Protection

#### **Password Security**
- Minimum 8-character requirement
- Secure hashing with Werkzeug
- No plain text storage
- Password strength validation

#### **Session Management**
- Secure token storage
- Automatic session cleanup
- Activity monitoring
- Expiration enforcement

## Usage Examples

### Setting Up Routing

```jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ProtectedRoute from './components/ProtectedRoute';
import PublicRoute from './components/PublicRoute';

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={
            <PublicRoute><LoginPage /></PublicRoute>
          } />
          <Route path="/chat" element={
            <ProtectedRoute><ChatPage /></ProtectedRoute>
          } />
        </Routes>
      </Router>
    </AuthProvider>
  );
}
```

### Using Navigation

```jsx
import { useNavigate } from 'react-router-dom';

function MyComponent() {
  const navigate = useNavigate();
  
  const handleGoToHistory = () => {
    navigate('/history');
  };
  
  const handleGoToChat = (sessionId) => {
    navigate(`/chat?session=${sessionId}`);
  };
}
```

### Implementing Session Warning

```jsx
import SessionExpirationWarning from './components/SessionExpirationWarning';

function MyPage() {
  return (
    <div>
      {/* Page content */}
      <SessionExpirationWarning />
    </div>
  );
}
```

### Handling Authentication

```jsx
import { useAuth } from './context/AuthContext';

function MyComponent() {
  const { user, logout, isAuthenticated, sessionExpiry } = useAuth();
  
  const handleLogout = async () => {
    await logout();
    // User will be redirected automatically
  };
}
```

## Configuration

### Environment Variables

```env
# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRES=24  # Hours

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chatbot_db
DB_USER=chatbot_user
DB_PASSWORD=chatbot_password
```

### Frontend Configuration

```javascript
// Router configuration
const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to="/chat" replace />
  },
  {
    path: "/login",
    element: <PublicRoute><LoginPage /></PublicRoute>
  },
  {
    path: "/chat",
    element: <ProtectedRoute><ChatPage /></ProtectedRoute>
  }
]);
```

## Best Practices

### Security Best Practices

1. **Token Management**
   - Use secure token storage
   - Implement automatic refresh
   - Handle token expiration gracefully
   - Clear tokens on logout

2. **Session Security**
   - Track session activity
   - Implement session timeouts
   - Monitor for suspicious activity
   - Provide session management tools

3. **Authentication Flow**
   - Validate on both client and server
   - Use secure password requirements
   - Implement proper error handling
   - Provide clear user feedback

### User Experience Best Practices

1. **Navigation**
   - Provide clear navigation paths
   - Highlight active routes
   - Support browser back/forward
   - Handle deep linking

2. **Session Management**
   - Warn before session expiry
   - Provide easy session extension
   - Handle network interruptions
   - Maintain user context

3. **Error Handling**
   - Provide meaningful error messages
   - Handle network failures gracefully
   - Offer recovery options
   - Log errors for debugging

## Troubleshooting

### Common Issues

#### **Routing Problems**
- Check route configuration
- Verify authentication state
- Ensure proper imports
- Test route guards

#### **Session Issues**
- Verify token storage
- Check expiration times
- Test refresh mechanism
- Monitor network requests

#### **Authentication Errors**
- Validate credentials
- Check token format
- Verify server endpoints
- Test error handling

### Debug Information

#### **Client-Side Debugging**
```javascript
// Check authentication state
console.log('Auth state:', useAuth());

// Check current route
console.log('Current route:', useLocation());

// Check session expiry
console.log('Session expires:', sessionExpiry);
```

#### **Server-Side Debugging**
```python
# Check token validation
print(f"Token valid: {token_valid}")

# Check session data
print(f"Session: {user_session}")

# Check user data
print(f"User: {user.to_dict()}")
```

For additional support, refer to the main documentation or contact the development team.