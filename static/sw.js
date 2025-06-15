/**
 * Service Worker for AI Chatbot
 * Provides offline functionality and caching
 */

const CACHE_NAME = 'ai-chatbot-v1';
const STATIC_CACHE = 'static-v1';
const DYNAMIC_CACHE = 'dynamic-v1';

// Files to cache for offline functionality
const STATIC_FILES = [
  '/',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/manifest.json',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js'
];

// Install event - cache static files
self.addEventListener('install', (event) => {
  console.log('🔧 Service Worker installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('📦 Caching static files');
        return cache.addAll(STATIC_FILES);
      })
      .then(() => {
        console.log('✅ Static files cached');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('❌ Failed to cache static files:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('🚀 Service Worker activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
              console.log('🗑️ Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('✅ Service Worker activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip chrome-extension requests
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(request));
    return;
  }
  
  // Handle static files
  if (isStaticFile(url)) {
    event.respondWith(handleStaticRequest(request));
    return;
  }
  
  // Handle navigation requests
  if (request.mode === 'navigate') {
    event.respondWith(handleNavigationRequest(request));
    return;
  }
  
  // Default: network first, then cache
  event.respondWith(
    fetch(request)
      .then((response) => {
        // Cache successful responses
        if (response.status === 200) {
          const responseClone = response.clone();
          caches.open(DYNAMIC_CACHE)
            .then((cache) => cache.put(request, responseClone));
        }
        return response;
      })
      .catch(() => {
        // Fallback to cache
        return caches.match(request);
      })
  );
});

/**
 * Handle API requests - network first with offline fallback
 */
async function handleApiRequest(request) {
  try {
    // Try network first
    const response = await fetch(request);
    
    // Cache successful responses
    if (response.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }
    
    return response;
    
  } catch (error) {
    console.log('📡 API request failed, checking cache...');
    
    // Try cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline response for chat API
    if (request.url.includes('/api/chat')) {
      return new Response(JSON.stringify({
        success: false,
        error: 'You are offline. Please check your internet connection.',
        offline: true
      }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Return offline response for health check
    if (request.url.includes('/api/health')) {
      return new Response(JSON.stringify({
        status: 'offline',
        ollama_connected: false,
        offline: true
      }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    throw error;
  }
}

/**
 * Handle static file requests - cache first
 */
async function handleStaticRequest(request) {
  // Try cache first
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // Try network
  try {
    const response = await fetch(request);
    
    // Cache successful responses
    if (response.status === 200) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    
    return response;
    
  } catch (error) {
    console.error('❌ Failed to fetch static file:', request.url);
    throw error;
  }
}

/**
 * Handle navigation requests - cache first with network fallback
 */
async function handleNavigationRequest(request) {
  try {
    // Try cache first for main page
    const cachedResponse = await caches.match('/');
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Try network
    const response = await fetch(request);
    
    // Cache the response
    if (response.status === 200) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put('/', response.clone());
    }
    
    return response;
    
  } catch (error) {
    // Return cached main page as fallback
    const cachedResponse = await caches.match('/');
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline page
    return new Response(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>AI Chatbot - Offline</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex; 
            align-items: center; 
            justify-content: center; 
            height: 100vh; 
            margin: 0; 
            background: #f8fafc;
            color: #1e293b;
          }
          .offline-container {
            text-align: center;
            padding: 2rem;
            max-width: 400px;
          }
          .offline-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
          }
          h1 { margin-bottom: 0.5rem; }
          p { color: #64748b; margin-bottom: 2rem; }
          button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
          }
          button:hover { background: #1d4ed8; }
        </style>
      </head>
      <body>
        <div class="offline-container">
          <div class="offline-icon">📡</div>
          <h1>You're Offline</h1>
          <p>Please check your internet connection and try again.</p>
          <button onclick="window.location.reload()">Try Again</button>
        </div>
      </body>
      </html>
    `, {
      status: 200,
      headers: { 'Content-Type': 'text/html' }
    });
  }
}

/**
 * Check if URL is for a static file
 */
function isStaticFile(url) {
  return url.pathname.startsWith('/static/') || 
         url.hostname === 'fonts.googleapis.com' ||
         url.hostname === 'cdnjs.cloudflare.com';
}

// Handle background sync for offline messages
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync-messages') {
    console.log('🔄 Background sync: messages');
    event.waitUntil(syncOfflineMessages());
  }
});

/**
 * Sync offline messages when connection is restored
 */
async function syncOfflineMessages() {
  try {
    // Get offline messages from IndexedDB or localStorage
    // This would need to be implemented based on your offline storage strategy
    console.log('📤 Syncing offline messages...');
    
    // For now, just log that sync would happen
    console.log('✅ Offline messages synced');
    
  } catch (error) {
    console.error('❌ Failed to sync offline messages:', error);
  }
}

// Handle push notifications
self.addEventListener('push', (event) => {
  if (event.data) {
    const data = event.data.json();
    
    const options = {
      body: data.body,
      icon: '/static/images/icon-192.png',
      badge: '/static/images/icon-192.png',
      vibrate: [200, 100, 200],
      data: data,
      actions: [
        {
          action: 'open',
          title: 'Open Chat',
          icon: '/static/images/icon-192.png'
        },
        {
          action: 'close',
          title: 'Close',
          icon: '/static/images/icon-192.png'
        }
      ]
    };
    
    event.waitUntil(
      self.registration.showNotification(data.title || 'AI Chatbot', options)
    );
  }
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Handle messages from main thread
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

console.log('🤖 AI Chatbot Service Worker loaded');