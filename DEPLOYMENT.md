# Deployment Guide

This guide covers different ways to deploy the AI Chatbot Web GUI for production use.

## Local Development

### Quick Start
```bash
# Clone/download the project
# Install dependencies
pip install -r requirements.txt

# Start Ollama (if using)
ollama serve

# Run the application
python app.py
```

### Using the Startup Scripts
```bash
# Linux/Mac
./start.sh

# Windows
start.bat
```

## Production Deployment

### 1. Using Gunicorn (Recommended for Linux)

#### Install Gunicorn
```bash
pip install gunicorn eventlet
```

#### Create Gunicorn Configuration
```python
# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "eventlet"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
```

#### Run with Gunicorn
```bash
gunicorn --config gunicorn.conf.py app:app
```

### 2. Using Docker

#### Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn eventlet

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 chatbot && chown -R chatbot:chatbot /app
USER chatbot

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "eventlet", "app:app"]
```

#### Create docker-compose.yml
```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "5000:5000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - DEFAULT_MODEL=llama2
      - SECRET_KEY=your-production-secret-key
    depends_on:
      - ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  ollama_data:
```

#### Deploy with Docker
```bash
# Build and run
docker-compose up -d

# Pull AI model
docker-compose exec ollama ollama pull llama2

# View logs
docker-compose logs -f chatbot
```

### 3. Using Nginx (Reverse Proxy)

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/chatbot
server {
    listen 80;
    server_name your-domain.com;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Static files
    location /static/ {
        alias /path/to/your/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # WebSocket support
    location /socket.io/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

#### Enable and Start
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 4. Using Systemd Service

#### Create Service File
```ini
# /etc/systemd/system/chatbot.service
[Unit]
Description=AI Chatbot Web GUI
After=network.target

[Service]
Type=exec
User=chatbot
Group=chatbot
WorkingDirectory=/opt/chatbot
Environment=PATH=/opt/chatbot/venv/bin
Environment=OLLAMA_BASE_URL=http://localhost:11434
Environment=DEFAULT_MODEL=llama2
Environment=SECRET_KEY=your-production-secret-key
ExecStart=/opt/chatbot/venv/bin/gunicorn --config gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable chatbot

# Start service
sudo systemctl start chatbot

# Check status
sudo systemctl status chatbot
```

## Cloud Deployment

### 1. Heroku

#### Create Procfile
```
web: gunicorn --worker-class eventlet -w 1 app:app
```

#### Deploy
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create your-chatbot-app

# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set OLLAMA_BASE_URL=your-ollama-url

# Deploy
git push heroku main
```

### 2. Railway

#### Create railway.toml
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "gunicorn --worker-class eventlet -w 1 app:app"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

### 3. DigitalOcean App Platform

#### Create .do/app.yaml
```yaml
name: ai-chatbot
services:
- name: web
  source_dir: /
  github:
    repo: your-username/your-repo
    branch: main
  run_command: gunicorn --worker-class eventlet -w 1 app:app
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: SECRET_KEY
    value: your-secret-key
  - key: OLLAMA_BASE_URL
    value: your-ollama-url
  http_port: 5000
```

## Environment Variables

### Required
- `SECRET_KEY`: Flask session secret key
- `OLLAMA_BASE_URL`: Ollama API endpoint
- `DEFAULT_MODEL`: AI model to use

### Optional
- `FLASK_ENV`: Environment (development/production)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)

## Security Considerations

### 1. HTTPS/SSL
Always use HTTPS in production:
```bash
# Using Certbot with Nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Environment Variables
Never commit secrets to version control:
```bash
# Use .env file for local development
echo "SECRET_KEY=your-secret-key" > .env

# Use environment variables in production
export SECRET_KEY=your-production-secret-key
```

### 3. Firewall
Restrict access to necessary ports:
```bash
# UFW example
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

### 4. Rate Limiting
Consider implementing rate limiting:
```python
# Add to app.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat_api():
    # ... existing code
```

## Monitoring

### 1. Health Checks
The application includes a health endpoint at `/api/health`

### 2. Logging
Configure proper logging:
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/chatbot.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

### 3. Metrics
Consider using tools like:
- Prometheus + Grafana
- New Relic
- DataDog

## Backup and Recovery

### 1. Database Backup
If using a database, implement regular backups

### 2. Configuration Backup
Backup your configuration files and environment variables

### 3. Model Backup
Backup your AI models if using custom ones

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if eventlet is installed
   - Verify proxy configuration for WebSockets

2. **Ollama Connection Error**
   - Ensure Ollama is running
   - Check firewall settings
   - Verify OLLAMA_BASE_URL

3. **High Memory Usage**
   - Reduce number of workers
   - Implement conversation history limits
   - Monitor AI model memory usage

4. **Slow Response Times**
   - Use faster AI models
   - Implement caching
   - Optimize database queries

### Logs
Check application logs for errors:
```bash
# Systemd service
sudo journalctl -u chatbot -f

# Docker
docker-compose logs -f chatbot

# Direct run
tail -f logs/chatbot.log
```