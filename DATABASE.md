# Database Setup and Management

This document provides comprehensive information about the PostgreSQL database setup, schema management, and testing for the AI Chatbot application.

## Table of Contents

- [Database Architecture](#database-architecture)
- [Setup Instructions](#setup-instructions)
- [Schema Management](#schema-management)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## Database Architecture

### Database Schema

The application uses PostgreSQL with the following main tables:

#### Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

#### Chat Sessions Table
```sql
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Chat Messages Table
```sql
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    message_type message_type_enum NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used VARCHAR(100),
    response_time FLOAT,
    token_count INTEGER
);
```

#### User Sessions Table
```sql
CREATE TABLE user_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE
);
```

### Indexes

The following indexes are created for optimal performance:

- `ix_users_username` - Unique index on username
- `ix_users_email` - Unique index on email
- `ix_chat_sessions_user_id` - Index on user_id for sessions
- `ix_chat_messages_session_id` - Index on session_id for messages
- `ix_chat_messages_timestamp` - Index on timestamp for chronological queries
- `ix_chat_messages_session_timestamp` - Composite index for session + timestamp
- `ix_user_sessions_expires_active` - Composite index for session cleanup

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local development)
- PostgreSQL client tools (optional, for direct database access)

### Environment Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Configure database settings in `.env`:
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chatbot_db
DB_USER=chatbot_user
DB_PASSWORD=chatbot_password

# Test Database Configuration
TEST_DB_NAME=chatbot_test_db
TEST_DB_USER=chatbot_test_user
TEST_DB_PASSWORD=chatbot_test_password
```

### Docker Setup

#### Production Environment
```bash
# Start PostgreSQL and application
docker-compose up -d

# Check database health
docker-compose exec postgres pg_isready -U chatbot_user -d chatbot_db
```

#### Development Environment
```bash
# Start development environment with separate databases
docker-compose -f docker-compose.dev.yml up -d

# Development database (port 5433)
docker-compose -f docker-compose.dev.yml exec postgres-dev pg_isready

# Test database (port 5434)
docker-compose -f docker-compose.dev.yml exec postgres-test pg_isready
```

### Local Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up local PostgreSQL (if not using Docker):
```bash
# Create databases
createdb chatbot_db
createdb chatbot_test_db

# Create users
createuser chatbot_user
createuser chatbot_test_user

# Grant permissions
psql -c "GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO chatbot_user;"
psql -c "GRANT ALL PRIVILEGES ON DATABASE chatbot_test_db TO chatbot_test_user;"
```

## Schema Management

### Database Migrations

The application uses Flask-Migrate (Alembic) for schema management.

#### Initialize Migrations (First Time)
```bash
python manage_db.py init-migrations
```

#### Create New Migration
```bash
python manage_db.py create-migration --message "Add new feature"
```

#### Apply Migrations
```bash
# Upgrade to latest
python manage_db.py upgrade-db

# Downgrade to previous version
python manage_db.py downgrade-db --revision -1
```

#### Check Migration Status
```bash
# Current revision
python manage_db.py current-revision

# Migration history
python manage_db.py migration-history
```

### Manual Database Operations

#### Create Tables (Without Migrations)
```bash
python manage_db.py create-tables
```

#### Reset Database
```bash
python manage_db.py reset-db
```

#### Test Database Connection
```bash
python manage_db.py test-connection
```

## Testing

### Test Database Setup

The application uses a separate test database to ensure test isolation.

#### Automated Test Setup
```bash
# Run all tests with database setup
python run_tests.py --type all --coverage

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type api
python run_tests.py --type integration
```

#### Manual Test Database Setup
```bash
# Set test environment
export DB_NAME=chatbot_test_db
export DB_USER=chatbot_test_user
export DB_PASSWORD=chatbot_test_password

# Run tests
pytest tests/ -v
```

### Test Categories

#### Unit Tests (`tests/test_models.py`)
- Database model functionality
- Data validation and constraints
- Model relationships and cascading

#### API Tests (`tests/test_auth_api.py`, `tests/test_chat_api.py`)
- Authentication endpoints
- Chat functionality
- Session management
- Error handling

#### Integration Tests (`tests/test_integration.py`)
- Complete user workflows
- Cross-component interactions
- Data consistency
- Performance scenarios

### Test Coverage

Generate coverage reports:
```bash
# HTML coverage report
python run_tests.py --coverage --html-report

# View coverage
open htmlcov/index.html
```

## Development Workflow

### Daily Development

1. **Start Development Environment**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

2. **Apply Any New Migrations**
```bash
python manage_db.py upgrade-db
```

3. **Run Tests Before Changes**
```bash
python run_tests.py --type all
```

4. **Make Code Changes**

5. **Create Migration (if schema changed)**
```bash
python manage_db.py create-migration --message "Description of changes"
```

6. **Run Tests After Changes**
```bash
python run_tests.py --type all --coverage
```

7. **Commit Changes**
```bash
git add .
git commit -m "Description of changes"
```

### Database Maintenance

#### Clean Up Expired Sessions
```bash
python manage_db.py cleanup-sessions
```

#### Get User Statistics
```bash
# General stats
python manage_db.py user-stats

# Specific user
python manage_db.py user-stats --user-id 1
```

#### Create Admin User
```bash
python manage_db.py create-admin
```

## Production Deployment

### Environment Variables

Set the following environment variables in production:

```env
# Production database URL
DATABASE_URL=postgresql://username:password@host:port/database

# Or individual components
DB_HOST=your-postgres-host
DB_PORT=5432
DB_NAME=chatbot_production
DB_USER=chatbot_prod_user
DB_PASSWORD=secure-production-password

# Security
SECRET_KEY=your-secure-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
```

### Database Backup

#### Create Backup
```bash
# Using Docker
docker-compose exec postgres pg_dump -U chatbot_user chatbot_db > backup.sql

# Direct connection
pg_dump -h localhost -U chatbot_user chatbot_db > backup.sql
```

#### Restore Backup
```bash
# Using Docker
docker-compose exec -T postgres psql -U chatbot_user chatbot_db < backup.sql

# Direct connection
psql -h localhost -U chatbot_user chatbot_db < backup.sql
```

### Performance Optimization

#### Database Configuration

Add to PostgreSQL configuration:
```sql
-- Increase shared buffers
shared_buffers = 256MB

-- Increase work memory
work_mem = 4MB

-- Enable query planning statistics
shared_preload_libraries = 'pg_stat_statements'
```

#### Monitoring Queries

```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

#### Migration Errors
```bash
# Check current migration status
python manage_db.py current-revision

# Force migration to specific revision
python manage_db.py downgrade-db --revision base
python manage_db.py upgrade-db
```

#### Test Database Issues
```bash
# Reset test database
export DB_NAME=chatbot_test_db
python manage_db.py reset-db

# Run tests with fresh database
python run_tests.py --type all
```

#### Permission Errors
```sql
-- Grant all permissions to user
GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO chatbot_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO chatbot_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chatbot_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO chatbot_user;
```

### Performance Issues

#### Slow Queries
```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1s

-- Reload configuration
SELECT pg_reload_conf();
```

#### Index Analysis
```sql
-- Check unused indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_tup_read DESC;
```

### Data Integrity

#### Check Constraints
```sql
-- Verify foreign key constraints
SELECT conname, conrelid::regclass, confrelid::regclass
FROM pg_constraint
WHERE contype = 'f';

-- Check for orphaned records
SELECT COUNT(*) FROM chat_messages cm
LEFT JOIN chat_sessions cs ON cm.session_id = cs.id
WHERE cs.id IS NULL;
```

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Flask-SQLAlchemy Documentation](https://flask-sqlalchemy.palletsprojects.com/)
- [Flask-Migrate Documentation](https://flask-migrate.readthedocs.io/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)

For additional help, check the application logs or contact the development team.