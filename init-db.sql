-- PostgreSQL initialization script for AI Chatbot application
-- This script sets up the database with proper permissions and extensions

-- Create the main database if it doesn't exist (handled by POSTGRES_DB env var)
-- CREATE DATABASE chatbot_db;

-- Connect to the database
\c chatbot_db;

-- Create extensions that might be useful
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For better indexing

-- Create a schema for the application (optional, using public for simplicity)
-- CREATE SCHEMA IF NOT EXISTS chatbot;

-- Grant necessary permissions to the application user
GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO chatbot_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO chatbot_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chatbot_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO chatbot_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO chatbot_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO chatbot_user;

-- Create indexes that will be useful for the application
-- Note: These will be created by SQLAlchemy migrations, but we can prepare for them

-- Performance optimization settings
-- These are applied at the database level for better performance
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET pg_stat_statements.track = 'all';

-- Reload configuration
SELECT pg_reload_conf();

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
END $$;