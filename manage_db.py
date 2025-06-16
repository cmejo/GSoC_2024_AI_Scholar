#!/usr/bin/env python3
"""
Database management script for AI Chatbot application
Provides commands for database initialization, migration, and management
"""

import os
import sys
import click
from flask import Flask
from flask_migrate import Migrate, init, migrate, upgrade, downgrade, current, history
from models import db, User, ChatSession, ChatMessage, UserSession, DatabaseQueries
import subprocess


def create_app():
    """Create Flask app for database operations"""
    app = Flask(__name__)
    
    # Database configuration
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL:
        app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    else:
        DB_HOST = os.environ.get('DB_HOST', 'localhost')
        DB_PORT = os.environ.get('DB_PORT', '5432')
        DB_NAME = os.environ.get('DB_NAME', 'chatbot_db')
        DB_USER = os.environ.get('DB_USER', 'chatbot_user')
        DB_PASSWORD = os.environ.get('DB_PASSWORD', 'chatbot_password')
        
        app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    db.init_app(app)
    migrate = Migrate(app, db)
    
    return app, migrate


@click.group()
def cli():
    """Database management commands"""
    pass


@cli.command()
def init_migrations():
    """Initialize migration repository"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            init()
            click.echo("✅ Migration repository initialized")
        except Exception as e:
            click.echo(f"❌ Failed to initialize migrations: {e}")


@cli.command()
@click.option('--message', '-m', default='Auto migration', help='Migration message')
def create_migration(message):
    """Create a new migration"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            migrate(message=message)
            click.echo(f"✅ Migration created: {message}")
        except Exception as e:
            click.echo(f"❌ Failed to create migration: {e}")


@cli.command()
def upgrade_db():
    """Upgrade database to latest migration"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            upgrade()
            click.echo("✅ Database upgraded successfully")
        except Exception as e:
            click.echo(f"❌ Failed to upgrade database: {e}")


@cli.command()
@click.option('--revision', '-r', default='-1', help='Revision to downgrade to')
def downgrade_db(revision):
    """Downgrade database to specific revision"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            downgrade(revision=revision)
            click.echo(f"✅ Database downgraded to revision: {revision}")
        except Exception as e:
            click.echo(f"❌ Failed to downgrade database: {e}")


@cli.command()
def current_revision():
    """Show current database revision"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            current()
        except Exception as e:
            click.echo(f"❌ Failed to get current revision: {e}")


@cli.command()
def migration_history():
    """Show migration history"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            history()
        except Exception as e:
            click.echo(f"❌ Failed to get migration history: {e}")


@cli.command()
def create_tables():
    """Create all database tables (without migrations)"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            db.create_all()
            click.echo("✅ Database tables created successfully")
        except Exception as e:
            click.echo(f"❌ Failed to create tables: {e}")


@cli.command()
def drop_tables():
    """Drop all database tables"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            db.drop_all()
            click.echo("✅ Database tables dropped successfully")
        except Exception as e:
            click.echo(f"❌ Failed to drop tables: {e}")


@cli.command()
def reset_db():
    """Reset database (drop and recreate all tables)"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            db.drop_all()
            db.create_all()
            click.echo("✅ Database reset successfully")
        except Exception as e:
            click.echo(f"❌ Failed to reset database: {e}")


@cli.command()
@click.option('--username', prompt=True, help='Username for the admin user')
@click.option('--email', prompt=True, help='Email for the admin user')
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True, help='Password for the admin user')
def create_admin(username, email, password):
    """Create an admin user"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            # Check if user already exists
            existing_user = User.query.filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                click.echo(f"❌ User with username '{username}' or email '{email}' already exists")
                return
            
            # Create admin user
            admin_user = User(username=username, email=email, password=password)
            db.session.add(admin_user)
            db.session.commit()
            
            click.echo(f"✅ Admin user '{username}' created successfully")
        except Exception as e:
            db.session.rollback()
            click.echo(f"❌ Failed to create admin user: {e}")


@cli.command()
def cleanup_sessions():
    """Clean up expired user sessions"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            count = DatabaseQueries.cleanup_expired_sessions()
            click.echo(f"✅ Cleaned up {count} expired sessions")
        except Exception as e:
            click.echo(f"❌ Failed to cleanup sessions: {e}")


@cli.command()
@click.option('--user-id', type=int, help='User ID to get stats for')
def user_stats(user_id):
    """Get user statistics"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            if user_id:
                stats = DatabaseQueries.get_user_stats(user_id)
                if stats:
                    click.echo(f"User Statistics for {stats['username']}:")
                    click.echo(f"  Total Sessions: {stats['total_sessions']}")
                    click.echo(f"  Total Messages: {stats['total_messages']}")
                    click.echo(f"  User Messages: {stats['user_messages']}")
                    click.echo(f"  Bot Messages: {stats['bot_messages']}")
                    click.echo(f"  Member Since: {stats['member_since']}")
                    click.echo(f"  Last Login: {stats['last_login']}")
                else:
                    click.echo(f"❌ User with ID {user_id} not found")
            else:
                # Show general stats
                total_users = User.query.count()
                total_sessions = ChatSession.query.count()
                total_messages = ChatMessage.query.count()
                
                click.echo("General Statistics:")
                click.echo(f"  Total Users: {total_users}")
                click.echo(f"  Total Sessions: {total_sessions}")
                click.echo(f"  Total Messages: {total_messages}")
        except Exception as e:
            click.echo(f"❌ Failed to get user stats: {e}")


@cli.command()
def test_connection():
    """Test database connection"""
    app, migrate_obj = create_app()
    with app.app_context():
        try:
            # Try to execute a simple query
            result = db.session.execute('SELECT 1').scalar()
            if result == 1:
                click.echo("✅ Database connection successful")
                
                # Show database info
                db_url = app.config['SQLALCHEMY_DATABASE_URI']
                # Hide password in output
                safe_url = db_url.split('@')[1] if '@' in db_url else db_url
                click.echo(f"📍 Connected to: {safe_url}")
                
                # Check if tables exist
                inspector = db.inspect(db.engine)
                tables = inspector.get_table_names()
                click.echo(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
            else:
                click.echo("❌ Database connection failed")
        except Exception as e:
            click.echo(f"❌ Database connection error: {e}")


if __name__ == '__main__':
    # Load environment variables from .env file if it exists
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    cli()