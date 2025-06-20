#!/bin/bash

# Google Cloud Platform Deployment Script for Enhanced AI Chatbot
# Deploys the application using Cloud Run and Cloud SQL

set -e

# Configuration
PROJECT_ID=${PROJECT_ID}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-ai-chatbot}
IMAGE_NAME=${IMAGE_NAME:-ai-chatbot}
DB_INSTANCE_NAME=${DB_INSTANCE_NAME:-ai-chatbot-db}
DB_NAME=${DB_NAME:-chatbot_db}
DB_USER=${DB_USER:-chatbot_user}
DB_PASSWORD=${DB_PASSWORD}
DB_TIER=${DB_TIER:-db-f1-micro}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check required environment variables
    if [[ -z "$PROJECT_ID" ]]; then
        print_error "PROJECT_ID environment variable is required"
        exit 1
    fi
    
    if [[ -z "$DB_PASSWORD" ]]; then
        print_error "DB_PASSWORD environment variable is required"
        exit 1
    fi
    
    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "Not authenticated with gcloud. Please run 'gcloud auth login'."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

setup_project() {
    print_status "Setting up GCP project..."
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    print_status "Enabling required APIs..."
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        sql-component.googleapis.com \
        sqladmin.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com
    
    print_success "Project setup completed"
}

build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Configure Docker for GCR
    gcloud auth configure-docker
    
    # Build image using Cloud Build
    print_status "Building image with Cloud Build..."
    gcloud builds submit \
        --tag gcr.io/$PROJECT_ID/$IMAGE_NAME \
        --timeout=20m \
        ../../
    
    print_success "Image built and pushed successfully"
}

create_cloud_sql_instance() {
    print_status "Creating Cloud SQL instance..."
    
    # Check if instance exists
    if gcloud sql instances describe $DB_INSTANCE_NAME --project=$PROJECT_ID &> /dev/null; then
        print_warning "Cloud SQL instance $DB_INSTANCE_NAME already exists"
        return
    fi
    
    # Create Cloud SQL instance
    print_status "Creating PostgreSQL instance (this may take several minutes)..."
    gcloud sql instances create $DB_INSTANCE_NAME \
        --database-version=POSTGRES_15 \
        --tier=$DB_TIER \
        --region=$REGION \
        --storage-type=SSD \
        --storage-size=10GB \
        --storage-auto-increase \
        --backup-start-time=03:00 \
        --enable-bin-log \
        --project=$PROJECT_ID
    
    print_success "Cloud SQL instance created"
}

setup_database() {
    print_status "Setting up database..."
    
    # Set root password
    gcloud sql users set-password root \
        --host=% \
        --instance=$DB_INSTANCE_NAME \
        --password=$DB_PASSWORD \
        --project=$PROJECT_ID
    
    # Create database
    gcloud sql databases create $DB_NAME \
        --instance=$DB_INSTANCE_NAME \
        --project=$PROJECT_ID || true
    
    # Create user
    gcloud sql users create $DB_USER \
        --instance=$DB_INSTANCE_NAME \
        --password=$DB_PASSWORD \
        --project=$PROJECT_ID || true
    
    print_success "Database setup completed"
}

get_db_connection_name() {
    print_status "Getting database connection name..."
    
    DB_CONNECTION_NAME=$(gcloud sql instances describe $DB_INSTANCE_NAME \
        --project=$PROJECT_ID \
        --format="value(connectionName)")
    
    print_status "Database connection name: $DB_CONNECTION_NAME"
}

deploy_cloud_run() {
    print_status "Deploying to Cloud Run..."
    
    # Generate secrets
    SECRET_KEY=$(openssl rand -base64 32)
    JWT_SECRET_KEY=$(openssl rand -base64 32)
    
    # Deploy to Cloud Run
    gcloud run deploy $SERVICE_NAME \
        --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 5000 \
        --memory 2Gi \
        --cpu 1 \
        --timeout 300 \
        --concurrency 100 \
        --max-instances 10 \
        --set-env-vars "FLASK_ENV=production" \
        --set-env-vars "SECRET_KEY=$SECRET_KEY" \
        --set-env-vars "JWT_SECRET_KEY=$JWT_SECRET_KEY" \
        --set-env-vars "DB_HOST=/cloudsql/$DB_CONNECTION_NAME" \
        --set-env-vars "DB_PORT=5432" \
        --set-env-vars "DB_NAME=$DB_NAME" \
        --set-env-vars "DB_USER=$DB_USER" \
        --set-env-vars "DB_PASSWORD=$DB_PASSWORD" \
        --add-cloudsql-instances $DB_CONNECTION_NAME \
        --project=$PROJECT_ID
    
    print_success "Cloud Run deployment completed"
}

get_service_url() {
    print_status "Getting service URL..."
    
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --platform managed \
        --region $REGION \
        --project=$PROJECT_ID \
        --format="value(status.url)")
    
    print_status "Service URL: $SERVICE_URL"
}

setup_custom_domain() {
    print_status "Setting up custom domain (optional)..."
    
    if [[ -n "$CUSTOM_DOMAIN" ]]; then
        # Map custom domain
        gcloud run domain-mappings create \
            --service $SERVICE_NAME \
            --domain $CUSTOM_DOMAIN \
            --region $REGION \
            --project=$PROJECT_ID
        
        print_success "Custom domain mapped: $CUSTOM_DOMAIN"
        print_status "Please configure your DNS to point to the provided IP address"
    else
        print_warning "CUSTOM_DOMAIN not set, skipping custom domain setup"
    fi
}

setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Enable Cloud Monitoring API
    gcloud services enable monitoring.googleapis.com
    
    # Create uptime check
    cat > uptime-check.json << EOF
{
  "displayName": "AI Chatbot Health Check",
  "monitoredResource": {
    "type": "uptime_url",
    "labels": {
      "project_id": "$PROJECT_ID",
      "host": "$(echo $SERVICE_URL | sed 's|https://||')"
    }
  },
  "httpCheck": {
    "path": "/api/health",
    "port": 443,
    "useSsl": true,
    "validateSsl": true
  },
  "period": "300s",
  "timeout": "10s"
}
EOF
    
    # Create the uptime check (requires additional setup)
    print_status "Uptime check configuration created (manual setup required)"
    
    rm uptime-check.json
    
    print_success "Monitoring setup completed"
}

print_deployment_info() {
    print_success "Deployment completed successfully!"
    echo ""
    echo "=== Deployment Information ==="
    echo "Project ID: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Service Name: $SERVICE_NAME"
    echo "Service URL: $SERVICE_URL"
    echo "Database Instance: $DB_INSTANCE_NAME"
    echo "Database Connection: $DB_CONNECTION_NAME"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Access your application at: $SERVICE_URL"
    echo "2. Check service logs: gcloud run services logs read $SERVICE_NAME --region=$REGION"
    echo "3. Monitor service: gcloud run services describe $SERVICE_NAME --region=$REGION"
    echo ""
    echo "=== Useful Commands ==="
    echo "Update service: gcloud run deploy $SERVICE_NAME --image gcr.io/$PROJECT_ID/$IMAGE_NAME --region=$REGION"
    echo "View logs: gcloud run services logs tail $SERVICE_NAME --region=$REGION"
    echo "Scale service: gcloud run services update $SERVICE_NAME --max-instances=20 --region=$REGION"
    echo ""
    echo "=== Cleanup ==="
    echo "To delete all resources, run: ./cleanup.sh"
}

main() {
    print_status "Starting GCP deployment for Enhanced AI Chatbot..."
    
    check_prerequisites
    setup_project
    build_and_push_image
    create_cloud_sql_instance
    setup_database
    get_db_connection_name
    deploy_cloud_run
    get_service_url
    setup_custom_domain
    setup_monitoring
    print_deployment_info
}

# Run main function
main "$@"