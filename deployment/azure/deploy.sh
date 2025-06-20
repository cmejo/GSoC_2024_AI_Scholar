#!/bin/bash

# Microsoft Azure Deployment Script for Enhanced AI Chatbot
# Deploys the application using Azure Container Instances and Azure Database for PostgreSQL

set -e

# Configuration
RESOURCE_GROUP=${RESOURCE_GROUP:-ai-chatbot-rg}
LOCATION=${LOCATION:-eastus}
CONTAINER_GROUP_NAME=${CONTAINER_GROUP_NAME:-ai-chatbot-cg}
CONTAINER_NAME=${CONTAINER_NAME:-ai-chatbot}
ACR_NAME=${ACR_NAME:-aichatbotacr$(date +%s)}
DB_SERVER_NAME=${DB_SERVER_NAME:-ai-chatbot-db-$(date +%s)}
DB_NAME=${DB_NAME:-chatbot_db}
DB_USER=${DB_USER:-chatbot_user}
DB_PASSWORD=${DB_PASSWORD}
SUBSCRIPTION_ID=${SUBSCRIPTION_ID}

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
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check required environment variables
    if [[ -z "$DB_PASSWORD" ]]; then
        print_error "DB_PASSWORD environment variable is required"
        exit 1
    fi
    
    # Check Azure authentication
    if ! az account show &> /dev/null; then
        print_error "Not authenticated with Azure. Please run 'az login'."
        exit 1
    fi
    
    # Set subscription if provided
    if [[ -n "$SUBSCRIPTION_ID" ]]; then
        az account set --subscription $SUBSCRIPTION_ID
    fi
    
    print_success "Prerequisites check passed"
}

create_resource_group() {
    print_status "Creating resource group..."
    
    az group create \
        --name $RESOURCE_GROUP \
        --location $LOCATION
    
    print_success "Resource group created"
}

create_container_registry() {
    print_status "Creating Azure Container Registry..."
    
    # Create ACR
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --admin-enabled true
    
    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show \
        --name $ACR_NAME \
        --resource-group $RESOURCE_GROUP \
        --query loginServer \
        --output tsv)
    
    print_success "Container registry created: $ACR_LOGIN_SERVER"
}

build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Build and push using ACR
    az acr build \
        --registry $ACR_NAME \
        --image $CONTAINER_NAME:latest \
        ../../
    
    print_success "Image built and pushed successfully"
}

create_postgresql_server() {
    print_status "Creating Azure Database for PostgreSQL..."
    
    # Create PostgreSQL server
    az postgres server create \
        --resource-group $RESOURCE_GROUP \
        --name $DB_SERVER_NAME \
        --location $LOCATION \
        --admin-user $DB_USER \
        --admin-password $DB_PASSWORD \
        --sku-name B_Gen5_1 \
        --version 11 \
        --storage-size 5120 \
        --ssl-enforcement Enabled
    
    # Create database
    az postgres db create \
        --resource-group $RESOURCE_GROUP \
        --server-name $DB_SERVER_NAME \
        --name $DB_NAME
    
    # Configure firewall to allow Azure services
    az postgres server firewall-rule create \
        --resource-group $RESOURCE_GROUP \
        --server $DB_SERVER_NAME \
        --name AllowAzureServices \
        --start-ip-address 0.0.0.0 \
        --end-ip-address 0.0.0.0
    
    # Get database FQDN
    DB_FQDN=$(az postgres server show \
        --resource-group $RESOURCE_GROUP \
        --name $DB_SERVER_NAME \
        --query fullyQualifiedDomainName \
        --output tsv)
    
    print_success "PostgreSQL server created: $DB_FQDN"
}

get_acr_credentials() {
    print_status "Getting ACR credentials..."
    
    ACR_USERNAME=$(az acr credential show \
        --name $ACR_NAME \
        --resource-group $RESOURCE_GROUP \
        --query username \
        --output tsv)
    
    ACR_PASSWORD=$(az acr credential show \
        --name $ACR_NAME \
        --resource-group $RESOURCE_GROUP \
        --query passwords[0].value \
        --output tsv)
    
    print_success "ACR credentials retrieved"
}

deploy_container_group() {
    print_status "Deploying container group..."
    
    # Generate secrets
    SECRET_KEY=$(openssl rand -base64 32)
    JWT_SECRET_KEY=$(openssl rand -base64 32)
    
    # Create container group
    az container create \
        --resource-group $RESOURCE_GROUP \
        --name $CONTAINER_GROUP_NAME \
        --image $ACR_LOGIN_SERVER/$CONTAINER_NAME:latest \
        --registry-login-server $ACR_LOGIN_SERVER \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --dns-name-label $CONTAINER_GROUP_NAME \
        --ports 5000 \
        --cpu 1 \
        --memory 2 \
        --environment-variables \
            FLASK_ENV=production \
            SECRET_KEY="$SECRET_KEY" \
            JWT_SECRET_KEY="$JWT_SECRET_KEY" \
            DB_HOST="$DB_FQDN" \
            DB_PORT=5432 \
            DB_NAME="$DB_NAME" \
            DB_USER="$DB_USER@$DB_SERVER_NAME" \
            DB_PASSWORD="$DB_PASSWORD"
    
    # Get container group FQDN
    CONTAINER_FQDN=$(az container show \
        --resource-group $RESOURCE_GROUP \
        --name $CONTAINER_GROUP_NAME \
        --query ipAddress.fqdn \
        --output tsv)
    
    print_success "Container group deployed: http://$CONTAINER_FQDN:5000"
}

setup_application_gateway() {
    print_status "Setting up Application Gateway (optional)..."
    
    if [[ "$SETUP_APP_GATEWAY" == "true" ]]; then
        # Create virtual network
        az network vnet create \
            --resource-group $RESOURCE_GROUP \
            --name ai-chatbot-vnet \
            --address-prefix 10.0.0.0/16 \
            --subnet-name default \
            --subnet-prefix 10.0.0.0/24
        
        # Create public IP
        az network public-ip create \
            --resource-group $RESOURCE_GROUP \
            --name ai-chatbot-pip \
            --allocation-method Static \
            --sku Standard
        
        # Create Application Gateway
        az network application-gateway create \
            --name ai-chatbot-appgw \
            --location $LOCATION \
            --resource-group $RESOURCE_GROUP \
            --vnet-name ai-chatbot-vnet \
            --subnet default \
            --capacity 2 \
            --sku Standard_v2 \
            --http-settings-cookie-based-affinity Disabled \
            --frontend-port 80 \
            --http-settings-port 5000 \
            --http-settings-protocol Http \
            --public-ip-address ai-chatbot-pip \
            --servers $CONTAINER_FQDN
        
        # Get Application Gateway public IP
        APP_GATEWAY_IP=$(az network public-ip show \
            --resource-group $RESOURCE_GROUP \
            --name ai-chatbot-pip \
            --query ipAddress \
            --output tsv)
        
        print_success "Application Gateway created: http://$APP_GATEWAY_IP"
    else
        print_warning "Application Gateway setup skipped (set SETUP_APP_GATEWAY=true to enable)"
    fi
}

setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Create Log Analytics workspace
    az monitor log-analytics workspace create \
        --resource-group $RESOURCE_GROUP \
        --workspace-name ai-chatbot-logs \
        --location $LOCATION
    
    # Get workspace ID
    WORKSPACE_ID=$(az monitor log-analytics workspace show \
        --resource-group $RESOURCE_GROUP \
        --workspace-name ai-chatbot-logs \
        --query customerId \
        --output tsv)
    
    # Get workspace key
    WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
        --resource-group $RESOURCE_GROUP \
        --workspace-name ai-chatbot-logs \
        --query primarySharedKey \
        --output tsv)
    
    print_success "Log Analytics workspace created"
    print_status "Workspace ID: $WORKSPACE_ID"
}

create_backup_strategy() {
    print_status "Setting up backup strategy..."
    
    # Enable automated backups for PostgreSQL (already enabled by default)
    print_status "PostgreSQL automated backups are enabled by default (7-day retention)"
    
    # Create additional backup configuration if needed
    az postgres server configuration set \
        --resource-group $RESOURCE_GROUP \
        --server-name $DB_SERVER_NAME \
        --name backup_retention_days \
        --value 30
    
    print_success "Backup strategy configured"
}

print_deployment_info() {
    print_success "Deployment completed successfully!"
    echo ""
    echo "=== Deployment Information ==="
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Location: $LOCATION"
    echo "Container Group: $CONTAINER_GROUP_NAME"
    echo "Application URL: http://$CONTAINER_FQDN:5000"
    echo "Database Server: $DB_FQDN"
    echo "Container Registry: $ACR_LOGIN_SERVER"
    
    if [[ -n "$APP_GATEWAY_IP" ]]; then
        echo "Application Gateway: http://$APP_GATEWAY_IP"
    fi
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Access your application at: http://$CONTAINER_FQDN:5000"
    echo "2. Check container logs: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_GROUP_NAME"
    echo "3. Monitor container: az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_GROUP_NAME"
    echo ""
    echo "=== Useful Commands ==="
    echo "Restart container: az container restart --resource-group $RESOURCE_GROUP --name $CONTAINER_GROUP_NAME"
    echo "Update container: az container create [with updated image]"
    echo "Scale database: az postgres server update --resource-group $RESOURCE_GROUP --name $DB_SERVER_NAME --sku-name GP_Gen5_2"
    echo ""
    echo "=== Cleanup ==="
    echo "To delete all resources, run: ./cleanup.sh"
}

main() {
    print_status "Starting Azure deployment for Enhanced AI Chatbot..."
    
    check_prerequisites
    create_resource_group
    create_container_registry
    build_and_push_image
    create_postgresql_server
    get_acr_credentials
    deploy_container_group
    setup_application_gateway
    setup_monitoring
    create_backup_strategy
    print_deployment_info
}

# Run main function
main "$@"