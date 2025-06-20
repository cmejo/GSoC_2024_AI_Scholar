#!/bin/bash

# AWS Deployment Script for Enhanced AI Chatbot
# Deploys the application using AWS ECS with Fargate

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-west-2}
CLUSTER_NAME=${CLUSTER_NAME:-ai-chatbot-cluster}
SERVICE_NAME=${SERVICE_NAME:-ai-chatbot-service}
TASK_FAMILY=${TASK_FAMILY:-ai-chatbot-task}
ECR_REPOSITORY=${ECR_REPOSITORY:-ai-chatbot}
IMAGE_TAG=${IMAGE_TAG:-latest}
VPC_ID=${VPC_ID}
SUBNET_IDS=${SUBNET_IDS}
SECURITY_GROUP_ID=${SECURITY_GROUP_ID}

# Database configuration
DB_INSTANCE_CLASS=${DB_INSTANCE_CLASS:-db.t3.micro}
DB_ALLOCATED_STORAGE=${DB_ALLOCATED_STORAGE:-20}
DB_NAME=${DB_NAME:-chatbot_db}
DB_USERNAME=${DB_USERNAME:-chatbot_user}
DB_PASSWORD=${DB_PASSWORD}

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
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    # Check required environment variables
    if [[ -z "$VPC_ID" ]]; then
        print_error "VPC_ID environment variable is required"
        exit 1
    fi
    
    if [[ -z "$SUBNET_IDS" ]]; then
        print_error "SUBNET_IDS environment variable is required (comma-separated)"
        exit 1
    fi
    
    if [[ -z "$DB_PASSWORD" ]]; then
        print_error "DB_PASSWORD environment variable is required"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

create_ecr_repository() {
    print_status "Creating ECR repository..."
    
    # Check if repository exists
    if aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION &> /dev/null; then
        print_warning "ECR repository $ECR_REPOSITORY already exists"
    else
        aws ecr create-repository \
            --repository-name $ECR_REPOSITORY \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true
        print_success "ECR repository created"
    fi
    
    # Get repository URI
    ECR_URI=$(aws ecr describe-repositories \
        --repository-names $ECR_REPOSITORY \
        --region $AWS_REGION \
        --query 'repositories[0].repositoryUri' \
        --output text)
    
    print_status "ECR Repository URI: $ECR_URI"
}

build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Get ECR login token
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Build image
    print_status "Building Docker image..."
    docker build -t $ECR_REPOSITORY:$IMAGE_TAG ../../
    
    # Tag image
    docker tag $ECR_REPOSITORY:$IMAGE_TAG $ECR_URI:$IMAGE_TAG
    
    # Push image
    print_status "Pushing image to ECR..."
    docker push $ECR_URI:$IMAGE_TAG
    
    print_success "Image pushed successfully"
}

create_rds_instance() {
    print_status "Creating RDS PostgreSQL instance..."
    
    # Check if DB instance exists
    if aws rds describe-db-instances --db-instance-identifier $DB_NAME --region $AWS_REGION &> /dev/null; then
        print_warning "RDS instance $DB_NAME already exists"
        return
    fi
    
    # Create DB subnet group
    print_status "Creating DB subnet group..."
    aws rds create-db-subnet-group \
        --db-subnet-group-name ${DB_NAME}-subnet-group \
        --db-subnet-group-description "Subnet group for $DB_NAME" \
        --subnet-ids $(echo $SUBNET_IDS | tr ',' ' ') \
        --region $AWS_REGION || true
    
    # Create RDS instance
    print_status "Creating RDS PostgreSQL instance (this may take several minutes)..."
    aws rds create-db-instance \
        --db-instance-identifier $DB_NAME \
        --db-instance-class $DB_INSTANCE_CLASS \
        --engine postgres \
        --engine-version 15.4 \
        --master-username $DB_USERNAME \
        --master-user-password $DB_PASSWORD \
        --allocated-storage $DB_ALLOCATED_STORAGE \
        --vpc-security-group-ids $SECURITY_GROUP_ID \
        --db-subnet-group-name ${DB_NAME}-subnet-group \
        --backup-retention-period 7 \
        --storage-encrypted \
        --region $AWS_REGION
    
    # Wait for DB to be available
    print_status "Waiting for RDS instance to be available..."
    aws rds wait db-instance-available \
        --db-instance-identifier $DB_NAME \
        --region $AWS_REGION
    
    print_success "RDS instance created successfully"
}

get_db_endpoint() {
    print_status "Getting RDS endpoint..."
    
    DB_ENDPOINT=$(aws rds describe-db-instances \
        --db-instance-identifier $DB_NAME \
        --region $AWS_REGION \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)
    
    print_status "Database endpoint: $DB_ENDPOINT"
}

create_ecs_cluster() {
    print_status "Creating ECS cluster..."
    
    # Check if cluster exists
    if aws ecs describe-clusters --clusters $CLUSTER_NAME --region $AWS_REGION | grep -q "ACTIVE"; then
        print_warning "ECS cluster $CLUSTER_NAME already exists"
    else
        aws ecs create-cluster \
            --cluster-name $CLUSTER_NAME \
            --capacity-providers FARGATE \
            --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
            --region $AWS_REGION
        print_success "ECS cluster created"
    fi
}

create_task_definition() {
    print_status "Creating ECS task definition..."
    
    # Create task definition JSON
    cat > task-definition.json << EOF
{
    "family": "$TASK_FAMILY",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "ai-chatbot-backend",
            "image": "$ECR_URI:$IMAGE_TAG",
            "portMappings": [
                {
                    "containerPort": 5000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "DB_HOST",
                    "value": "$DB_ENDPOINT"
                },
                {
                    "name": "DB_PORT",
                    "value": "5432"
                },
                {
                    "name": "DB_NAME",
                    "value": "$DB_NAME"
                },
                {
                    "name": "DB_USER",
                    "value": "$DB_USERNAME"
                },
                {
                    "name": "DB_PASSWORD",
                    "value": "$DB_PASSWORD"
                },
                {
                    "name": "FLASK_ENV",
                    "value": "production"
                },
                {
                    "name": "SECRET_KEY",
                    "value": "$(openssl rand -base64 32)"
                },
                {
                    "name": "JWT_SECRET_KEY",
                    "value": "$(openssl rand -base64 32)"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/$TASK_FAMILY",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:5000/api/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}
EOF
    
    # Create CloudWatch log group
    aws logs create-log-group \
        --log-group-name "/ecs/$TASK_FAMILY" \
        --region $AWS_REGION || true
    
    # Register task definition
    aws ecs register-task-definition \
        --cli-input-json file://task-definition.json \
        --region $AWS_REGION
    
    print_success "Task definition created"
    
    # Clean up
    rm task-definition.json
}

create_ecs_service() {
    print_status "Creating ECS service..."
    
    # Check if service exists
    if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION | grep -q "ACTIVE"; then
        print_warning "ECS service $SERVICE_NAME already exists, updating..."
        
        # Update service
        aws ecs update-service \
            --cluster $CLUSTER_NAME \
            --service $SERVICE_NAME \
            --task-definition $TASK_FAMILY \
            --region $AWS_REGION
    else
        # Create service
        aws ecs create-service \
            --cluster $CLUSTER_NAME \
            --service-name $SERVICE_NAME \
            --task-definition $TASK_FAMILY \
            --desired-count 1 \
            --launch-type FARGATE \
            --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
            --region $AWS_REGION
    fi
    
    print_success "ECS service created/updated"
}

create_load_balancer() {
    print_status "Creating Application Load Balancer..."
    
    # Create ALB
    ALB_ARN=$(aws elbv2 create-load-balancer \
        --name ${CLUSTER_NAME}-alb \
        --subnets $(echo $SUBNET_IDS | tr ',' ' ') \
        --security-groups $SECURITY_GROUP_ID \
        --region $AWS_REGION \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text 2>/dev/null || echo "exists")
    
    if [[ "$ALB_ARN" == "exists" ]]; then
        print_warning "Load balancer already exists"
        ALB_ARN=$(aws elbv2 describe-load-balancers \
            --names ${CLUSTER_NAME}-alb \
            --region $AWS_REGION \
            --query 'LoadBalancers[0].LoadBalancerArn' \
            --output text)
    fi
    
    # Create target group
    TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
        --name ${CLUSTER_NAME}-tg \
        --protocol HTTP \
        --port 5000 \
        --vpc-id $VPC_ID \
        --target-type ip \
        --health-check-path /api/health \
        --region $AWS_REGION \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text 2>/dev/null || echo "exists")
    
    if [[ "$TARGET_GROUP_ARN" == "exists" ]]; then
        TARGET_GROUP_ARN=$(aws elbv2 describe-target-groups \
            --names ${CLUSTER_NAME}-tg \
            --region $AWS_REGION \
            --query 'TargetGroups[0].TargetGroupArn' \
            --output text)
    fi
    
    # Create listener
    aws elbv2 create-listener \
        --load-balancer-arn $ALB_ARN \
        --protocol HTTP \
        --port 80 \
        --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
        --region $AWS_REGION 2>/dev/null || true
    
    # Update ECS service to use target group
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --load-balancers targetGroupArn=$TARGET_GROUP_ARN,containerName=ai-chatbot-backend,containerPort=5000 \
        --region $AWS_REGION
    
    # Get ALB DNS name
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --load-balancer-arns $ALB_ARN \
        --region $AWS_REGION \
        --query 'LoadBalancers[0].DNSName' \
        --output text)
    
    print_success "Load balancer created: http://$ALB_DNS"
}

wait_for_deployment() {
    print_status "Waiting for service to be stable..."
    
    aws ecs wait services-stable \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --region $AWS_REGION
    
    print_success "Service is stable and running"
}

print_deployment_info() {
    print_success "Deployment completed successfully!"
    echo ""
    echo "=== Deployment Information ==="
    echo "Region: $AWS_REGION"
    echo "Cluster: $CLUSTER_NAME"
    echo "Service: $SERVICE_NAME"
    echo "Database: $DB_ENDPOINT"
    echo "Load Balancer: http://$ALB_DNS"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Wait a few minutes for the service to fully start"
    echo "2. Access your application at: http://$ALB_DNS"
    echo "3. Check service logs: aws logs tail /ecs/$TASK_FAMILY --follow"
    echo "4. Monitor service: aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME"
    echo ""
    echo "=== Cleanup ==="
    echo "To delete all resources, run: ./cleanup.sh"
}

main() {
    print_status "Starting AWS deployment for Enhanced AI Chatbot..."
    
    check_prerequisites
    create_ecr_repository
    build_and_push_image
    create_rds_instance
    get_db_endpoint
    create_ecs_cluster
    create_task_definition
    create_ecs_service
    create_load_balancer
    wait_for_deployment
    print_deployment_info
}

# Run main function
main "$@"