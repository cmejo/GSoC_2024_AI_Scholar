#!/bin/bash

# AWS Cleanup Script for Enhanced AI Chatbot
# Removes all AWS resources created by the deployment script

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-west-2}
CLUSTER_NAME=${CLUSTER_NAME:-ai-chatbot-cluster}
SERVICE_NAME=${SERVICE_NAME:-ai-chatbot-service}
TASK_FAMILY=${TASK_FAMILY:-ai-chatbot-task}
ECR_REPOSITORY=${ECR_REPOSITORY:-ai-chatbot}
DB_NAME=${DB_NAME:-chatbot_db}

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

confirm_cleanup() {
    echo -e "${RED}WARNING: This will delete ALL AWS resources created for the AI Chatbot!${NC}"
    echo "This includes:"
    echo "- ECS Cluster and Service"
    echo "- RDS Database (with all data)"
    echo "- Load Balancer and Target Groups"
    echo "- ECR Repository (with all images)"
    echo "- CloudWatch Log Groups"
    echo ""
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirmation
    
    if [[ "$confirmation" != "yes" ]]; then
        print_status "Cleanup cancelled"
        exit 0
    fi
}

delete_ecs_service() {
    print_status "Deleting ECS service..."
    
    # Scale service to 0
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --desired-count 0 \
        --region $AWS_REGION 2>/dev/null || true
    
    # Wait for tasks to stop
    print_status "Waiting for tasks to stop..."
    aws ecs wait services-stable \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --region $AWS_REGION 2>/dev/null || true
    
    # Delete service
    aws ecs delete-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --region $AWS_REGION 2>/dev/null || true
    
    print_success "ECS service deleted"
}

delete_load_balancer() {
    print_status "Deleting Load Balancer..."
    
    # Get ALB ARN
    ALB_ARN=$(aws elbv2 describe-load-balancers \
        --names ${CLUSTER_NAME}-alb \
        --region $AWS_REGION \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text 2>/dev/null || echo "NotFound")
    
    if [[ "$ALB_ARN" != "NotFound" && "$ALB_ARN" != "None" ]]; then
        # Delete listeners
        LISTENER_ARNS=$(aws elbv2 describe-listeners \
            --load-balancer-arn $ALB_ARN \
            --region $AWS_REGION \
            --query 'Listeners[].ListenerArn' \
            --output text 2>/dev/null || echo "")
        
        for listener_arn in $LISTENER_ARNS; do
            aws elbv2 delete-listener \
                --listener-arn $listener_arn \
                --region $AWS_REGION 2>/dev/null || true
        done
        
        # Delete load balancer
        aws elbv2 delete-load-balancer \
            --load-balancer-arn $ALB_ARN \
            --region $AWS_REGION 2>/dev/null || true
    fi
    
    # Delete target group
    TARGET_GROUP_ARN=$(aws elbv2 describe-target-groups \
        --names ${CLUSTER_NAME}-tg \
        --region $AWS_REGION \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text 2>/dev/null || echo "NotFound")
    
    if [[ "$TARGET_GROUP_ARN" != "NotFound" && "$TARGET_GROUP_ARN" != "None" ]]; then
        aws elbv2 delete-target-group \
            --target-group-arn $TARGET_GROUP_ARN \
            --region $AWS_REGION 2>/dev/null || true
    fi
    
    print_success "Load Balancer deleted"
}

delete_ecs_cluster() {
    print_status "Deleting ECS cluster..."
    
    aws ecs delete-cluster \
        --cluster $CLUSTER_NAME \
        --region $AWS_REGION 2>/dev/null || true
    
    print_success "ECS cluster deleted"
}

delete_task_definitions() {
    print_status "Deregistering task definitions..."
    
    # Get all task definition revisions
    TASK_ARNS=$(aws ecs list-task-definitions \
        --family-prefix $TASK_FAMILY \
        --region $AWS_REGION \
        --query 'taskDefinitionArns[]' \
        --output text 2>/dev/null || echo "")
    
    for task_arn in $TASK_ARNS; do
        aws ecs deregister-task-definition \
            --task-definition $task_arn \
            --region $AWS_REGION 2>/dev/null || true
    done
    
    print_success "Task definitions deregistered"
}

delete_rds_instance() {
    print_status "Deleting RDS instance..."
    
    # Delete RDS instance
    aws rds delete-db-instance \
        --db-instance-identifier $DB_NAME \
        --skip-final-snapshot \
        --delete-automated-backups \
        --region $AWS_REGION 2>/dev/null || true
    
    # Wait for deletion (optional, can take a while)
    print_status "Waiting for RDS instance deletion (this may take several minutes)..."
    aws rds wait db-instance-deleted \
        --db-instance-identifier $DB_NAME \
        --region $AWS_REGION 2>/dev/null || true
    
    # Delete DB subnet group
    aws rds delete-db-subnet-group \
        --db-subnet-group-name ${DB_NAME}-subnet-group \
        --region $AWS_REGION 2>/dev/null || true
    
    print_success "RDS instance deleted"
}

delete_ecr_repository() {
    print_status "Deleting ECR repository..."
    
    aws ecr delete-repository \
        --repository-name $ECR_REPOSITORY \
        --force \
        --region $AWS_REGION 2>/dev/null || true
    
    print_success "ECR repository deleted"
}

delete_cloudwatch_logs() {
    print_status "Deleting CloudWatch log groups..."
    
    aws logs delete-log-group \
        --log-group-name "/ecs/$TASK_FAMILY" \
        --region $AWS_REGION 2>/dev/null || true
    
    print_success "CloudWatch log groups deleted"
}

main() {
    print_status "Starting AWS cleanup for Enhanced AI Chatbot..."
    
    confirm_cleanup
    
    delete_ecs_service
    delete_load_balancer
    delete_ecs_cluster
    delete_task_definitions
    delete_rds_instance
    delete_ecr_repository
    delete_cloudwatch_logs
    
    print_success "Cleanup completed successfully!"
    echo ""
    echo "All AWS resources have been deleted."
    echo "Note: Some resources (like RDS) may take additional time to fully delete."
}

# Run main function
main "$@"