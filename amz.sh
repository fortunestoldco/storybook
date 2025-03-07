#!/bin/bash

set -e

# Default configuration
DEFAULT_PROJECT_NAME="storybook"
DEFAULT_REGION="us-east-1"
CONFIG_FILE=".storybook_config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Define checkpoint files
CHECKPOINT_DIR=".storybook_checkpoints"
TABLES_CHECKPOINT="$CHECKPOINT_DIR/tables_created"
IAM_CHECKPOINT="$CHECKPOINT_DIR/iam_created"
LAMBDA_CHECKPOINT="$CHECKPOINT_DIR/lambda_created"
API_CHECKPOINT="$CHECKPOINT_DIR/api_created"
KB_CHECKPOINT="$CHECKPOINT_DIR/kb_created"
FLOW_CHECKPOINT="$CHECKPOINT_DIR/flow_created"
AGENT_CHECKPOINT="$CHECKPOINT_DIR/agent_created" # New checkpoint for agents

# Function to create checkpoint directory
setup_checkpoints() {
    mkdir -p "$CHECKPOINT_DIR"
}

# Function to check if a checkpoint exists
checkpoint_exists() {
    [ -f "$1" ]
}

# Function to create a checkpoint
create_checkpoint() {
    # Ensure checkpoint directory exists before creating a checkpoint file
    mkdir -p "$(dirname "$1")"
    touch "$1"
}

# Function to handle errors
handle_error() {
    local error_section=$1
    local error_msg=$2
    local fix_instructions=$3
    local console_url=$4

    echo -e "${RED}ERROR in ${error_section}: ${error_msg}${NC}"
    echo -e "${YELLOW}Suggested fix: ${fix_instructions}${NC}"

    if [ ! -z "$console_url" ]; then
        echo -e "${BLUE}AWS Console URL: ${NC}${console_url}"
    fi

    read -p "Continue with the rest of the deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Deployment aborted.${NC}"
        return 1
    fi
    return 0
}

# Load configuration if exists
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
    else
        PROJECT_NAME=$DEFAULT_PROJECT_NAME
        REGION=$DEFAULT_REGION
        CLAUDE_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"

        # Save default config
        save_config
    fi
}

# Save configuration
save_config() {
    cat > "$CONFIG_FILE" << EOF
# Storybook System Configuration
PROJECT_NAME="$PROJECT_NAME"
REGION="$REGION"
CLAUDE_MODEL="$CLAUDE_MODEL"
EOF
}

# Update configuration
update_config() {
    clear
    echo -e "${BLUE}=== Configuration Settings ===${NC}"
    echo -e "Current settings:"
    echo -e "1. Project Name: ${GREEN}$PROJECT_NAME${NC}"
    echo -e "2. AWS Region: ${GREEN}$REGION${NC}"
    echo -e "3. Foundation Model: ${GREEN}$CLAUDE_MODEL${NC}"
    echo -e "4. Return to main menu"
    echo

    read -p "Select a setting to change (1-4): " config_choice

    case $config_choice in
        1)
            read -p "Enter new project name: " new_project_name
            if [ ! -z "$new_project_name" ]; then
                PROJECT_NAME=$new_project_name
                save_config
                echo -e "${GREEN}Project name updated.${NC}"
            fi
            ;;
        2)
            echo "Available regions with Bedrock support:"
            echo "1. us-east-1 (N. Virginia)"
            echo "2. us-west-2 (Oregon)"
            echo "3. eu-central-1 (Frankfurt)"
            echo "4. ap-northeast-1 (Tokyo)"
            read -p "Select a region (1-4): " region_choice
            case $region_choice in
                1) REGION="us-east-1" ;;
                2) REGION="us-west-2" ;;
                3) REGION="eu-central-1" ;;
                4) REGION="ap-northeast-1" ;;
                *) echo -e "${RED}Invalid choice. Region not changed.${NC}" ;;
            esac
            save_config
            echo -e "${GREEN}Region updated.${NC}"
            ;;
        3)
            echo "Available Foundation Models:"
            echo "1. Claude 3 Sonnet (anthropic.claude-3-sonnet-20240229-v1:0)"
            echo "2. Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)"
            echo "3. Claude 3 Opus (anthropic.claude-3-opus-20240229-v1:0)"
            read -p "Select a model (1-3): " model_choice
            case $model_choice in
                1) CLAUDE_MODEL="anthropic.claude-3-sonnet-20240229-v1:0" ;;
                2) CLAUDE_MODEL="anthropic.claude-3-haiku-20240307-v1:0" ;;
                3) CLAUDE_MODEL="anthropic.claude-3-opus-20240229-v1:0" ;;
                *) echo -e "${RED}Invalid choice. Model not changed.${NC}" ;;
            esac
            save_config
            echo -e "${GREEN}Foundation model updated.${NC}"
            ;;
        4)
            return
            ;;
        *)
            echo -e "${RED}Invalid choice.${NC}"
            ;;
    esac

    read -p "Press Enter to continue..."
    update_config
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        if ! handle_error "Prerequisites" "AWS CLI is not installed" "Install AWS CLI from https://aws.amazon.com/cli/"; then
            return 1
        fi
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        if ! handle_error "Prerequisites" "jq is not installed" "Install jq using your package manager (e.g., apt install jq, brew install jq)"; then
            return 1
        fi
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        if ! handle_error "Prerequisites" "AWS credentials not configured" "Run 'aws configure' to set up your credentials" "https://console.aws.amazon.com/iam/home?#/security_credentials"; then
            return 1
        fi
    fi

    # Check for necessary AWS Bedrock model access
    echo -e "${BLUE}Checking model access...${NC}"
    MODEL_ACCESS=$(aws bedrock list-foundation-models --region $REGION 2>/dev/null || echo "")

    if [ -z "$MODEL_ACCESS" ]; then
        if ! handle_error "Bedrock Access" "Cannot access Bedrock API. You may not have Bedrock enabled in your account" "Enable Bedrock in the AWS Console and request model access" "https://console.aws.amazon.com/bedrock/home"; then
            return 1
        fi
    fi

    CLAUDE_ACCESS=$(echo $MODEL_ACCESS | grep -c "$CLAUDE_MODEL" || true)

    if [ $CLAUDE_ACCESS -eq 0 ]; then
        if ! handle_error "Bedrock Model Access" "Cannot find $CLAUDE_MODEL access in region $REGION" "Request access to the model in the AWS Bedrock Console" "https://console.aws.amazon.com/bedrock/home?#/models"; then
            # Continue anyway if the user chose to proceed
            echo -e "${YELLOW}Continuing without confirmed model access...${NC}"
        fi
    fi

    # Get ACCOUNT_ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

    return 0
}

# Deploy the storybook system
deploy_system() {
    if ! check_prerequisites; then
        echo -e "${RED}Prerequisites check failed. Cannot deploy.${NC}"
        return 1
    fi

    # Get account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

    echo -e "${BLUE}=== Deploying storybook system using AWS Bedrock ===${NC}"
    echo -e "${BLUE}Project: $PROJECT_NAME${NC}"
    echo -e "${BLUE}Region: $REGION${NC}"
    echo -e "${BLUE}Model: $CLAUDE_MODEL${NC}"

    # Create S3 bucket for assets
    BUCKET_NAME="${PROJECT_NAME}-assets-${ACCOUNT_ID}"
    echo -e "${BLUE}Creating S3 bucket $BUCKET_NAME for assets...${NC}"

    if ! aws s3api create-bucket \
        --bucket $BUCKET_NAME \
        --region $REGION \
        $(if [ "$REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=$REGION"; fi) \
        2>/dev/null; then

        # Check if bucket exists
        if aws s3api head-bucket --bucket $BUCKET_NAME 2>/dev/null; then
            echo -e "${YELLOW}Bucket $BUCKET_NAME already exists. Continuing...${NC}"
        else
            if ! handle_error "S3 Bucket Creation" "Failed to create bucket $BUCKET_NAME" "Check if the bucket name is already taken globally or if you have insufficient permissions" "https://console.aws.amazon.com/s3/home"; then
                return 1
            fi
        fi
    fi

    # Enable bucket versioning
    if ! aws s3api put-bucket-versioning \
        --bucket $BUCKET_NAME \
        --versioning-configuration Status=Enabled 2>/dev/null; then
        if ! handle_error "S3 Bucket Versioning" "Failed to enable versioning on bucket $BUCKET_NAME" "Check bucket permissions in the S3 console" "https://console.aws.amazon.com/s3/buckets/$BUCKET_NAME?tab=properties"; then
            return 1
        fi
    fi

    # Create DynamoDB tables for storing state
    echo -e "${BLUE}Creating DynamoDB tables for state management...${NC}"

    # Project state table
    if ! aws dynamodb create-table \
        --table-name storybook-projects \
        --attribute-definitions AttributeName=project_id,AttributeType=S \
        --key-schema AttributeName=project_id,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region $REGION \
        2>/dev/null; then

        # Check if table exists
        if aws dynamodb describe-table --table-name storybook-projects --region $REGION 2>/dev/null; then
            echo -e "${YELLOW}Projects table already exists. Continuing...${NC}"
        else
            if ! handle_error "DynamoDB Table Creation" "Failed to create storybook-projects table" "Check if you have sufficient permissions to create DynamoDB tables" "https://console.aws.amazon.com/dynamodb/home?region=$REGION#tables:"; then
                return 1
            fi
        fi
    fi

    # Chapter table
    if ! aws dynamodb create-table \
        --table-name storybook-chapters \
        --attribute-definitions \
            AttributeName=project_id,AttributeType=S \
            AttributeName=chapter_id,AttributeType=S \
        --key-schema \
            AttributeName=project_id,KeyType=HASH \
            AttributeName=chapter_id,KeyType=RANGE \
        --billing-mode PAY_PER_REQUEST \
        --region $REGION \
        2>/dev/null; then

        # Check if table exists
        if aws dynamodb describe-table --table-name storybook-chapters --region $REGION 2>/dev/null; then
            echo -e "${YELLOW}Chapters table already exists. Continuing...${NC}"
        else
            if ! handle_error "DynamoDB Table Creation" "Failed to create storybook-chapters table" "Check if you have sufficient permissions to create DynamoDB tables" "https://console.aws.amazon.com/dynamodb/home?region=$REGION#tables:"; then
                return 1
            fi
        fi
    fi

    # Character table
    if ! aws dynamodb create-table \
        --table-name storybook-characters \
        --attribute-definitions \
            AttributeName=project_id,AttributeType=S \
            AttributeName=character_id,AttributeType=S \
        --key-schema \
            AttributeName=project_id,KeyType=HASH \
            AttributeName=character_id,KeyType=RANGE \
        --billing-mode PAY_PER_REQUEST \
        --region $REGION \
        2>/dev/null; then

        # Check if table exists
        if aws dynamodb describe-table --table-name storybook-characters --region $REGION 2>/dev/null; then
            echo -e "${YELLOW}Characters table already exists. Continuing...${NC}"
        else
            if ! handle_error "DynamoDB Table Creation" "Failed to create storybook-characters table" "Check if you have sufficient permissions to create DynamoDB tables" "https://console.aws.amazon.com/dynamodb/home?region=$REGION#tables:"; then
                return 1
            fi
        fi
    fi

    # Agent table (new)
    if ! aws dynamodb create-table \
        --table-name storybook-agents \
        --attribute-definitions \
            AttributeName=agent_id,AttributeType=S \
        --key-schema \
            AttributeName=agent_id,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region $REGION \
        2>/dev/null; then

        # Check if table exists
        if aws dynamodb describe-table --table-name storybook-agents --region $REGION 2>/dev/null; then
            echo -e "${YELLOW}Agents table already exists. Continuing...${NC}"
        else
            if ! handle_error "DynamoDB Table Creation" "Failed to create storybook-agents table" "Check if you have sufficient permissions to create DynamoDB tables" "https://console.aws.amazon.com/dynamodb/home?region=$REGION#tables:"; then
                return 1
            fi
        fi
    fi

    # Create checkpoint
    create_checkpoint "$TABLES_CHECKPOINT"

    # Create IAM Flow execution role
    echo -e "${BLUE}Creating IAM roles for Bedrock Flows and Lambda...${NC}"
    FLOW_ROLE_NAME="${PROJECT_NAME}-flowrole"
    LAMBDA_ROLE_NAME="${PROJECT_NAME}-lambdarole"

    # Create a policy document for the flow role trust relationship
    cat > flow-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create a policy document for the Lambda trust relationship
    cat > lambda-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create the flow IAM role
    if ! aws iam create-role \
        --role-name $FLOW_ROLE_NAME \
        --assume-role-policy-document file://flow-trust-policy.json \
        2>/dev/null; then

        # Check if role exists
        if aws iam get-role --role-name $FLOW_ROLE_NAME 2>/dev/null; then
            echo -e "${YELLOW}Role $FLOW_ROLE_NAME already exists. Continuing...${NC}"
        else
            if ! handle_error "IAM Role Creation" "Failed to create role $FLOW_ROLE_NAME" "Check if you have sufficient permissions to create IAM roles" "https://console.aws.amazon.com/iam/home?#/roles"; then
                return 1
            fi
        fi
    fi

    # Create the Lambda execution role
    if ! aws iam create-role \
        --role-name $LAMBDA_ROLE_NAME \
        --assume-role-policy-document file://lambda-trust-policy.json \
        2>/dev/null; then

        # Check if role exists
        if aws iam get-role --role-name $LAMBDA_ROLE_NAME 2>/dev/null; then
            echo -e "${YELLOW}Role $LAMBDA_ROLE_NAME already exists. Continuing...${NC}"
        else
            if ! handle_error "IAM Role Creation" "Failed to create role $LAMBDA_ROLE_NAME" "Check if you have sufficient permissions to create IAM roles" "https://console.aws.amazon.com/iam/home?#/roles"; then
                return 1
            fi
        fi
    fi

    # Create Bedrock Flow policy with updated permissions for web crawling and agents
    cat > bedrock-flow-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateFlow",
                "bedrock:UpdateFlow",
                "bedrock:GetFlow",
                "bedrock:ListFlows", 
                "bedrock:DeleteFlow",
                "bedrock:ValidateFlowDefinition", 
                "bedrock:CreateFlowVersion",
                "bedrock:GetFlowVersion",
                "bedrock:ListFlowVersions",
                "bedrock:DeleteFlowVersion",
                "bedrock:CreateFlowAlias",
                "bedrock:UpdateFlowAlias",
                "bedrock:GetFlowAlias",
                "bedrock:ListFlowAliases",
                "bedrock:DeleteFlowAlias",
                "bedrock:InvokeFlow",
                "bedrock:TagResource",
                "bedrock:UntagResource", 
                "bedrock:ListTagsForResource"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:ApplyGuardrail",
                "bedrock:InvokeGuardrail",
                "bedrock:InvokeModel",
                "bedrock:GetCustomModel",
                "bedrock:InvokeAgent",
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate",
                "bedrock:GetPrompt",
                "bedrock:ListPrompts",
                "bedrock:RenderPrompt",
                "bedrock:WebSearch"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:GetAgent",
                "bedrock:GetKnowledgeBase",
                "bedrock:GetGuardrail",
                "bedrock:GetPrompt"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "lambda:InvokeFunction",
                "lambda:GetFunction"
            ],
            "Resource": "*"
        }
    ]
}
EOF

    # Create Lambda policy
    cat > lambda-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:GetItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Scan",
                "dynamodb:Query",
                "dynamodb:BatchWriteItem",
                "dynamodb:BatchGetItem"
            ],
            "Resource": [
                "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/storybook-projects",
                "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/storybook-chapters",
                "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/storybook-characters",
                "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/storybook-agents"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET_NAME}",
                "arn:aws:s3:::${BUCKET_NAME}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock-runtime:InvokeModel",
                "bedrock:WebSearch",
                "bedrock:InvokeAgent"
            ],
            "Resource": "*"
        }
    ]
}
EOF

    # Create or check for existing flow policy
    if ! aws iam create-policy \
        --policy-name ${PROJECT_NAME}-flow-policy \
        --policy-document file://bedrock-flow-policy.json \
        2>/dev/null; then

        # Check if policy exists
        FLOW_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-flow-policy"
        if aws iam get-policy --policy-arn $FLOW_POLICY_ARN 2>/dev/null; then
            echo -e "${YELLOW}Flow policy already exists. Continuing...${NC}"
        else
            if ! handle_error "IAM Policy Creation" "Failed to create flow policy" "Check if you have sufficient permissions to create IAM policies" "https://console.aws.amazon.com/iam/home?#/policies"; then
                return 1
            fi
        fi
    fi

    # Create or check for existing Lambda policy
    if ! aws iam create-policy \
        --policy-name ${PROJECT_NAME}-lambda-policy \
        --policy-document file://lambda-policy.json \
        2>/dev/null; then

        # Check if policy exists
        LAMBDA_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-lambda-policy"
        if aws iam get-policy --policy-arn $LAMBDA_POLICY_ARN 2>/dev/null; then
            echo -e "${YELLOW}Lambda policy already exists. Continuing...${NC}"
        else
            if ! handle_error "IAM Policy Creation" "Failed to create Lambda policy" "Check if you have sufficient permissions to create IAM policies" "https://console.aws.amazon.com/iam/home?#/policies"; then
                return 1
            fi
        fi
    fi

    # Attach policies to roles
    echo -e "${BLUE}Attaching policies to roles...${NC}"
    FLOW_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-flow-policy"
    LAMBDA_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-lambda-policy"

    # Attach policy to Flow role
    if ! aws iam attach-role-policy \
        --role-name $FLOW_ROLE_NAME \
        --policy-arn $FLOW_POLICY_ARN \
        2>/dev/null; then
        if ! handle_error "IAM Policy Attachment" "Failed to attach flow policy" "Check role permissions in IAM console" "https://console.aws.amazon.com/iam/home?#/roles/$FLOW_ROLE_NAME"; then
            return 1
        fi
    fi

    # Attach policies to Lambda role
    if ! aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn $LAMBDA_POLICY_ARN \
        2>/dev/null; then
        if ! handle_error "IAM Policy Attachment" "Failed to attach Lambda policy" "Check role permissions in IAM console" "https://console.aws.amazon.com/iam/home?#/roles/$LAMBDA_ROLE_NAME"; then
            return 1
        fi
    fi

    if ! aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
        2>/dev/null; then
        if ! handle_error "IAM Policy Attachment" "Failed to attach Lambda execution policy" "Check role permissions in IAM console" "https://console.aws.amazon.com/iam/home?#/roles/$LAMBDA_ROLE_NAME"; then
            return 1
        fi
    fi

    # Create a policy document for the Knowledge Base service role trust relationship
    cat > kb-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create the Knowledge Base service role
    KB_ROLE_NAME="${PROJECT_NAME}-kbrole"
    if ! aws iam create-role \
        --role-name $KB_ROLE_NAME \
        --assume-role-policy-document file://kb-trust-policy.json \
        2>/dev/null; then

        # Check if role exists
        if aws iam get-role --role-name $KB_ROLE_NAME 2>/dev/null; then
            echo -e "${YELLOW}Knowledge Base role already exists. Continuing...${NC}"
        else
            if ! handle_error "IAM Role Creation" "Failed to create Knowledge Base role $KB_ROLE_NAME" "Check if you have sufficient permissions to create IAM roles" "https://console.aws.amazon.com/iam/home?#/roles"; then
                return 1
            fi
        fi
    fi

    # Attach policies for Knowledge Base access to S3
    if ! aws iam attach-role-policy \
        --role-name $KB_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
        2>/dev/null; then
        if ! handle_error "IAM Policy Attachment" "Failed to attach S3 read policy to Knowledge Base role" "Check role permissions in IAM console" "https://console.aws.amazon.com/iam/home?#/roles/$KB_ROLE_NAME"; then
            return 1
        fi
    fi

    # Create web crawling policy for Knowledge Base
    cat > kb-web-crawling-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:WebSearch"
            ],
            "Resource": "*"
        }
    ]
}
EOF

    # Create or check for existing web crawling policy
    if ! aws iam create-policy \
        --policy-name ${PROJECT_NAME}-kb-webcrawl-policy \
        --policy-document file://kb-web-crawling-policy.json \
        2>/dev/null; then

        # Check if policy exists
        KB_WEBCRAWL_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-kb-webcrawl-policy"
        if aws iam get-policy --policy-arn $KB_WEBCRAWL_POLICY_ARN 2>/dev/null; then
            echo -e "${YELLOW}KB web crawling policy already exists. Continuing...${NC}"
        else
            if ! handle_error "IAM Policy Creation" "Failed to create KB web crawling policy" "Check if you have sufficient permissions to create IAM policies" "https://console.aws.amazon.com/iam/home?#/policies"; then
                return 1
            fi
        fi
    fi

    # Attach web crawling policy to KB role
    KB_WEBCRAWL_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-kb-webcrawl-policy"
    if ! aws iam attach-role-policy \
        --role-name $KB_ROLE_NAME \
        --policy-arn $KB_WEBCRAWL_POLICY_ARN \
        2>/dev/null; then
        if ! handle_error "IAM Policy Attachment" "Failed to attach web crawling policy to Knowledge Base role" "Check role permissions in IAM console" "https://console.aws.amazon.com/iam/home?#/roles/$KB_ROLE_NAME"; then
            return 1
        fi
    fi

    # Create checkpoint
    create_checkpoint "$IAM_CHECKPOINT"

    # Wait for IAM role to propagate
    echo -e "${BLUE}Waiting for IAM roles to propagate...${NC}"
    # Increase the delay to ensure IAM roles have time to propagate
    sleep 45

    # Create Lambda functions for the orchestration and data processing
    echo -e "${BLUE}Creating Lambda functions...${NC}"
    mkdir -p lambda

    # Create orchestrator Lambda function with Web Search capabilities
    cat > lambda/orchestrator.py << 'EOF'
import json
import boto3
import os
import uuid
from datetime import datetime

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('REGION', 'us-east-1'))
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('REGION', 'us-east-1'))
bedrock_client = boto3.client('bedrock', region_name=os.environ.get('REGION', 'us-east-1'))

# Define tables
projects_table = dynamodb.Table('storybook-projects')
chapters_table = dynamodb.Table('storybook-chapters')
characters_table = dynamodb.Table('storybook-characters')
agents_table = dynamodb.Table('storybook-agents')

# Environment variables
FLOW_ID = os.environ.get('FLOW_ID', '')
FLOW_ALIAS = os.environ.get('FLOW_ALIAS', 'latest')
CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
AGENT_ID = os.environ.get('AGENT_ID', '')
AGENT_ALIAS_ID = os.environ.get('AGENT_ALIAS_ID', 'latest')

def lambda_handler(event, context):
    """Main handler for orchestrating the storybook system."""
    
    # Parse input
    body = event
    if 'body' in event:
        if isinstance(event['body'], str):
            body = json.loads(event['body'])
        else:
            body = event['body']

    # Determine operation to perform
    operation = body.get('operation', 'default')

    if operation == 'create_project':
        return create_new_project(body)
    elif operation == 'invoke_flow':
        return invoke_flow(body)
    elif operation == 'invoke_agent':
        return invoke_agent(body)
    elif operation == 'transition_phase':
        return transition_phase(body)
    elif operation == 'store_chapter':
        return store_chapter(body)
    elif operation == 'store_character':
        return store_character(body)
    elif operation == 'retrieve_project':
        return retrieve_project(body)
    elif operation == 'web_search':
        return perform_web_search(body)
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Invalid operation: {operation}'})
        }

def perform_web_search(body):
    """Performs a web search using Bedrock's web search capability."""
    query = body.get('query', '')
    max_results = body.get('max_results', 5)
    
    if not query:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameter: query'})
        }
    
    try:
        # Use Bedrock's web search capability
        response = bedrock_runtime.invoke_model(
            modelId=CLAUDE_MODEL,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please search the web for: {query}\nReturn only the most relevant information in a concise format. Limit to {max_results} results."
                            }
                        ]
                    }
                ]
            })
        )
        
        # Parse and format the response
        response_body = json.loads(response['body'].read())
        search_results = response_body['content'][0]['text']
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'query': query,
                'results': search_results
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Error performing web search: {str(e)}'
            })
        }

def invoke_agent(body):
    """Invoke a Bedrock agent for processing."""
    
    agent_id = body.get('agent_id', AGENT_ID)
    agent_alias_id = body.get('agent_alias_id', AGENT_ALIAS_ID)
    input_text = body.get('input_text', '')
    session_id = body.get('session_id', str(uuid.uuid4()))
    
    if not agent_id or not input_text:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameters: agent_id and input_text'})
        }
    
    try:
        # Invoke Bedrock agent
        response = bedrock_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=input_text
        )
        
        # Extract agent response
        agent_response = ""
        for event in response.get('completion', []):
            if 'chunk' in event:
                agent_response += event['chunk']['bytes'].decode('utf-8')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'agent_id': agent_id,
                'session_id': session_id,
                'response': agent_response
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Error invoking agent: {str(e)}'
            })
        }

def create_new_project(body):
    """Create a new novel project."""

    project_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Extract project data
    title = body.get('title', 'Untitled Novel')
    synopsis = body.get('synopsis', '')
    genre = body.get('genre', ['fiction'])
    target_audience = body.get('target_audience', ['adult'])

    # Create project record
    project = {
        'project_id': project_id,
        'title': title,
        'synopsis': synopsis,
        'genre': genre,
        'target_audience': target_audience,
        'phase': 'initialization',
        'status': 'active',
        'created_at': timestamp,
        'updated_at': timestamp,
        'quality_assessment': {}
    }

    # Store in DynamoDB
    projects_table.put_item(Item=project)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'message': 'Project created successfully',
            'project': project
        })
    }

def invoke_flow(body):
    """Invoke a flow to process a task."""
    
    project_id = body.get('project_id')
    task = body.get('task', '')
    flow_inputs = body.get('flow_inputs', {})
    
    if not project_id:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameter: project_id'})
        }
    
    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Project not found'})
        }
    
    project = project_response['Item']
    
    # Prepare input for flow
    if not flow_inputs:
        flow_inputs = {
            'project_id': project_id,
            'project_title': project.get('title', ''),
            'project_synopsis': project.get('synopsis', ''),
            'project_phase': project.get('phase', 'initialization'),
            'task': task
        }
    
    # Determine which node to target based on task or phase
    input_node = determine_flow_node(project.get('phase', 'initialization'), task)
    
    try:
        # Convert flow_inputs to the proper format for flow invocation
        formatted_inputs = [{
            "content": flow_inputs,
            "nodeName": input_node,
            "nodeOutputName": "document"
        }]
        
        # Invoke the Bedrock Flow
        response = bedrock_client.invoke_flow(
            flowIdentifier=FLOW_ID,
            flowAliasIdentifier=FLOW_ALIAS,
            inputs=formatted_inputs
        )
        
        # Process response stream
        flow_response = ""
        for event in response.get('responseStream', []):
            chunk = json.loads(event['bytes'].decode('utf-8'))
            if 'content' in chunk:
                flow_response += chunk['content']
        
        # Update project with flow execution results
        timestamp = datetime.now().isoformat()
        
        if 'flow_outputs' not in project:
            project['flow_outputs'] = []
        
        project['flow_outputs'].append({
            'timestamp': timestamp,
            'task': task,
            'input': flow_inputs,
            'response': flow_response
        })
        
        project['updated_at'] = timestamp
        project['last_task'] = task
        
        # Save updated project
        projects_table.put_item(Item=project)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'project_id': project_id,
                'task': task,
                'response': flow_response
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def determine_flow_node(phase, task):
    """Determine which flow node to use based on the phase and task."""
    # This is a simplified mapping
    if 'initialization' in phase:
        return 'InitializationInput'
    elif 'development' in phase:
        return 'DevelopmentInput'
    elif 'creation' in phase:
        return 'CreationInput'
    elif 'refinement' in phase:
        return 'RefinementInput'
    elif 'finalization' in phase:
        return 'FinalizationInput'
    else:
        return 'FlowInputNode'  # Default input node

def transition_phase(body):
    """Transition a project to a new phase."""

    project_id = body.get('project_id')
    new_phase = body.get('new_phase')

    valid_phases = ['initialization', 'development', 'creation', 'refinement', 'finalization', 'complete']

    if not project_id or new_phase not in valid_phases:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid project_id or phase'})
        }

    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Project not found'})
        }

    project = project_response['Item']
    current_phase = project['phase']
    timestamp = datetime.now().isoformat()

    # Check if transition is valid
    phase_order = {phase: idx for idx, phase in enumerate(valid_phases)}
    if phase_order.get(new_phase, 99) <= phase_order.get(current_phase, 0):
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid phase transition. Cannot move backward.'})
        }

    # Update phase history
    if 'phase_history' not in project:
        project['phase_history'] = {}

    if current_phase not in project['phase_history']:
        project['phase_history'][current_phase] = []

    project['phase_history'][current_phase].append({
        'end_time': timestamp,
        'transition_to': new_phase,
        'quality_assessment': project.get('quality_assessment', {})
    })

    # Update project phase
    project['phase'] = new_phase
    project['updated_at'] = timestamp

    # Save updated project
    projects_table.put_item(Item=project)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'previous_phase': current_phase,
            'current_phase': new_phase,
            'message': f'Project transitioned from {current_phase} to {new_phase}'
        })
    }

def store_chapter(body):
    """Store a chapter in the database."""
    project_id = body.get('project_id')
    chapter_id = body.get('chapter_id', str(uuid.uuid4()))
    title = body.get('title')
    content = body.get('content')

    if not project_id or not title or not content:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameters'})
        }

    # Create chapter record
    timestamp = datetime.now().isoformat()
    chapter = {
        'project_id': project_id,
        'chapter_id': chapter_id,
        'title': title,
        'content': content,
        'created_at': timestamp,
        'updated_at': timestamp
    }

    # Store in DynamoDB
    chapters_table.put_item(Item=chapter)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'chapter_id': chapter_id,
            'message': 'Chapter stored successfully'
        })
    }

def store_character(body):
    """Store a character in the database."""
    project_id = body.get('project_id')
    character_id = body.get('character_id', str(uuid.uuid4()))
    name = body.get('name')
    description = body.get('description', '')
    role = body.get('role', '')

    if not project_id or not name:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameters'})
        }

    # Create character record
    timestamp = datetime.now().isoformat()
    character = {
        'project_id': project_id,
        'character_id': character_id,
        'name': name,
        'description': description,
        'role': role,
        'created_at': timestamp,
        'updated_at': timestamp
    }

    # Store in DynamoDB
    characters_table.put_item(Item=character)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'character_id': character_id,
            'message': 'Character stored successfully'
        })
    }

def retrieve_project(body):
    """Retrieve a project from the database."""
    project_id = body.get('project_id')

    if not project_id:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing project_id parameter'})
        }

    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Project not found'})
        }

    project = project_response['Item']

    # Get chapters
    chapter_response = chapters_table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('project_id').eq(project_id)
    )
    chapters = chapter_response.get('Items', [])

    # Get characters
    character_response = characters_table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('project_id').eq(project_id)
    )
    characters = character_response.get('Items', [])

    # Combine data
    project_data = {
        'project': project,
        'chapters': chapters,
        'characters': characters
    }

    return {
        'statusCode': 200,
        'body': json.dumps(project_data)
    }
EOF

    # Create data processor Lambda function with web search and agent support
    cat > lambda/data_processor.py << 'EOF'
import json
import boto3
import os
import uuid
from datetime import datetime

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('REGION', 'us-east-1'))
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('REGION', 'us-east-1'))
bedrock_client = boto3.client('bedrock', region_name=os.environ.get('REGION', 'us-east-1'))

# Define tables
projects_table = dynamodb.Table('storybook-projects')
chapters_table = dynamodb.Table('storybook-chapters')
characters_table = dynamodb.Table('storybook-characters')
agents_table = dynamodb.Table('storybook-agents')

def lambda_handler(event, context):
    """Handler for data processing operations within flows."""
    
    # Determine which operation to perform
    operation = event.get('operation', '')
    
    if operation == 'store_project_data':
        return store_project_data(event)
    elif operation == 'retrieve_project_data':
        return retrieve_project_data(event)
    elif operation == 'create_character':
        return create_character(event)
    elif operation == 'create_chapter':
        return create_chapter(event)
    elif operation == 'update_project_quality':
        return update_project_quality(event)
    elif operation == 'web_search':
        return web_search(event)
    elif operation == 'create_agent':
        return create_agent(event)
    elif operation == 'invoke_agent':
        return invoke_agent(event)
    else:
        return {
            'statusCode': 400,
            'message': f'Invalid operation: {operation}',
            'output': {}
        }

def store_project_data(event):
    """Store or update project data in DynamoDB."""
    
    project_id = event.get('project_id')
    data = event.get('data', {})
    
    if not project_id:
        return {
            'statusCode': 400,
            'message': 'Missing project_id parameter',
            'output': {}
        }
    
    # Get existing project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    
    if 'Item' in project_response:
        # Update existing project
        project = project_response['Item']
        project.update(data)
        project['updated_at'] = datetime.now().isoformat()
    else:
        # Create new project if it doesn't exist
        project = {
            'project_id': project_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'status': 'active',
            'phase': 'initialization'
        }
        project.update(data)
    
    # Store updated project
    projects_table.put_item(Item=project)
    
    return {
        'statusCode': 200,
        'message': 'Project data stored successfully',
        'output': {'project_id': project_id}
    }

def retrieve_project_data(event):
    """Retrieve project data from DynamoDB."""
    
    project_id = event.get('project_id')
    include_chapters = event.get('include_chapters', False)
    include_characters = event.get('include_characters', False)
    
    if not project_id:
        return {
            'statusCode': 400,
            'message': 'Missing project_id parameter',
            'output': {}
        }
    
    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'message': 'Project not found',
            'output': {}
        }
    
    project = project_response['Item']
    result = {'project': project}
    
    # Get chapters if requested
    if include_chapters:
        chapter_response = chapters_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('project_id').eq(project_id)
        )
        result['chapters'] = chapter_response.get('Items', [])
    
    # Get characters if requested
    if include_characters:
        character_response = characters_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('project_id').eq(project_id)
        )
        result['characters'] = character_response.get('Items', [])
    
    return {
        'statusCode': 200,
        'message': 'Project data retrieved successfully',
        'output': result
    }

def create_character(event):
    """Create a character in the database."""
    
    project_id = event.get('project_id')
    name = event.get('name')
    description = event.get('description', '')
    role = event.get('role', '')
    character_id = event.get('character_id', str(uuid.uuid4()))
    
    if not project_id or not name:
        return {
            'statusCode': 400,
            'message': 'Missing required parameters: project_id and name',
            'output': {}
        }
    
    # Create character record
    timestamp = datetime.now().isoformat()
    character = {
        'project_id': project_id,
        'character_id': character_id,
        'name': name,
        'description': description,
        'role': role,
        'created_at': timestamp,
        'updated_at': timestamp
    }
    
    # Store in DynamoDB
    characters_table.put_item(Item=character)
    
    return {
        'statusCode': 200,
        'message': 'Character created successfully',
        'output': {
            'project_id': project_id,
            'character_id': character_id,
            'name': name
        }
    }

def create_chapter(event):
    """Create a chapter in the database."""
    
    project_id = event.get('project_id')
    title = event.get('title')
    content = event.get('content', '')
    chapter_id = event.get('chapter_id', str(uuid.uuid4()))
    
    if not project_id or not title:
        return {
            'statusCode': 400,
            'message': 'Missing required parameters: project_id and title',
            'output': {}
        }
    
    # Create chapter record
    timestamp = datetime.now().isoformat()
    chapter = {
        'project_id': project_id,
        'chapter_id': chapter_id,
        'title': title,
        'content': content,
        'created_at': timestamp,
        'updated_at': timestamp
    }
    
    # Store in DynamoDB
    chapters_table.put_item(Item=chapter)
    
    return {
        'statusCode': 200,
        'message': 'Chapter created successfully',
        'output': {
            'project_id': project_id,
            'chapter_id': chapter_id,
            'title': title
        }
    }

def update_project_quality(event):
    """Update project quality assessment data."""
    
    project_id = event.get('project_id')
    quality_data = event.get('quality_data', {})
    
    if not project_id or not quality_data:
        return {
            'statusCode': 400,
            'message': 'Missing required parameters: project_id and quality_data',
            'output': {}
        }
    
    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'message': 'Project not found',
            'output': {}
        }
    
    project = project_response['Item']
    
    # Update quality assessment data
    if 'quality_assessment' not in project:
        project['quality_assessment'] = {}
    
    project['quality_assessment'].update(quality_data)
    project['updated_at'] = datetime.now().isoformat()
    
    # Store updated project
    projects_table.put_item(Item=project)
    
    return {
        'statusCode': 200,
        'message': 'Quality assessment updated successfully',
        'output': {
            'project_id': project_id,
            'quality_assessment': project['quality_assessment']
        }
    }

def web_search(event):
    """Perform a web search using Bedrock's web search capability."""
    
    query = event.get('query')
    max_results = event.get('max_results', 5)
    
    if not query:
        return {
            'statusCode': 400,
            'message': 'Missing required parameter: query',
            'output': {}
        }
    
    try:
        # Invoke Bedrock model with web search capabilities
        response = bedrock_runtime.invoke_model(
            modelId=os.environ.get('CLAUDE_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please search the web for information about: {query}. Return the top {max_results} relevant results in a clear, structured format."
                            }
                        ]
                    }
                ]
            })
        )
        
        # Parse and format the response
        response_body = json.loads(response['body'].read())
        search_results = response_body['content'][0]['text']
        
        return {
            'statusCode': 200,
            'message': 'Web search completed successfully',
            'output': {
                'query': query,
                'results': search_results
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'message': f'Error performing web search: {str(e)}',
            'output': {}
        }

def create_agent(event):
    """Create an agent record in the database."""
    
    agent_name = event.get('agent_name')
    agent_role = event.get('agent_role')
    agent_instructions = event.get('agent_instructions')
    agent_id = event.get('agent_id', str(uuid.uuid4()))
    
    if not agent_name or not agent_role:
        return {
            'statusCode': 400,
            'message': 'Missing required parameters: agent_name and agent_role',
            'output': {}
        }
    
    # Create agent record
    timestamp = datetime.now().isoformat()
    agent = {
        'agent_id': agent_id,
        'name': agent_name,
        'role': agent_role,
        'instructions': agent_instructions,
        'created_at': timestamp,
        'updated_at': timestamp
    }
    
    # Store in DynamoDB
    agents_table.put_item(Item=agent)
    
    return {
        'statusCode': 200,
        'message': 'Agent created successfully',
        'output': {
            'agent_id': agent_id,
            'name': agent_name
        }
    }

def invoke_agent(event):
    """Invoke a Bedrock agent for processing."""
    
    agent_id = event.get('agent_id')
    agent_alias_id = event.get('agent_alias_id', 'latest')
    input_text = event.get('input_text')
    session_id = event.get('session_id', str(uuid.uuid4()))
    
    if not agent_id or not input_text:
        return {
            'statusCode': 400,
            'message': 'Missing required parameters: agent_id and input_text',
            'output': {}
        }
    
    try:
        # Invoke Bedrock agent
        response = bedrock_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=input_text
        )
        
        # Extract agent response
        agent_response = ""
        for event in response.get('completion', []):
            if 'chunk' in event:
                agent_response += event['chunk']['bytes'].decode('utf-8')
        
        return {
            'statusCode': 200,
            'message': 'Agent invoked successfully',
            'output': {
                'agent_id': agent_id,
                'session_id': session_id,
                'response': agent_response
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'message': f'Error invoking agent: {str(e)}',
            'output': {}
        }
EOF

    # Zip the Lambda functions
    cd lambda
    zip -r orchestrator.zip orchestrator.py
    zip -r data_processor.zip data_processor.py
    cd ..

    # Get the IAM role ARNs
    LAMBDA_ROLE_ARN=$(aws iam get-role --role-name ${LAMBDA_ROLE_NAME} --query 'Role.Arn' --output text 2>/dev/null)
    FLOW_ROLE_ARN=$(aws iam get-role --role-name ${FLOW_ROLE_NAME} --query 'Role.Arn' --output text 2>/dev/null)
    KB_ROLE_ARN=$(aws iam get-role --role-name ${KB_ROLE_NAME} --query 'Role.Arn' --output text 2>/dev/null)

    # Verify that we have valid role ARNs before proceeding
    if [ -z "$LAMBDA_ROLE_ARN" ]; then
        if ! handle_error "Lambda Creation" "Could not retrieve Lambda role ARN" "Check if the IAM role was created correctly in the IAM console" "https://console.aws.amazon.com/iam/home?#/roles"; then
            return 1
        fi
    fi

    if [ -z "$FLOW_ROLE_ARN" ]; then
        if ! handle_error "Flow Creation" "Could not retrieve Flow role ARN" "Check if the IAM role was created correctly in the IAM console" "https://console.aws.amazon.com/iam/home?#/roles"; then
            return 1
        fi
    fi

    echo -e "${BLUE}Using Lambda IAM role: $LAMBDA_ROLE_ARN${NC}"
    echo -e "${BLUE}Using Flow IAM role: $FLOW_ROLE_ARN${NC}"
    if [ ! -z "$KB_ROLE_ARN" ]; then
        echo -e "${BLUE}Using Knowledge Base role: $KB_ROLE_ARN${NC}"
    else
        echo -e "${YELLOW}Knowledge Base role not found. Knowledge Base features may not work correctly.${NC}"
    fi

    # Create Lambda functions
    echo -e "${BLUE}Creating Lambda functions...${NC}"

    # Create orchestrator Lambda
    ORCHESTRATOR_ARN=$(aws lambda create-function \
        --function-name ${PROJECT_NAME}-orchestrator \
        --runtime python3.10 \
        --role $LAMBDA_ROLE_ARN \
        --handler orchestrator.lambda_handler \
        --zip-file fileb://lambda/orchestrator.zip \
        --timeout 300 \
        --environment "Variables={REGION=$REGION,CLAUDE_MODEL=$CLAUDE_MODEL}" \
        --region $REGION \
        --query 'FunctionArn' --output text \
        2>/dev/null || \
        aws lambda update-function-code \
           --function-name ${PROJECT_NAME}-orchestrator \
           --zip-file fileb://lambda/orchestrator.zip \
           --region $REGION \
           --query 'FunctionArn' --output text \
           2>/dev/null)

    if [ -z "$ORCHESTRATOR_ARN" ]; then
        if ! handle_error "Lambda Creation" "Failed to create orchestrator Lambda function" "Check Lambda service permissions in the AWS console" "https://console.aws.amazon.com/lambda/home?region=$REGION#/functions"; then
            return 1
        fi
    else
        echo -e "${GREEN}Lambda orchestrator function created: $ORCHESTRATOR_ARN${NC}"
    fi

    # Create data processor Lambda
    DATA_PROCESSOR_ARN=$(aws lambda create-function \
        --function-name ${PROJECT_NAME}-data-processor \
        --runtime python3.10 \
        --role $LAMBDA_ROLE_ARN \
        --handler data_processor.lambda_handler \
        --zip-file fileb://lambda/data_processor.zip \
        --timeout 300 \
        --environment "Variables={REGION=$REGION,CLAUDE_MODEL=$CLAUDE_MODEL}" \
        --region $REGION \
        --query 'FunctionArn' --output text \
        2>/dev/null || \
        aws lambda update-function-code \
           --function-name ${PROJECT_NAME}-data-processor \
           --zip-file fileb://lambda/data_processor.zip \
           --region $REGION \
           --query 'FunctionArn' --output text \
           2>/dev/null)

    if [ -z "$DATA_PROCESSOR_ARN" ]; then
        if ! handle_error "Lambda Creation" "Failed to create data processor Lambda function" "Check Lambda service permissions in the AWS console" "https://console.aws.amazon.com/lambda/home?region=$REGION#/functions"; then
            return 1
        fi
    else
        echo -e "${GREEN}Lambda data processor function created: $DATA_PROCESSOR_ARN${NC}"
        create_checkpoint "$LAMBDA_CHECKPOINT"
    fi

    # Create a Knowledge Base for project reference materials
    echo -e "${BLUE}Creating Amazon Bedrock Knowledge Base...${NC}"

    # Create a directory for knowledge base files
    mkdir -p kb_files
    cat > kb_files/novel_writing_guide.txt << EOF
# Novel Writing Guide

## Structure
- Three-act structure: setup, confrontation, resolution
- Five-act structure: exposition, rising action, climax, falling action, resolution
- Hero's journey: ordinary world, call to adventure, refusal, meeting the mentor, crossing the threshold, tests/allies/enemies, approach, ordeal, reward, road back, resurrection, return

## Character Development
- Give characters clear motivations
- Create internal and external conflicts
- Develop character arcs showing growth or change
- Use character backstory to inform decisions

## Plot Development
- Create a compelling inciting incident
- Build rising tension throughout
- Plant setups and payoffs
- Create meaningful stakes
- Design a satisfying climax and resolution

## Genre Conventions
- Romance: Meet cute, obstacles, dark moment, resolution
- Mystery: Crime, investigation, red herrings, resolution
- Fantasy: World-building, magic systems, epic stakes
- Thriller: Ticking clock, high stakes, twists
EOF

    # Upload knowledge base files to S3
    if ! aws s3 cp kb_files/novel_writing_guide.txt s3://$BUCKET_NAME/kb_files/novel_writing_guide.txt 2>/dev/null; then
        if ! handle_error "S3 Upload" "Failed to upload knowledge base files to S3" "Check S3 bucket permissions in the AWS console" "https://console.aws.amazon.com/s3/buckets/$BUCKET_NAME?region=$REGION&tab=permissions"; then
            return 1
        fi
    fi

    # Create Knowledge Base
    KB_SUPPORTED=0
    aws bedrock-agent help create-knowledge-base >/dev/null 2>&1 || KB_SUPPORTED=1

    KB_ID=""
    if [ $KB_SUPPORTED -eq 0 ]; then
        echo -e "${BLUE}Creating Knowledge Base...${NC}"

        # Try to create encryption policy - this may fail if OpenSearch Serverless isn't available
        if ! aws opensearchserverless create-security-policy \
            --name "${PROJECT_NAME}-encryption-policy" \
            --type encryption \
            --policy "{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${PROJECT_NAME}-vectors\"],\"KmsKeyId\":\"aws/opensearch\"}],\"AWSOwnedKey\":true}" \
            --region $REGION \
            2>/dev/null; then

            echo -e "${YELLOW}Could not create OpenSearch encryption policy. Knowledge Base features may not work correctly.${NC}"
            echo -e "${YELLOW}This may be because OpenSearch Serverless is not available in your region or account.${NC}"
        fi

        # Try to create network policy
        if ! aws opensearchserverless create-security-policy \
            --name "${PROJECT_NAME}-network-policy" \
            --type network \
            --policy "{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${PROJECT_NAME}-vectors\"],\"AllowFromPublic\":true,\"SourceVPCEs\":[]}],\"Description\":\"Network policy for vector collection\"}" \
            --region $REGION \
            2>/dev/null; then

            echo -e "${YELLOW}Could not create OpenSearch network policy. Knowledge Base features may not work correctly.${NC}"
        fi

        # Try to create collection
        COLLECTION_ID=$(aws opensearchserverless create-collection \
            --name "${PROJECT_NAME}-vectors" \
            --type "VECTORSEARCH" \
            --region $REGION \
            --query 'createCollectionDetail.id' --output text \
            2>/dev/null || echo "")

        if [ ! -z "$COLLECTION_ID" ]; then
            echo -e "${GREEN}Created vector collection: $COLLECTION_ID${NC}"
            
            # Wait for collection to be active
            echo -e "${BLUE}Waiting for vector collection to be active...${NC}"
            
            IS_ACTIVE=false
            RETRY_COUNT=0
            MAX_RETRIES=12 # 6 minutes total wait time
            
            while [ "$IS_ACTIVE" = false ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
                STATUS=$(aws opensearchserverless get-collection \
                    --id $COLLECTION_ID \
                    --region $REGION \
                    --query 'collectionDetail.status' --output text \
                    2>/dev/null || echo "FAILED")
                
                if [ "$STATUS" = "ACTIVE" ]; then
                    IS_ACTIVE=true
                    echo -e "${GREEN}Vector collection is now active.${NC}"
                else
                    echo -e "${YELLOW}Collection status: $STATUS. Waiting... (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)${NC}"
                    RETRY_COUNT=$((RETRY_COUNT+1))
                    sleep 30
                fi
            done
            
            if [ "$IS_ACTIVE" = false ]; then
                echo -e "${YELLOW}Vector collection did not become active within the waiting period. Continuing without Knowledge Base.${NC}"
            else
                # Get collection endpoint
                COLLECTION_ENDPOINT=$(aws opensearchserverless get-collection \
                    --id $COLLECTION_ID \
                    --region $REGION \
                    --query 'collectionDetail.collectionEndpoint' --output text \
                    2>/dev/null || echo "")
                
                if [ ! -z "$COLLECTION_ENDPOINT" ]; then
                    # Create the Knowledge Base
                    KB_ID=$(aws bedrock-agent create-knowledge-base \
                        --name "${PROJECT_NAME}-knowledge-base" \
                        --role-arn "$KB_ROLE_ARN" \
                        --knowledge-base-configuration "{\"type\":\"VECTOR\",\"vectorKnowledgeBaseConfiguration\":{\"embeddingModelArn\":\"arn:aws:bedrock:$REGION::foundation-model/amazon.titan-embed-text-v1\"}}" \
                        --storage-configuration "{\"type\":\"OPENSEARCH_SERVERLESS\",\"opensearchServerlessConfiguration\":{\"collectionArn\":\"arn:aws:aoss:$REGION:$ACCOUNT_ID:collection/$COLLECTION_ID\",\"vectorIndexName\":\"bedrock-kb-index\",\"fieldMapping\":{\"textField\":\"text\",\"metadataField\":\"metadata\",\"vectorField\":\"vector\"}}}" \
                        --region $REGION \
                        --query 'knowledgeBase.knowledgeBaseId' --output text \
                        2>/dev/null || echo "")
                    
                    if [ ! -z "$KB_ID" ]; then
                        echo -e "${GREEN}Created Knowledge Base: $KB_ID${NC}"
                        
                        # Create data source
                        DS_ID=$(aws bedrock-agent create-data-source \
                            --knowledge-base-id $KB_ID \
                            --name "${PROJECT_NAME}-guide" \
                            --data-source-configuration "{\"type\":\"S3\",\"s3Configuration\":{\"bucketArn\":\"arn:aws:s3:::$BUCKET_NAME\",\"inclusionPrefixes\":[\"kb_files/\"]}}" \
                            --region $REGION \
                            --query 'dataSource.dataSourceId' --output text \
                            2>/dev/null || echo "")
                        
                        if [ ! -z "$DS_ID" ]; then
                            echo -e "${GREEN}Created data source: $DS_ID${NC}"
                            
                            # Start data ingestion
                            if aws bedrock-agent start-ingestion-job \
                                --knowledge-base-id $KB_ID \
                                --data-source-id $DS_ID \
                                --region $REGION \
                                2>/dev/null; then
                                echo -e "${GREEN}Started data ingestion for Knowledge Base.${NC}"
                            else
                                echo -e "${YELLOW}Failed to start data ingestion. You'll need to manually start ingestion in the Bedrock console.${NC}"
                            fi
                        else
                            echo -e "${YELLOW}Failed to create data source. You'll need to manually create it in the Bedrock console.${NC}"
                        fi
                        
                        create_checkpoint "$KB_CHECKPOINT"
                    else
                        echo -e "${YELLOW}Failed to create Knowledge Base. Continuing without it.${NC}"
                    fi
                else
                    echo -e "${YELLOW}Could not get collection endpoint. Continuing without Knowledge Base.${NC}"
                fi
            fi
        else
            echo -e "${YELLOW}Failed to create vector collection. Continuing without Knowledge Base.${NC}"
        fi
    else
        echo -e "${YELLOW}Bedrock Knowledge Base is not available in this region. Skipping Knowledge Base creation.${NC}"
        echo -e "${YELLOW}You'll need to manually create the Knowledge Base in the Bedrock console if needed.${NC}"
    fi

    # Create Web Crawler Knowledge Base if supported
    echo -e "${BLUE}Checking for Web Crawler Knowledge Base support...${NC}"
    WEBCRAWLER_SUPPORTED=0
    aws bedrock-agent help create-data-source --data-source-configuration '{"type":"WEB"}' >/dev/null 2>&1 || WEBCRAWLER_SUPPORTED=1

    WEB_KB_ID=""
    if [ $WEBCRAWLER_SUPPORTED -eq 0 ] && [ ! -z "$COLLECTION_ID" ] && [ "$IS_ACTIVE" = true ]; then
        echo -e "${BLUE}Creating Web Crawler Knowledge Base...${NC}"

        # Create the Knowledge Base for web crawling
        WEB_KB_ID=$(aws bedrock-agent create-knowledge-base \
            --name "${PROJECT_NAME}-web-kb" \
            --description "Knowledge Base with web crawled data" \
            --role-arn "$KB_ROLE_ARN" \
            --knowledge-base-configuration "{\"type\":\"VECTOR\",\"vectorKnowledgeBaseConfiguration\":{\"embeddingModelArn\":\"arn:aws:bedrock:$REGION::foundation-model/amazon.titan-embed-text-v1\"}}" \
            --storage-configuration "{\"type\":\"OPENSEARCH_SERVERLESS\",\"opensearchServerlessConfiguration\":{\"collectionArn\":\"arn:aws:aoss:$REGION:$ACCOUNT_ID:collection/$COLLECTION_ID\",\"vectorIndexName\":\"bedrock-web-kb-index\",\"fieldMapping\":{\"textField\":\"text\",\"metadataField\":\"metadata\",\"vectorField\":\"vector\"}}}" \
            --region $REGION \
            --query 'knowledgeBase.knowledgeBaseId' --output text \
            2>/dev/null || echo "")
        
        if [ ! -z "$WEB_KB_ID" ]; then
            echo -e "${GREEN}Created Web Knowledge Base: $WEB_KB_ID${NC}"
            
            # Create web crawling data source
            WEB_DS_ID=$(aws bedrock-agent create-data-source \
                --knowledge-base-id $WEB_KB_ID \
                --name "${PROJECT_NAME}-web-source" \
                --data-source-configuration "{\"type\":\"WEB\",\"webConfiguration\":{\"sourceConfiguration\":{\"urlConfiguration\":{\"seedUrls\":[{\"url\":\"https://docs.aws.amazon.com/bedrock/\"}]}}}}" \
                --vector-ingestion-configuration "{\"chunkingConfiguration\":{\"chunkingStrategy\":\"FIXED_SIZE\",\"fixedSizeChunkingConfiguration\":{\"maxTokens\":300,\"overlapPercentage\":20}}}" \
                --region $REGION \
                --query 'dataSource.dataSourceId' --output text \
                2>/dev/null || echo "")
            
            if [ ! -z "$WEB_DS_ID" ]; then
                echo -e "${GREEN}Created web crawling data source: $WEB_DS_ID${NC}"
                
                # Start data ingestion
                if aws bedrock-agent start-ingestion-job \
                    --knowledge-base-id $WEB_KB_ID \
                    --data-source-id $WEB_DS_ID \
                    --region $REGION \
                    2>/dev/null; then
                    echo -e "${GREEN}Started web crawling ingestion for Knowledge Base.${NC}"
                else
                    echo -e "${YELLOW}Failed to start web crawling ingestion. You'll need to manually start ingestion in the Bedrock console.${NC}"
                fi
            else
                echo -e "${YELLOW}Failed to create web crawling data source. You'll need to manually create it in the Bedrock console.${NC}"
            fi
        else
            echo -e "${YELLOW}Failed to create Web Knowledge Base. Continuing without it.${NC}"
        fi
    else
        echo -e "${YELLOW}Web Crawler Knowledge Base is not supported in this region or OpenSearch collection is not ready. Skipping Web Crawler Knowledge Base creation.${NC}"
        echo -e "${YELLOW}You'll need to manually create the Web Crawler Knowledge Base in the Bedrock console if needed.${NC}"
    fi

    # Create API Gateway
    echo -e "${BLUE}Creating API Gateway...${NC}"
    API_ID=$(aws apigateway create-rest-api \
        --name ${PROJECT_NAME}-api \
        --description "API for storybook System" \
        --region $REGION \
        --query 'id' --output text \
        2>/dev/null || echo "")

    if [ -z "$API_ID" ]; then
        # Try to get existing API
        API_ID=$(aws apigateway get-rest-apis \
            --region $REGION \
            --query "items[?name=='${PROJECT_NAME}-api'].id" \
            --output text 2>/dev/null || echo "")

        if [ -z "$API_ID" ]; then
            if ! handle_error "API Gateway Creation" "Failed to create or retrieve API Gateway" "Check API Gateway permissions in the AWS console" "https://console.aws.amazon.com/apigateway/home?region=$REGION#/apis"; then
                return 1
            fi
        else
            echo -e "${YELLOW}Using existing API Gateway: $API_ID${NC}"
        fi
    fi

    if [ ! -z "$API_ID" ]; then
        # Get the root resource ID
        ROOT_RESOURCE_ID=$(aws apigateway get-resources \
            --rest-api-id $API_ID \
            --region $REGION \
            --query 'items[0].id' --output text 2>/dev/null || echo "")

        if [ -z "$ROOT_RESOURCE_ID" ]; then
            if ! handle_error "API Gateway Resource" "Failed to get root resource ID" "Check API Gateway in the AWS console" "https://console.aws.amazon.com/apigateway/home?region=$REGION#/apis/$API_ID/resources"; then
                return 1
            fi
        fi

        # Create a resource
        RESOURCE_ID=$(aws apigateway create-resource \
            --rest-api-id $API_ID \
            --parent-id $ROOT_RESOURCE_ID \
            --path-part "storybook" \
            --region $REGION \
            --query 'id' --output text \
            2>/dev/null || echo "")

        if [ -z "$RESOURCE_ID" ]; then
            # Try to find existing resource
            RESOURCE_ID=$(aws apigateway get-resources \
                --rest-api-id $API_ID \
                --region $REGION \
                --query "items[?pathPart=='storybook'].id" \
                --output text 2>/dev/null || echo "")

            if [ -z "$RESOURCE_ID" ]; then
                if ! handle_error "API Gateway Resource" "Failed to create API resource" "Check API Gateway permissions in the AWS console" "https://console.aws.amazon.com/apigateway/home?region=$REGION#/apis/$API_ID/resources"; then
                    return 1
                fi
            else
                echo -e "${YELLOW}Using existing API resource: $RESOURCE_ID${NC}"
            fi
        fi

        # Create POST method
        METHOD_RESULT=$(aws apigateway put-method \
            --rest-api-id $API_ID \
            --resource-id $RESOURCE_ID \
            --http-method POST \
            --authorization-type NONE \
            --region $REGION \
            2>/dev/null || echo "")

        if [ -z "$METHOD_RESULT" ]; then
            echo -e "${YELLOW}Method may already exist, continuing...${NC}"
        fi

        # Set up the integration with Lambda
        INTEGRATION_RESULT=$(aws apigateway put-integration \
            --rest-api-id $API_ID \
            --resource-id $RESOURCE_ID \
            --http-method POST \
            --type AWS_PROXY \
            --integration-http-method POST \
            --uri arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${ORCHESTRATOR_ARN}/invocations \
            --region $REGION \
            2>/dev/null || echo "")

        if [ -z "$INTEGRATION_RESULT" ]; then
            echo -e "${YELLOW}Integration may already exist, continuing...${NC}"
        fi

        # Deploy the API
        DEPLOYMENT_ID=$(aws apigateway create-deployment \
            --rest-api-id $API_ID \
            --stage-name prod \
            --region $REGION \
            --query 'id' --output text \
            2>/dev/null || echo "")

        if [ -z "$DEPLOYMENT_ID" ]; then
            if ! handle_error "API Gateway Deployment" "Failed to deploy API" "Check API Gateway in the AWS console" "https://console.aws.amazon.com/apigateway/home?region=$REGION#/apis/$API_ID/stages"; then
                return 1
            fi
        fi

        # Add permission for API Gateway to invoke Lambda
        PERMISSION_RESULT=$(aws lambda add-permission \
            --function-name ${PROJECT_NAME}-orchestrator \
            --statement-id apigateway-prod-$(date +%s) \
            --action lambda:InvokeFunction \
            --principal apigateway.amazonaws.com \
            --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/POST/storybook" \
            --region $REGION \
            2>/dev/null || echo "")

        if [ -z "$PERMISSION_RESULT" ]; then
            echo -e "${YELLOW}Lambda permission may already exist, continuing...${NC}"
        }

        echo -e "${GREEN}API Gateway deployed. Endpoint: https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook${NC}"
        create_checkpoint "$API_CHECKPOINT"
    else
        echo -e "${RED}Failed to create or retrieve API Gateway. You'll need to manually configure API Gateway in the AWS console.${NC}"
    fi

    # Create Flow templates directory
    mkdir -p flow_templates

    # Now let's create our Bedrock Flow templates
    echo -e "${BLUE}Creating Bedrock Flow templates...${NC}"

    # Create the multi-agent flow template
    cat > flow_templates/storybook_flow.json << EOF
{
    "name": "${PROJECT_NAME}-workflow",
    "description": "Storybook novel creation workflow using Bedrock Flows",
    "executionRoleArn": "${FLOW_ROLE_ARN}",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [{ "name": "document", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "ProjectManagerPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Executive Director for novel creation, responsible for overseeing the entire novel creation process and delegating tasks. Review the project context and task, then provide a detailed, thoughtful response.\n\nProject Context: {{projectContext}}\nTask: {{task}}\n\nGive a comprehensive plan and guidance for this task, considering all relevant aspects of novel creation."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Condition",
                "name": "PhaseCondition",
                "configuration": {
                    "condition": {
                        "conditions": [
                            { "name": "initialization", "expression": "$.project_phase == 'initialization'" },
                            { "name": "development", "expression": "$.project_phase == 'development'" },
                            { "name": "creation", "expression": "$.project_phase == 'creation'" },
                            { "name": "refinement", "expression": "$.project_phase == 'refinement'" },
                            { "name": "finalization", "expression": "$.project_phase == 'finalization'" },
                            { "name": "default" }
                        ]
                    }
                },
                "inputs": [
                    { "name": "project_phase", "type": "String", "expression": "$.project_phase" }
                ]
            },
            {
                "type": "Prompt",
                "name": "InitializationPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Creative Director for this novel project, currently in the INITIALIZATION phase. In this phase, you focus on establishing the core vision and concept.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nExecutive Director's Guidance: {{directorGuidance}}\n\nProvide creative direction for this initialization task, focusing on the overall concept, genre conventions, target audience, and unique selling points for this novel."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "directorGuidance", "type": "String", "expression": "$.directorOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "DevelopmentPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Structure Architect for this novel project, currently in the DEVELOPMENT phase. In this phase, you focus on designing the overall narrative structure, character arcs, and world-building elements.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nExecutive Director's Guidance: {{directorGuidance}}\n\nProvide detailed structural development for this task, focusing on plot structure, character development, world-building, and thematic elements."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "directorGuidance", "type": "String", "expression": "$.directorOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "CreationPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Content Development Director for this novel project, currently in the CREATION phase. In this phase, you focus on actual content generation, scene construction, and dialogue crafting.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nExecutive Director's Guidance: {{directorGuidance}}\n\nGenerate compelling content for this task, focusing on vivid scenes, authentic dialogue, emotional resonance, and narrative flow."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "directorGuidance", "type": "String", "expression": "$.directorOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "RefinementPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Editorial Director for this novel project, currently in the REFINEMENT phase. In this phase, you focus on improving and polishing the existing content.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nExecutive Director's Guidance: {{directorGuidance}}\n\nProvide editorial refinement for this task, focusing on prose enhancement, structural coherence, character consistency, and thematic depth."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "directorGuidance", "type": "String", "expression": "$.directorOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "FinalizationPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Quality Assessment Director for this novel project, currently in the FINALIZATION phase. In this phase, you focus on final quality checks and market positioning.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nExecutive Director's Guidance: {{directorGuidance}}\n\nProvide finalization assessment for this task, focusing on overall quality, market positioning, blurb optimization, and final formatting requirements."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "directorGuidance", "type": "String", "expression": "$.directorOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "DefaultAgentPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are a Novel Writing Assistant for this project. Provide guidance and assistance based on the current task.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nExecutive Director's Guidance: {{directorGuidance}}\n\nProvide detailed assistance for this task, focusing on practical writing advice, research guidance, and creative suggestions."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "directorGuidance", "type": "String", "expression": "$.directorOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Lambda",
                "name": "StoreResults",
                "configuration": {
                    "lambda": {
                        "functionArn": "${DATA_PROCESSOR_ARN}"
                    }
                },
                "inputs": [
                    { "name": "operation", "type": "String", "expression": "'store_project_data'" },
                    { "name": "project_id", "type": "String", "expression": "$.project_id" },
                    { "name": "data", "type": "Map", "expression": "{ 'task_response': $.specialist_response, 'last_task': $.task, 'updated_at': $$.Execution.CurrentTimestamp }" }
                ],
                "outputs": [
                    { "name": "statusCode", "type": "Number" },
                    { "name": "message", "type": "String" },
                    { "name": "output", "type": "Map" }
                ]
            },
            {
                "type": "Output",
                "name": "FlowOutput",
                "inputs": [
                    { "name": "document", "type": "String", "expression": "$.specialist_response" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_ExecutiveDirector",
                "source": "FlowInputNode",
                "target": "ProjectManagerPrompt",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "document", "targetInput": "$.document" }
                }
            },
            {
                "name": "ExecutiveDirector_to_PhaseCondition",
                "source": "ProjectManagerPrompt",
                "target": "PhaseCondition",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "directorOutput"
                    }
                }
            },
            {
                "name": "PhaseCondition_to_Initialization",
                "source": "PhaseCondition",
                "target": "InitializationPrompt",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "initialization" }
                }
            },
            {
                "name": "PhaseCondition_to_Development",
                "source": "PhaseCondition",
                "target": "DevelopmentPrompt",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "development" }
                }
            },
            {
                "name": "PhaseCondition_to_Creation",
                "source": "PhaseCondition",
                "target": "CreationPrompt",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "creation" }
                }
            },
            {
                "name": "PhaseCondition_to_Refinement",
                "source": "PhaseCondition",
                "target": "RefinementPrompt",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "refinement" }
                }
            },
            {
                "name": "PhaseCondition_to_Finalization",
                "source": "PhaseCondition",
                "target": "FinalizationPrompt",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "finalization" }
                }
            },
            {
                "name": "PhaseCondition_to_Default",
                "source": "PhaseCondition",
                "target": "DefaultAgentPrompt",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "default" }
                }
            },
            {
                "name": "Initialization_to_StoreResults",
                "source": "InitializationPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "specialist_response"
                    }
                }
            },
            {
                "name": "Development_to_StoreResults",
                "source": "DevelopmentPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "specialist_response"
                    }
                }
            },
            {
                "name": "Creation_to_StoreResults",
                "source": "CreationPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "specialist_response"
                    }
                }
            },
            {
                "name": "Refinement_to_StoreResults",
                "source": "RefinementPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "specialist_response"
                    }
                }
            },
            {
                "name": "Finalization_to_StoreResults",
                "source": "FinalizationPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "specialist_response"
                    }
                }
            },
            {
                "name": "Default_to_StoreResults",
                "source": "DefaultAgentPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "specialist_response"
                    }
                }
            },
            {
                "name": "StoreResults_to_Output",
                "source": "StoreResults",
                "target": "FlowOutput",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "output.specialist_response",
                        "targetInput": "document"
                    }
                }
            }
        ]
    }
}
EOF

    # Create agent-based flow template
    cat > flow_templates/storybook_agent_flow.json << EOF
{
    "name": "${PROJECT_NAME}-agent-workflow",
    "description": "Storybook novel creation workflow using Bedrock Agents with multi-agent collaboration",
    "executionRoleArn": "${FLOW_ROLE_ARN}",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [{ "name": "document", "type": "String" }]
            },
            {
                "type": "Lambda",
                "name": "PrepareAgentInput",
                "configuration": {
                    "lambda": {
                        "functionArn": "${DATA_PROCESSOR_ARN}"
                    }
                },
                "inputs": [
                    { "name": "operation", "type": "String", "expression": "'web_search'" },
                    { "name": "query", "type": "String", "expression": "$.document.task" }
                ],
                "outputs": [
                    { "name": "statusCode", "type": "Number" },
                    { "name": "message", "type": "String" },
                    { "name": "output", "type": "Map" }
                ]
            },
            {
                "type": "Lambda",
                "name": "InvokeAgent",
                "configuration": {
                    "lambda": {
                        "functionArn": "${ORCHESTRATOR_ARN}"
                    }
                },
                "inputs": [
                    { "name": "operation", "type": "String", "expression": "'invoke_agent'" },
                    { "name": "input_text", "type": "String", "expression": "$.document.task + ' ' + $.search_results" },
                    { "name": "agent_id", "type": "String", "expression": "'AGENT_ID_PLACEHOLDER'" },
                    { "name": "agent_alias_id", "type": "String", "expression": "'AGENT_ALIAS_ID_PLACEHOLDER'" }
                ],
                "outputs": [
                    { "name": "statusCode", "type": "Number" },
                    { "name": "body", "type": "Map" }
                ]
            },
            {
                "type": "Lambda",
                "name": "StoreResults",
                "configuration": {
                    "lambda": {
                        "functionArn": "${DATA_PROCESSOR_ARN}"
                    }
                },
                "inputs": [
                    { "name": "operation", "type": "String", "expression": "'store_project_data'" },
                    { "name": "project_id", "type": "String", "expression": "$.document.project_id" },
                    { "name": "data", "type": "Map", "expression": "{ 'task_response': $.agent_response, 'last_task': $.document.task, 'updated_at': $$.Execution.CurrentTimestamp }" }
                ],
                "outputs": [
                    { "name": "statusCode", "type": "Number" },
                    { "name": "message", "type": "String" },
                    { "name": "output", "type": "Map" }
                ]
            },
            {
                "type": "Output",
                "name": "FlowOutput",
                "inputs": [
                    { "name": "document", "type": "String", "expression": "$.agent_response" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_PrepareAgentInput",
                "source": "FlowInputNode",
                "target": "PrepareAgentInput",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "document", "targetInput": "$.document" }
                }
            },
            {
                "name": "PrepareAgentInput_to_InvokeAgent",
                "source": "PrepareAgentInput",
                "target": "InvokeAgent",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "output.results",
                        "targetInput": "search_results"
                    }
                }
            },
            {
                "name": "InvokeAgent_to_StoreResults",
                "source": "InvokeAgent",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "body.response",
                        "targetInput": "agent_response"
                    }
                }
            },
            {
                "name": "StoreResults_to_Output",
                "source": "StoreResults",
                "target": "FlowOutput",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "output.task_response",
                        "targetInput": "document"
                    }
                }
            }
        ]
    }
}
EOF

    # Create flow with KB integration (if KB was created)
    if [ ! -z "$KB_ID" ]; then
        cat > flow_templates/storybook_kb_flow.json << EOF
{
    "name": "${PROJECT_NAME}-kb-workflow",
    "description": "Storybook novel creation workflow with Knowledge Base integration using Bedrock Flows",
    "executionRoleArn": "${FLOW_ROLE_ARN}",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [{ "name": "document", "type": "String" }]
            },
            {
                "type": "KnowledgeBase",
                "name": "WritingGuide",
                "configuration": {
                    "knowledgeBase": {
                        "knowledgeBaseId": "${KB_ID}",
                        "modelId": "${CLAUDE_MODEL}"
                    }
                },
                "inputs": [
                    { "name": "retrievalQuery", "type": "String", "expression": "$.task + ' ' + $.project_title" }
                ],
                "outputs": [{ "name": "outputText", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "ProjectManagerPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Executive Director for novel creation, responsible for overseeing the entire novel creation process and delegating tasks. Review the project context, task, and writing guide information, then provide a detailed, thoughtful response.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nWriting Guide Information: {{kbInfo}}\n\nGive a comprehensive plan and guidance for this task, considering all relevant aspects of novel creation and incorporating the insights from the writing guide."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis + ' - Current phase: ' + $.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "kbInfo", "type": "String", "expression": "$.kbInfo" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Condition",
                "name": "PhaseCondition",
                "configuration": {
                    "condition": {
                        "conditions": [
                            { "name": "initialization", "expression": "$.project_phase == 'initialization'" },
                            { "name": "development", "expression": "$.project_phase == 'development'" },
                            { "name": "creation", "expression": "$.project_phase == 'creation'" },
                            { "name": "refinement", "expression": "$.project_phase == 'refinement'" },
                            { "name": "finalization", "expression": "$.project_phase == 'finalization'" },
                            { "name": "default" }
                        ]
                    }
                },
                "inputs": [
                    { "name": "project_phase", "type": "String", "expression": "$.project_phase" }
                ]
            },
            {
                "type": "Prompt",
                "name": "SpecialistPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are a Novel Writing Specialist for this project in the {{projectPhase}} phase. Provide specialized guidance for this phase of novel development.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nWriting Guide Information: {{kbInfo}}\nExecutive Director's Guidance: {{directorGuidance}}\n\nProvide detailed specialized assistance for this task, focusing on the specific needs of the {{projectPhase}} phase."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis" },
                    { "name": "projectPhase", "type": "String", "expression": "$.project_phase" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "kbInfo", "type": "String", "expression": "$.kbInfo" },
                    { "name": "directorGuidance", "type": "String", "expression": "$.directorOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Lambda",
                "name": "StoreResults",
                "configuration": {
                    "lambda": {
                        "functionArn": "${DATA_PROCESSOR_ARN}"
                    }
                },
                "inputs": [
                    { "name": "operation", "type": "String", "expression": "'store_project_data'" },
                    { "name": "project_id", "type": "String", "expression": "$.project_id" },
                    { "name": "data", "type": "Map", "expression": "{ 'task_response': $.specialist_response, 'last_task': $.task, 'updated_at': $$.Execution.CurrentTimestamp }" }
                ],
                "outputs": [
                    { "name": "statusCode", "type": "Number" },
                    { "name": "message", "type": "String" },
                    { "name": "output", "type": "Map" }
                ]
            },
            {
                "type": "Output",
                "name": "FlowOutput",
                "inputs": [
                    { "name": "document", "type": "String", "expression": "$.specialist_response" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_KB",
                "source": "FlowInputNode",
                "target": "WritingGuide",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "document", "targetInput": "$.document" }
                }
            },
            {
                "name": "KB_to_ExecutiveDirector",
                "source": "WritingGuide",
                "target": "ProjectManagerPrompt",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "outputText", "targetInput": "kbInfo" }
                }
            },
            {
                "name": "ExecutiveDirector_to_PhaseCondition",
                "source": "ProjectManagerPrompt",
                "target": "PhaseCondition",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "directorOutput"
                    }
                }
            },
            {
                "name": "PhaseCondition_to_SpecialistPrompt",
                "source": "PhaseCondition",
                "target": "SpecialistPrompt",
                "type": "Data",
                "configuration": {
                    "data": { 
                        "sourceOutput": "",
                        "targetInput": "" 
                    }
                }
            },
            {
                "name": "SpecialistPrompt_to_StoreResults",
                "source": "SpecialistPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "specialist_response"
                    }
                }
            },
            {
                "name": "StoreResults_to_Output",
                "source": "StoreResults",
                "target": "FlowOutput",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "output.specialist_response",
                        "targetInput": "document"
                    }
                }
            }
        ]
    }
}
EOF
    }

    # If web crawler KB was created, create a flow template for it
    if [ ! -z "$WEB_KB_ID" ]; then
        cat > flow_templates/storybook_web_kb_flow.json << EOF
{
    "name": "${PROJECT_NAME}-web-kb-workflow",
    "description": "Storybook novel creation workflow with Web Crawler Knowledge Base integration",
    "executionRoleArn": "${FLOW_ROLE_ARN}",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [{ "name": "document", "type": "String" }]
            },
            {
                "type": "KnowledgeBase",
                "name": "WebResourcesKB",
                "configuration": {
                    "knowledgeBase": {
                        "knowledgeBaseId": "${WEB_KB_ID}",
                        "modelId": "${CLAUDE_MODEL}"
                    }
                },
                "inputs": [
                    { "name": "retrievalQuery", "type": "String", "expression": "$.task + ' ' + $.project_title" }
                ],
                "outputs": [{ "name": "outputText", "type": "String" }]
            },
            {
                "type": "Lambda",
                "name": "WebSearch",
                "configuration": {
                    "lambda": {
                        "functionArn": "${DATA_PROCESSOR_ARN}"
                    }
                },
                "inputs": [
                    { "name": "operation", "type": "String", "expression": "'web_search'" },
                    { "name": "query", "type": "String", "expression": "$.task + ' ' + $.project_title" },
                    { "name": "max_results", "type": "Number", "expression": "3" }
                ],
                "outputs": [
                    { "name": "statusCode", "type": "Number" },
                    { "name": "message", "type": "String" },
                    { "name": "output", "type": "Map" }
                ]
            },
            {
                "type": "Prompt",
                "name": "ContentGenerationPrompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "${CLAUDE_MODEL}",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.7, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are a Novel Writing Assistant that can research and incorporate web content. Provide specialized guidance.\n\nProject Context: {{projectContext}}\nTask: {{task}}\nWeb Knowledge Base Info: {{kbInfo}}\nWeb Search Results: {{webSearchResults}}\n\nProvide detailed assistance for this task, incorporating both the knowledge base information and web search results. Focus on creative, practical advice based on research."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "projectContext", "type": "String", "expression": "$.project_title + ' - ' + $.project_synopsis" },
                    { "name": "task", "type": "String", "expression": "$.task" },
                    { "name": "kbInfo", "type": "String", "expression": "$.kbInfo" },
                    { "name": "webSearchResults", "type": "String", "expression": "$.webSearchResults" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Lambda",
                "name": "StoreResults",
                "configuration": {
                    "lambda": {
                        "functionArn": "${DATA_PROCESSOR_ARN}"
                    }
                },
                "inputs": [
                    { "name": "operation", "type": "String", "expression": "'store_project_data'" },
                    { "name": "project_id", "type": "String", "expression": "$.project_id" },
                    { "name": "data", "type": "Map", "expression": "{ 'task_response': $.generated_content, 'last_task': $.task, 'updated_at': $$.Execution.CurrentTimestamp }" }
                ],
                "outputs": [
                    { "name": "statusCode", "type": "Number" },
                    { "name": "message", "type": "String" },
                    { "name": "output", "type": "Map" }
                ]
            },
            {
                "type": "Output",
                "name": "FlowOutput",
                "inputs": [
                    { "name": "document", "type": "String", "expression": "$.generated_content" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_KB",
                "source": "FlowInputNode",
                "target": "WebResourcesKB",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "document", "targetInput": "$.document" }
                }
            },
            {
                "name": "Input_to_WebSearch",
                "source": "FlowInputNode",
                "target": "WebSearch",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "document", "targetInput": "$.document" }
                }
            },
            {
                "name": "KB_to_ContentGeneration",
                "source": "WebResourcesKB",
                "target": "ContentGenerationPrompt",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "outputText", "targetInput": "kbInfo" }
                }
            },
            {
                "name": "WebSearch_to_ContentGeneration",
                "source": "WebSearch",
                "target": "ContentGenerationPrompt",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "output.results", "targetInput": "webSearchResults" }
                }
            },
            {
                "name": "ContentGeneration_to_StoreResults",
                "source": "ContentGenerationPrompt",
                "target": "StoreResults",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "modelCompletion",
                        "targetInput": "generated_content"
                    }
                }
            },
            {
                "name": "StoreResults_to_Output",
                "source": "StoreResults",
                "target": "FlowOutput",
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": "output.task_response",
                        "targetInput": "document"
                    }
                }
            }
        ]
    }
}
EOF
    fi

    # Create Agent setup for storybook
    echo -e "${BLUE}Creating Bedrock Agents for storybook workflow...${NC}"

    # Create agents if supported
    AGENTS_SUPPORTED=0
    aws bedrock-agent help create-agent >/dev/null 2>&1 || AGENTS_SUPPORTED=1

    if [ $AGENTS_SUPPORTED -eq 0 ]; then
        echo -e "${BLUE}Creating Bedrock Agents...${NC}"
        
        # Create agent role
        AGENT_ROLE_NAME="${PROJECT_NAME}-agentrole"
        
        # Create a policy document for the agent role trust relationship
        cat > agent-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

        # Create the agent IAM role
        if ! aws iam create-role \
            --role-name $AGENT_ROLE_NAME \
            --assume-role-policy-document file://agent-trust-policy.json \
            2>/dev/null; then

            # Check if role exists
            if aws iam get-role --role-name $AGENT_ROLE_NAME 2>/dev/null; then
                echo -e "${YELLOW}Role $AGENT_ROLE_NAME already exists. Continuing...${NC}"
            else
                if ! handle_error "IAM Role Creation" "Failed to create role $AGENT_ROLE_NAME" "Check if you have sufficient permissions to create IAM roles" "https://console.aws.amazon.com/iam/home?#/roles"; then
                    return 1
                fi
            fi
        fi
        
        # Get the IAM role ARN
        AGENT_ROLE_ARN=$(aws iam get-role --role-name ${AGENT_ROLE_NAME} --query 'Role.Arn' --output text 2>/dev/null)
        
        # Create agent policy
        cat > agent-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:WebSearch"
            ],
            "Resource": [
                "arn:aws:bedrock:${REGION}::foundation-model/${CLAUDE_MODEL}"
            ]
        }
    ]
}
EOF

        # Create or check for existing agent policy
        if ! aws iam create-policy \
            --policy-name ${PROJECT_NAME}-agent-policy \
            --policy-document file://agent-policy.json \
            2>/dev/null; then

            # Check if policy exists
            AGENT_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-agent-policy"
            if aws iam get-policy --policy-arn $AGENT_POLICY_ARN 2>/dev/null; then
                echo -e "${YELLOW}Agent policy already exists. Continuing...${NC}"
            else
                if ! handle_error "IAM Policy Creation" "Failed to create agent policy" "Check if you have sufficient permissions to create IAM policies" "https://console.aws.amazon.com/iam/home?#/policies"; then
                    return 1
                fi
            fi
        fi
        
        # Attach policy to Agent role
        AGENT_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-agent-policy"
        if ! aws iam attach-role-policy \
            --role-name $AGENT_ROLE_NAME \
            --policy-arn $AGENT_POLICY_ARN \
            2>/dev/null; then
            if ! handle_error "IAM Policy Attachment" "Failed to attach agent policy" "Check role permissions in IAM console" "https://console.aws.amazon.com/iam/home?#/roles/$AGENT_ROLE_NAME"; then
                return 1
            fi
        fi

        # Add KB permissions if KB was created
        if [ ! -z "$KB_ID" ] || [ ! -z "$WEB_KB_ID" ]; then
            # Create KB access policy for agent
            cat > agent-kb-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate"
            ],
            "Resource": [
EOF
            if [ ! -z "$KB_ID" ]; then
                echo "                \"arn:aws:bedrock:${REGION}:${ACCOUNT_ID}:knowledge-base/${KB_ID}\"" >> agent-kb-policy.json
                if [ ! -z "$WEB_KB_ID" ]; then
                    echo "                ," >> agent-kb-policy.json
                fi
            fi
            if [ ! -z "$WEB_KB_ID" ]; then
                echo "                \"arn:aws:bedrock:${REGION}:${ACCOUNT_ID}:knowledge-base/${WEB_KB_ID}\"" >> agent-kb-policy.json
            fi
            echo "            ]" >> agent-kb-policy.json
            echo "        }" >> agent-kb-policy.json
            echo "    ]" >> agent-kb-policy.json
            echo "}" >> agent-kb-policy.json

            # Create or check for existing agent KB policy
            if ! aws iam create-policy \
                --policy-name ${PROJECT_NAME}-agent-kb-policy \
                --policy-document file://agent-kb-policy.json \
                2>/dev/null; then

                # Check if policy exists
                AGENT_KB_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-agent-kb-policy"
                if aws iam get-policy --policy-arn $AGENT_KB_POLICY_ARN 2>/dev/null; then
                    echo -e "${YELLOW}Agent KB policy already exists. Continuing...${NC}"
                else
                    if ! handle_error "IAM Policy Creation" "Failed to create agent KB policy" "Check if you have sufficient permissions to create IAM policies" "https://console.aws.amazon.com/iam/home?#/policies"; then
                        return 1
                    fi
                fi
            fi

            # Attach KB policy to Agent role
            AGENT_KB_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-agent-kb-policy"
            if ! aws iam attach-role-policy \
                --role-name $AGENT_ROLE_NAME \
                --policy-arn $AGENT_KB_POLICY_ARN \
                2>/dev/null; then
                if ! handle_error "IAM Policy Attachment" "Failed to attach agent KB policy" "Check role permissions in IAM console" "https://console.aws.amazon.com/iam/home?#/roles/$AGENT_ROLE_NAME"; then
                    return 1
                fi
            fi
        fi
        
        # Wait for IAM role to propagate
        echo -e "${BLUE}Waiting for IAM agent role to propagate...${NC}"
        sleep 30
        
        # Create Executive Director Agent
        EXECUTIVE_AGENT_ID=$(aws bedrock-agent create-agent \
            --agent-name "${PROJECT_NAME}-executive-director" \
            --agent-resource-role-arn "$AGENT_ROLE_ARN" \
            --foundation-model "$CLAUDE_MODEL" \
            --instruction "You are the Executive Director for novel creation, responsible for overseeing the entire novel creation process and delegating tasks. You review project context and tasks, then provide detailed, thoughtful responses. You create comprehensive plans and guidance, considering all relevant aspects of novel creation." \
            --region $REGION \
            --query 'agent.agentId' --output text \
            2>/dev/null || echo "")
        
        if [ ! -z "$EXECUTIVE_AGENT_ID" ]; then
            echo -e "${GREEN}Created Executive Director Agent: $EXECUTIVE_AGENT_ID${NC}"
            
            # Create an agent alias
            EXECUTIVE_AGENT_ALIAS_ID=$(aws bedrock-agent create-agent-alias \
                --agent-id $EXECUTIVE_AGENT_ID \
                --agent-alias-name "latest" \
                --region $REGION \
                --query 'agentAlias.agentAliasId' --output text \
                2>/dev/null || echo "")
            
            if [ ! -z "$EXECUTIVE_AGENT_ALIAS_ID" ]; then
                echo -e "${GREEN}Created Executive Director Agent Alias: $EXECUTIVE_AGENT_ALIAS_ID${NC}"
            else
                echo -e "${YELLOW}Failed to create Executive Director Agent Alias. You'll need to create it manually.${NC}"
            fi
        else
            echo -e "${YELLOW}Failed to create Executive Director Agent. Continuing without it.${NC}"
        fi
        
        # Create Creative Director Agent
        CREATIVE_AGENT_ID=$(aws bedrock-agent create-agent \
            --agent-name "${PROJECT_NAME}-creative-director" \
            --agent-resource-role-arn "$AGENT_ROLE_ARN" \
            --foundation-model "$CLAUDE_MODEL" \
            --instruction "You are the Creative Director for novel projects, currently focused on the INITIALIZATION phase. In this phase, you focus on establishing the core vision and concept. You provide creative direction for initialization tasks, focusing on the overall concept, genre conventions, target audience, and unique selling points for novels." \
            --region $REGION \
            --query 'agent.agentId' --output text \
            2>/dev/null || echo "")
        
        if [ ! -z "$CREATIVE_AGENT_ID" ]; then
            echo -e "${GREEN}Created Creative Director Agent: $CREATIVE_AGENT_ID${NC}"
            
            # Create an agent alias
            CREATIVE_AGENT_ALIAS_ID=$(aws bedrock-agent create-agent-alias \
                --agent-id $CREATIVE_AGENT_ID \
                --agent-alias-name "latest" \
                --region $REGION \
                --query 'agentAlias.agentAliasId' --output text \
                2>/dev/null || echo "")
            
            if [ ! -z "$CREATIVE_AGENT_ALIAS_ID" ]; then
                echo -e "${GREEN}Created Creative Director Agent Alias: $CREATIVE_AGENT_ALIAS_ID${NC}"
            else
                echo -e "${YELLOW}Failed to create Creative Director Agent Alias. You'll need to create it manually.${NC}"
            fi
        else
            echo -e "${YELLOW}Failed to create Creative Director Agent. Continuing without it.${NC}"
        fi
        
        # Create Development Agent
        DEVELOPMENT_AGENT_ID=$(aws bedrock-agent create-agent \
            --agent-name "${PROJECT_NAME}-development-director" \
            --agent-resource-role-arn "$AGENT_ROLE_ARN" \
            --foundation-model "$CLAUDE_MODEL" \
            --instruction "You are the Development Director for novel projects, currently focused on the DEVELOPMENT phase. In this phase, you focus on designing the overall narrative structure, character arcs, and world-building elements. You provide detailed structural development, focusing on plot structure, character development, world-building, and thematic elements." \
            --region $REGION \
            --query 'agent.agentId' --output text \
            2>/dev/null || echo "")
        
        if [ ! -z "$DEVELOPMENT_AGENT_ID" ]; then
            echo -e "${GREEN}Created Development Director Agent: $DEVELOPMENT_AGENT_ID${NC}"
            
            # Create an agent alias
            DEVELOPMENT_AGENT_ALIAS_ID=$(aws bedrock-agent create-agent-alias \
                --agent-id $DEVELOPMENT_AGENT_ID \
                --agent-alias-name "latest" \
                --region $REGION \
                --query 'agentAlias.agentAliasId' --output text \
                2>/dev/null || echo "")
            
            if [ ! -z "$DEVELOPMENT_AGENT_ALIAS_ID" ]; then
                echo -e "${GREEN}Created Development Director Agent Alias: $DEVELOPMENT_AGENT_ALIAS_ID${NC}"
            else
                echo -e "${YELLOW}Failed to create Development Director Agent Alias. You'll need to create it manually.${NC}"
            fi
        else
            echo -e "${YELLOW}Failed to create Development Director Agent. Continuing without it.${NC}"
        fi
        
        # If we have created agents, create a supervisor agent with collaboration
        if [ ! -z "$EXECUTIVE_AGENT_ID" ] && [ ! -z "$CREATIVE_AGENT_ID" ] && [ ! -z "$DEVELOPMENT_AGENT_ID" ]; then
            echo -e "${BLUE}Creating Supervisor Agent with collaboration...${NC}"
            
            # Create supervisor agent
            SUPERVISOR_AGENT_ID=$(aws bedrock-agent create-agent \
                --agent-name "${PROJECT_NAME}-supervisor" \
                --agent-resource-role-arn "$AGENT_ROLE_ARN" \
                --foundation-model "$CLAUDE_MODEL" \
                --instruction "You are the Supervisor for the novel creation process. You coordinate between different specialized agents to complete the novel creation process efficiently. You delegate tasks to the appropriate specialized agents based on the current phase of the project." \
                --agent-collaboration "SUPERVISOR" \
                --region $REGION \
                --query 'agent.agentId' --output text \
                2>/dev/null || echo "")
            
            if [ ! -z "$SUPERVISOR_AGENT_ID" ]; then
                echo -e "${GREEN}Created Supervisor Agent: $SUPERVISOR_AGENT_ID${NC}"
                
                # Associate collaborator agents
                if aws bedrock-agent associate-agent-collaborator \
                    --agent-id $SUPERVISOR_AGENT_ID \
                    --agent-version "DRAFT" \
                    --agent-descriptor "{\"aliasArn\":\"arn:aws:bedrock:$REGION:$ACCOUNT_ID:agent-alias/$EXECUTIVE_AGENT_ID/$EXECUTIVE_AGENT_ALIAS_ID\"}" \
                    --collaborator-name "ExecutiveDirector" \
                    --collaboration-instruction "Use this collaborator for overseeing the entire novel creation process and delegating tasks." \
                    --relay-conversation-history "DISABLED" \
                    --region $REGION \
                    2>/dev/null; then
                    echo -e "${GREEN}Associated Executive Director Agent as collaborator.${NC}"
                else
                    echo -e "${YELLOW}Failed to associate Executive Director Agent as collaborator.${NC}"
                fi
                
                if aws bedrock-agent associate-agent-collaborator \
                    --agent-id $SUPERVISOR_AGENT_ID \
                    --agent-version "DRAFT" \
                    --agent-descriptor "{\"aliasArn\":\"arn:aws:bedrock:$REGION:$ACCOUNT_ID:agent-alias/$CREATIVE_AGENT_ID/$CREATIVE_AGENT_ALIAS_ID\"}" \
                    --collaborator-name "CreativeDirector" \
                    --collaboration-instruction "Use this collaborator for establishing the core vision and concept during the initialization phase." \
                    --relay-conversation-history "DISABLED" \
                    --region $REGION \
                    2>/dev/null; then
                    echo -e "${GREEN}Associated Creative Director Agent as collaborator.${NC}"
                else
                    echo -e "${YELLOW}Failed to associate Creative Director Agent as collaborator.${NC}"
                fi
                
                if aws bedrock-agent associate-agent-collaborator \
                    --agent-id $SUPERVISOR_AGENT_ID \
                    --agent-version "DRAFT" \
                    --agent-descriptor "{\"aliasArn\":\"arn:aws:bedrock:$REGION:$ACCOUNT_ID:agent-alias/$DEVELOPMENT_AGENT_ID/$DEVELOPMENT_AGENT_ALIAS_ID\"}" \
                    --collaborator-name "DevelopmentDirector" \
                    --collaboration-instruction "Use this collaborator for designing the overall narrative structure, character arcs, and world-building elements during the development phase." \
                    --relay-conversation-history "DISABLED" \
                    --region $REGION \
                    2>/dev/null; then
                    echo -e "${GREEN}Associated Development Director Agent as collaborator.${NC}"
                else
                    echo -e "${YELLOW}Failed to associate Development Director Agent as collaborator.${NC}"
                fi
                
                # Associate Knowledge Base with supervisor agent if KB exists
                if [ ! -z "$KB_ID" ]; then
                    if aws bedrock-agent associate-agent-knowledge-base \
                        --agent-id $SUPERVISOR_AGENT_ID \
                        --agent-version "DRAFT" \
                        --knowledge-base-id $KB_ID \
                        --description "Novel writing guide knowledge base" \
                        --region $REGION \
                        2>/dev/null; then
                        echo -e "${GREEN}Associated Knowledge Base with Supervisor Agent.${NC}"
                    else
                        echo -e "${YELLOW}Failed to associate Knowledge Base with Supervisor Agent.${NC}"
                    fi
                fi
                
                # Associate Web KB with supervisor agent if Web KB exists
                if [ ! -z "$WEB_KB_ID" ]; then
                    if aws bedrock-agent associate-agent-knowledge-base \
                        --agent-id $SUPERVISOR_AGENT_ID \
                        --agent-version "DRAFT" \
                        --knowledge-base-id $WEB_KB_ID \
                        --description "Web resources knowledge base" \
                        --region $REGION \
                        2>/dev/null; then
                        echo -e "${GREEN}Associated Web Knowledge Base with Supervisor Agent.${NC}"
                    else
                        echo -e "${YELLOW}Failed to associate Web Knowledge Base with Supervisor Agent.${NC}"
                    fi
                fi
                
                # Prepare and create alias for supervisor agent
                if aws bedrock-agent prepare-agent \
                    --agent-id $SUPERVISOR_AGENT_ID \
                    --region $REGION \
                    2>/dev/null; then
                    echo -e "${GREEN}Prepared Supervisor Agent.${NC}"
                    
                    # Wait for agent to be ready
                    echo -e "${BLUE}Waiting for agent to be ready...${NC}"
                    sleep 30
                    
                    # Create an agent alias
                    SUPERVISOR_AGENT_ALIAS_ID=$(aws bedrock-agent create-agent-alias \
                        --agent-id $SUPERVISOR_AGENT_ID \
                        --agent-alias-name "latest" \
                        --region $REGION \
                        --query 'agentAlias.agentAliasId' --output text \
                        2>/dev/null || echo "")
                    
                    if [ ! -z "$SUPERVISOR_AGENT_ALIAS_ID" ]; then
                        echo -e "${GREEN}Created Supervisor Agent Alias: $SUPERVISOR_AGENT_ALIAS_ID${NC}"
                        # Update the Lambda environment variables with agent ID
                        if ! aws lambda update-function-configuration \
                            --function-name ${PROJECT_NAME}-orchestrator \
                            --environment "Variables={REGION=$REGION,CLAUDE_MODEL=$CLAUDE_MODEL,AGENT_ID=$SUPERVISOR_AGENT_ID,AGENT_ALIAS_ID=$SUPERVISOR_AGENT_ALIAS_ID}" \
                            --region $REGION \
                            2>/dev/null; then
                            echo -e "${YELLOW}Failed to update Lambda environment variables with Agent ID. You'll need to update them manually.${NC}"
                        else
                            echo -e "${GREEN}Updated Lambda environment variables with Agent ID.${NC}"
                        fi
                    else
                        echo -e "${YELLOW}Failed to create Supervisor Agent Alias. You'll need to create it manually.${NC}"
                    fi
                else
                    echo -e "${YELLOW}Failed to prepare Supervisor Agent. You'll need to prepare it manually.${NC}"
                fi
                
                create_checkpoint "$AGENT_CHECKPOINT"
            else
                echo -e "${YELLOW}Failed to create Supervisor Agent. Continuing without it.${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Bedrock Agents are not supported in this region. Skipping Agent creation.${NC}"
        echo -e "${YELLOW}You'll need to manually create the Agents in the Bedrock console if needed.${NC}"
    fi

    # Create Bedrock Flow
    echo -e "${BLUE}Creating Bedrock Flow...${NC}"
    
    FLOW_ID=$(aws bedrock create-flow \
        --cli-input-json file://flow_templates/storybook_flow.json \
        --region $REGION \
        --query 'flowId' --output text \
        2>/dev/null || echo "")

    if [ -z "$FLOW_ID" ]; then
        if ! handle_error "Bedrock Flow Creation" "Failed to create Bedrock Flow" "Check if you have sufficient permissions to create Bedrock Flows" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows"; then
            return 1
        fi
    else
        echo -e "${GREEN}Created Bedrock Flow with ID: $FLOW_ID${NC}"

        # Prepare the flow
        echo -e "${BLUE}Preparing flow...${NC}"
        if ! aws bedrock prepare-flow \
            --flow-identifier $FLOW_ID \
            --region $REGION \
            2>/dev/null; then
            if ! handle_error "Bedrock Flow Preparation" "Failed to prepare Bedrock Flow" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                echo -e "${YELLOW}Continuing despite flow preparation failure...${NC}"
            fi
        else
            echo -e "${GREEN}Flow prepared successfully.${NC}"
            
            # Add a delay to ensure the flow preparation completes
            sleep 10
            
            # Create a flow version
            echo -e "${BLUE}Creating flow version...${NC}"
            FLOW_VERSION=$(aws bedrock create-flow-version \
                --flow-identifier $FLOW_ID \
                --region $REGION \
                --query 'flowVersion' --output text \
                2>/dev/null || echo "")
                
            if [ -z "$FLOW_VERSION" ]; then
                if ! handle_error "Bedrock Flow Version" "Failed to create flow version" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                    echo -e "${YELLOW}Continuing with DRAFT version...${NC}"
                    FLOW_VERSION="DRAFT"
                fi
            else
                echo -e "${GREEN}Created flow version: $FLOW_VERSION${NC}"
                
                # Create a flow alias
                echo -e "${BLUE}Creating flow alias...${NC}"
                FLOW_ALIAS=$(aws bedrock create-flow-alias \
                    --flow-identifier $FLOW_ID \
                    --flow-alias-name "latest" \
                    --routing-configuration "[{\"flowVersion\":\"$FLOW_VERSION\"}]" \
                    --region $REGION \
                    --query 'flowAliasId' --output text \
                    2>/dev/null || echo "")
                    
                if [ -z "$FLOW_ALIAS" ]; then
                    if ! handle_error "Bedrock Flow Alias" "Failed to create flow alias" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                        echo -e "${YELLOW}Continuing without alias...${NC}"
                    fi
                else
                    echo -e "${GREEN}Created flow alias: $FLOW_ALIAS${NC}"
                fi
            fi
        fi

        # Update Lambda environment variables with flow ID
        echo -e "${BLUE}Updating Lambda environment variables with flow ID...${NC}"
        if ! aws lambda update-function-configuration \
            --function-name ${PROJECT_NAME}-orchestrator \
            --environment "Variables={REGION=$REGION,FLOW_ID=$FLOW_ID,FLOW_ALIAS=latest,CLAUDE_MODEL=$CLAUDE_MODEL}" \
            --region $REGION \
            2>/dev/null; then
            if ! handle_error "Lambda Configuration" "Failed to update Lambda environment variables" "Check Lambda function in the AWS console" "https://console.aws.amazon.com/lambda/home?region=$REGION#/functions/${PROJECT_NAME}-orchestrator?tab=configuration"; then
                echo -e "${YELLOW}Continuing despite Lambda environment update failure...${NC}"
            fi
        else
            echo -e "${GREEN}Lambda environment variables updated with flow ID.${NC}"
        fi

        create_checkpoint "$FLOW_CHECKPOINT"
    fi

    # Create web KB flow if web KB exists
    if [ ! -z "$WEB_KB_ID" ]; then
        echo -e "${BLUE}Creating Web Knowledge Base integrated flow...${NC}"
        
        WEB_KB_FLOW_ID=$(aws bedrock create-flow \
            --cli-input-json file://flow_templates/storybook_web_kb_flow.json \
            --region $REGION \
            --query 'flowId' --output text \
            2>/dev/null || echo "")

        if [ -z "$WEB_KB_FLOW_ID" ]; then
            echo -e "${YELLOW}Failed to create Web KB flow. You can create it manually in the Bedrock console.${NC}"
        else
            echo -e "${GREEN}Created Web KB flow with ID: $WEB_KB_FLOW_ID${NC}"
            
            # Prepare the Web KB flow
            echo -e "${BLUE}Preparing Web KB flow...${NC}"
            if ! aws bedrock prepare-flow \
                --flow-identifier $WEB_KB_FLOW_ID \
                --region $REGION \
                2>/dev/null; then
                echo -e "${YELLOW}Failed to prepare Web KB flow. You can prepare it manually in the Bedrock console.${NC}"
            else
                echo -e "${GREEN}Web KB flow prepared successfully.${NC}"
                
                # Add a delay
                sleep 10
                
                # Create a Web KB flow version
                echo -e "${BLUE}Creating Web KB flow version...${NC}"
                WEB_KB_FLOW_VERSION=$(aws bedrock create-flow-version \
                    --flow-identifier $WEB_KB_FLOW_ID \
                    --region $REGION \
                    --query 'flowVersion' --output text \
                    2>/dev/null || echo "")
                    
                if [ -z "$WEB_KB_FLOW_VERSION" ]; then
                    echo -e "${YELLOW}Failed to create Web KB flow version. You can create it manually in the Bedrock console.${NC}"
                else
                    echo -e "${GREEN}Created Web KB flow version: $WEB_KB_FLOW_VERSION${NC}"
                    
                    # Create a Web KB flow alias
                    echo -e "${BLUE}Creating Web KB flow alias...${NC}"
                    WEB_KB_FLOW_ALIAS=$(aws bedrock create-flow-alias \
                        --flow-identifier $WEB_KB_FLOW_ID \
                        --flow-alias-name "latest" \
                        --routing-configuration "[{\"flowVersion\":\"$WEB_KB_FLOW_VERSION\"}]" \
                        --region $REGION \
                        --query 'flowAliasId' --output text \
                        2>/dev/null || echo "")
                        
                    if [ -z "$WEB_KB_FLOW_ALIAS" ]; then
                        echo -e "${YELLOW}Failed to create Web KB flow alias. You can create it manually in the Bedrock console.${NC}"
                    else
                        echo -e "${GREEN}Created Web KB flow alias: $WEB_KB_FLOW_ALIAS${NC}"
                        
                        # Update Lambda environment variables with both flow IDs
                        echo -e "${BLUE}Updating Lambda environment variables with web KB flow ID...${NC}"
                        if ! aws lambda update-function-configuration \
                            --function-name ${PROJECT_NAME}-orchestrator \
                            --environment "Variables={REGION=$REGION,FLOW_ID=$FLOW_ID,FLOW_ALIAS=latest,WEB_KB_FLOW_ID=$WEB_KB_FLOW_ID,WEB_KB_FLOW_ALIAS=latest,CLAUDE_MODEL=$CLAUDE_MODEL}" \
                            --region $REGION \
                            2>/dev/null; then
                            echo -e "${YELLOW}Failed to update Lambda environment variables with Web KB flow ID.${NC}"
                        else
                            echo -e "${GREEN}Lambda environment variables updated with Web KB flow ID.${NC}"
                        fi
                    fi
                fi
            fi
        fi
    fi

    # If KB exists, create KB flow too
    if [ ! -z "$KB_ID" ]; then
        echo -e "${BLUE}Creating Knowledge Base integrated flow...${NC}"
        
        KB_FLOW_ID=$(aws bedrock create-flow \
            --cli-input-json file://flow_templates/storybook_kb_flow.json \
            --region $REGION \
            --query 'flowId' --output text \
            2>/dev/null || echo "")

        if [ -z "$KB_FLOW_ID" ]; then
            echo -e "${YELLOW}Failed to create KB flow. You can create it manually in the Bedrock console.${NC}"
        else
            echo -e "${GREEN}Created KB flow with ID: $KB_FLOW_ID${NC}"
            
            # Prepare the KB flow
            echo -e "${BLUE}Preparing KB flow...${NC}"
            if ! aws bedrock prepare-flow \
                --flow-identifier $KB_FLOW_ID \
                --region $REGION \
                2>/dev/null; then
                echo -e "${YELLOW}Failed to prepare KB flow. You can prepare it manually in the Bedrock console.${NC}"
            else
                echo -e "${GREEN}KB flow prepared successfully.${NC}"
                
                # Add a delay
                sleep 10
                
                # Create a KB flow version
                echo -e "${BLUE}Creating KB flow version...${NC}"
                KB_FLOW_VERSION=$(aws bedrock create-flow-version \
                    --flow-identifier $KB_FLOW_ID \
                    --region $REGION \
                    --query 'flowVersion' --output text \
                    2>/dev/null || echo "")
                    
                if [ -z "$KB_FLOW_VERSION" ]; then
                    echo -e "${YELLOW}Failed to create KB flow version. You can create it manually in the Bedrock console.${NC}"
                else
                    echo -e "${GREEN}Created KB flow version: $KB_FLOW_VERSION${NC}"
                    
                    # Create a KB flow alias
                    echo -e "${BLUE}Creating KB flow alias...${NC}"
                    KB_FLOW_ALIAS=$(aws bedrock create-flow-alias \
                        --flow-identifier $KB_FLOW_ID \
                        --flow-alias-name "latest" \
                        --routing-configuration "[{\"flowVersion\":\"$KB_FLOW_VERSION\"}]" \
                        --region $REGION \
                        --query 'flowAliasId' --output text \
                        2>/dev/null || echo "")
                        
                    if [ -z "$KB_FLOW_ALIAS" ]; then
                        echo -e "${YELLOW}Failed to create KB flow alias. You can create it manually in the Bedrock console.${NC}"
                    else
                        echo -e "${GREEN}Created KB flow alias: $KB_FLOW_ALIAS${NC}"
                        
                        # Update Lambda environment variables with flow IDs
                        echo -e "${BLUE}Updating Lambda environment variables with all flow IDs...${NC}"
                        ENV_VARS="REGION=$REGION,FLOW_ID=$FLOW_ID,FLOW_ALIAS=latest,KB_FLOW_ID=$KB_FLOW_ID,KB_FLOW_ALIAS=latest,CLAUDE_MODEL=$CLAUDE_MODEL"
                        if [ ! -z "$WEB_KB_FLOW_ID" ]; then
                            ENV_VARS="$ENV_VARS,WEB_KB_FLOW_ID=$WEB_KB_FLOW_ID,WEB_KB_FLOW_ALIAS=latest"
                        fi
                        if [ ! -z "$SUPERVISOR_AGENT_ID" ]; then
                            ENV_VARS="$ENV_VARS,AGENT_ID=$SUPERVISOR_AGENT_ID,AGENT_ALIAS_ID=$SUPERVISOR_AGENT_ALIAS_ID"
                        fi
                        
                        if ! aws lambda update-function-configuration \
                            --function-name ${PROJECT_NAME}-orchestrator \
                            --environment "Variables={$ENV_VARS}" \
                            --region $REGION \
                            2>/dev/null; then
                            echo -e "${YELLOW}Failed to update Lambda environment variables with all flow IDs.${NC}"
                        else
                            echo -e "${GREEN}Lambda environment variables updated with all flow IDs.${NC}"
                        fi
                    fi
                fi
            fi
        fi
    fi

    # Create agent-based flow if agents were created
    if [ ! -z "$SUPERVISOR_AGENT_ID" ]; then
        echo -e "${BLUE}Creating Agent-based flow...${NC}"
        
        # Update the template with actual agent IDs
        sed -i.bak "s/AGENT_ID_PLACEHOLDER/$SUPERVISOR_AGENT_ID/g" flow_templates/storybook_agent_flow.json
        sed -i.bak "s/AGENT_ALIAS_ID_PLACEHOLDER/$SUPERVISOR_AGENT_ALIAS_ID/g" flow_templates/storybook_agent_flow.json
        
        AGENT_FLOW_ID=$(aws bedrock create-flow \
            --cli-input-json file://flow_templates/storybook_agent_flow.json \
            --region $REGION \
            --query 'flowId' --output text \
            2>/dev/null || echo "")

        if [ -z "$AGENT_FLOW_ID" ]; then
            echo -e "${YELLOW}Failed to create Agent-based flow. You can create it manually in the Bedrock console.${NC}"
        else
            echo -e "${GREEN}Created Agent-based flow with ID: $AGENT_FLOW_ID${NC}"
            
            # Prepare the Agent flow
            echo -e "${BLUE}Preparing Agent flow...${NC}"
            if ! aws bedrock prepare-flow \
                --flow-identifier $AGENT_FLOW_ID \
                --region $REGION \
                2>/dev/null; then
                echo -e "${YELLOW}Failed to prepare Agent flow. You can prepare it manually in the Bedrock console.${NC}"
            else
                echo -e "${GREEN}Agent flow prepared successfully.${NC}"
                
                # Add a delay
                sleep 10
                
                # Create a Agent flow version
                echo -e "${BLUE}Creating Agent flow version...${NC}"
                AGENT_FLOW_VERSION=$(aws bedrock create-flow-version \
                    --flow-identifier $AGENT_FLOW_ID \
                    --region $REGION \
                    --query 'flowVersion' --output text \
                    2>/dev/null || echo "")
                    
                if [ -z "$AGENT_FLOW_VERSION" ]; then
                    echo -e "${YELLOW}Failed to create Agent flow version. You can create it manually in the Bedrock console.${NC}"
                else
                    echo -e "${GREEN}Created Agent flow version: $AGENT_FLOW_VERSION${NC}"
                    
                    # Create a Agent flow alias
                    echo -e "${BLUE}Creating Agent flow alias...${NC}"
                    AGENT_FLOW_ALIAS=$(aws bedrock create-flow-alias \
                        --flow-identifier $AGENT_FLOW_ID \
                        --flow-alias-name "latest" \
                        --routing-configuration "[{\"flowVersion\":\"$AGENT_FLOW_VERSION\"}]" \
                        --region $REGION \
                        --query 'flowAliasId' --output text \
                        2>/dev/null || echo "")
                        
                    if [ -z "$AGENT_FLOW_ALIAS" ]; then
                        echo -e "${YELLOW}Failed to create Agent flow alias. You can create it manually in the Bedrock console.${NC}"
                    else
                        echo -e "${GREEN}Created Agent flow alias: $AGENT_FLOW_ALIAS${NC}"
                        
                        # Update Lambda environment variables with flow IDs
                        echo -e "${BLUE}Updating Lambda environment variables with all flow IDs...${NC}"
                        ENV_VARS="REGION=$REGION,FLOW_ID=$FLOW_ID,FLOW_ALIAS=latest,AGENT_FLOW_ID=$AGENT_FLOW_ID,AGENT_FLOW_ALIAS=latest,CLAUDE_MODEL=$CLAUDE_MODEL,AGENT_ID=$SUPERVISOR_AGENT_ID,AGENT_ALIAS_ID=$SUPERVISOR_AGENT_ALIAS_ID"
                        if [ ! -z "$KB_FLOW_ID" ]; then
                            ENV_VARS="$ENV_VARS,KB_FLOW_ID=$KB_FLOW_ID,KB_FLOW_ALIAS=latest"
                        fi
                        if [ ! -z "$WEB_KB_FLOW_ID" ]; then
                            ENV_VARS="$ENV_VARS,WEB_KB_FLOW_ID=$WEB_KB_FLOW_ID,WEB_KB_FLOW_ALIAS=latest"
                        fi
                        
                        if ! aws lambda update-function-configuration \
                            --function-name ${PROJECT_NAME}-orchestrator \
                            --environment "Variables={$ENV_VARS}" \
                            --region $REGION \
                            2>/dev/null; then
                            echo -e "${YELLOW}Failed to update Lambda environment variables with all flow IDs.${NC}"
                        else
                            echo -e "${GREEN}Lambda environment variables updated with all flow IDs.${NC}"
                        fi
                    fi
                fi
            fi
        fi
    fi

    # Clean up temporary files
    echo -e "${BLUE}Cleaning up temporary files...${NC}"
    rm -f flow-trust-policy.json lambda-trust-policy.json bedrock-flow-policy.json lambda-policy.json kb-trust-policy.json agent-trust-policy.json agent-policy.json agent-kb-policy.json kb-web-crawling-policy.json
    rm -rf lambda kb_files flow_templates

    echo -e "${GREEN}=== storybook System deployed successfully! ===${NC}"

    # Check if any components failed
    FAILED_COMPONENTS=""

    if ! checkpoint_exists "$TABLES_CHECKPOINT"; then
        FAILED_COMPONENTS="$FAILED_COMPONENTS DynamoDB tables,"
    fi

    if ! checkpoint_exists "$IAM_CHECKPOINT"; then
        FAILED_COMPONENTS="$FAILED_COMPONENTS IAM roles,"
    fi

    if ! checkpoint_exists "$LAMBDA_CHECKPOINT"; then
        FAILED_COMPONENTS="$FAILED_COMPONENTS Lambda functions,"
    fi

    if ! checkpoint_exists "$API_CHECKPOINT"; then
        FAILED_COMPONENTS="$FAILED_COMPONENTS API Gateway,"
    fi

    if ! checkpoint_exists "$KB_CHECKPOINT"; then
        FAILED_COMPONENTS="$FAILED_COMPONENTS Knowledge Base,"
    fi

    if ! checkpoint_exists "$FLOW_CHECKPOINT"; then
        FAILED_COMPONENTS="$FAILED_COMPONENTS Bedrock Flows,"
    fi

    if ! checkpoint_exists "$AGENT_CHECKPOINT"; then
        FAILED_COMPONENTS="$FAILED_COMPONENTS Bedrock Agents,"
    fi

    if [ ! -z "$FAILED_COMPONENTS" ]; then
        FAILED_COMPONENTS=${FAILED_COMPONENTS%,}
        echo -e "${YELLOW}Some components may not have been successfully deployed:${NC} $FAILED_COMPONENTS"
        echo -e "${YELLOW}Check the AWS console to confirm these components and manually configure if needed.${NC}"
    fi

    echo -e "${GREEN}Use the API Gateway endpoint to interact with the system.${NC}"
    if [ ! -z "$API_ID" ]; then
        echo -e "${GREEN}API Endpoint: https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook${NC}"
    else
        echo -e "${YELLOW}API Gateway endpoint not created. You'll need to manually set up the API Gateway.${NC}"
    fi

    echo -e "${GREEN}Example curl command to create a new project:${NC}"
    echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"operation\":\"create_project\",\"title\":\"My Novel\",\"synopsis\":\"A thrilling story about...\",\"genre\":[\"thriller\"],\"target_audience\":[\"young adult\"]}' https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook"

    echo -e "${GREEN}Example curl command to invoke a flow:${NC}"
    echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"operation\":\"invoke_flow\",\"project_id\":\"YOUR_PROJECT_ID\",\"task\":\"Create an initial project plan for the novel\"}' https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook"

    if [ ! -z "$SUPERVISOR_AGENT_ID" ]; then
        echo -e "${GREEN}Example curl command to invoke an agent:${NC}"
        echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"operation\":\"invoke_agent\",\"agent_id\":\"$SUPERVISOR_AGENT_ID\",\"input_text\":\"Create an initial project plan for my novel about a space adventure\"}' https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook"
    fi

    if [ ! -z "$WEB_KB_ID" ]; then
        echo -e "${GREEN}Example curl command to perform a web search:${NC}"
        echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"operation\":\"web_search\",\"query\":\"modern science fiction novel techniques\"}' https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook"
    fi

    return 0
}

# Update the flows
update_flows() {
    if ! check_prerequisites; then
        echo -e "${RED}Prerequisites check failed. Cannot update flows.${NC}"
        return 1
    fi

    echo -e "${BLUE}=== Updating storybook flows ===${NC}"
    echo -e "${BLUE}Project: $PROJECT_NAME${NC}"
    echo -e "${BLUE}Region: $REGION${NC}"

    # Get list of flows
    FLOWS=$(aws bedrock list-flows --region $REGION --query 'flowSummaries[*].[flowId,name]' --output text 2>/dev/null || echo "Failed to list flows")

    if [[ "$FLOWS" == "Failed to list flows" || -z "$FLOWS" ]]; then
        if ! handle_error "Flow Listing" "Failed to list flows or no flows found" "Check if flows were created and if you have permission to list them" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows"; then
            return 1
        fi
    fi

    # Show list of flows
    echo -e "${BLUE}Available flows:${NC}"
    echo "$FLOWS" | nl

    read -p "Enter the number of the flow to update (0 to cancel): " flow_num

    if [ "$flow_num" -eq 0 ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return 0
    fi

    # Get the selected flow
    SELECTED_FLOW=$(echo "$FLOWS" | sed -n "${flow_num}p")
    FLOW_ID=$(echo "$SELECTED_FLOW" | awk '{print $1}')
    FLOW_NAME=$(echo "$SELECTED_FLOW" | awk '{print $2}')

    if [ -z "$FLOW_ID" ]; then
        echo -e "${RED}Invalid selection.${NC}"
        return 1
    fi

    echo -e "${BLUE}Selected flow: $FLOW_NAME (ID: $FLOW_ID)${NC}"

    # Get FLOW_ROLE_ARN
    FLOW_ROLE_NAME="${PROJECT_NAME}-flowrole"
    FLOW_ROLE_ARN=$(aws iam get-role --role-name ${FLOW_ROLE_NAME} --query 'Role.Arn' --output text 2>/dev/null)

    # Create a temporary directory for the updated flow definition
    mkdir -p temp_flow

    # Get the flow definition
    echo -e "${BLUE}Retrieving flow definition...${NC}"
    FLOW_DEF=$(aws bedrock get-flow --flow-identifier $FLOW_ID --region $REGION --query 'definition' 2>/dev/null || echo "Failed to get flow definition")

    if [[ "$FLOW_DEF" == "Failed to get flow definition" ]]; then
        if ! handle_error "Flow Definition" "Failed to get flow definition" "Check if the flow exists and if you have permission to view it" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
            return 1
        fi
    fi

    # Create a complete flow definition
    cat > temp_flow/updated_flow.json << EOF
{
    "name": "$FLOW_NAME",
    "description": "Updated storybook workflow using Bedrock Flows",
    "executionRoleArn": "${FLOW_ROLE_ARN}",
    "definition": $FLOW_DEF
}
EOF

    # Allow the user to edit the flow definition
    echo -e "${BLUE}Opening flow definition for editing. Save and exit when done.${NC}"
    if command -v nano &> /dev/null; then
        nano temp_flow/updated_flow.json
    elif command -v vi &> /dev/null; then
        vi temp_flow/updated_flow.json
    else
        echo -e "${RED}No text editor (nano or vi) found.${NC}"
        rm -rf temp_flow
        return 1
    fi

    # Confirm update
    read -p "Update the flow with the modified definition? (y/n): " confirm

    if [[ $confirm =~ ^[Yy]$ ]]; then
        # Update the flow
        echo -e "${BLUE}Updating flow...${NC}"
        if ! aws bedrock update-flow \
            --flow-identifier $FLOW_ID \
            --cli-input-json file://temp_flow/updated_flow.json \
            --region $REGION \
            2>/dev/null; then
            if ! handle_error "Flow Update" "Failed to update flow" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                rm -rf temp_flow
                return 1
            fi
        else
            echo -e "${GREEN}Flow updated successfully.${NC}"
            
            # Prepare the updated flow
            echo -e "${BLUE}Preparing updated flow...${NC}"
            if ! aws bedrock prepare-flow \
                --flow-identifier $FLOW_ID \
                --region $REGION \
                2>/dev/null; then
                if ! handle_error "Flow Preparation" "Failed to prepare updated flow" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                    echo -e "${YELLOW}Continuing despite flow preparation failure...${NC}"
                fi
            else
                echo -e "${GREEN}Flow prepared successfully.${NC}"
                
                # Add a delay
                sleep 10
                
                # Create a new flow version
                echo -e "${BLUE}Creating new flow version...${NC}"
                FLOW_VERSION=$(aws bedrock create-flow-version \
                    --flow-identifier $FLOW_ID \
                    --region $REGION \
                    --query 'flowVersion' --output text \
                    2>/dev/null || echo "")
                    
                if [ -z "$FLOW_VERSION" ]; then
                    if ! handle_error "Flow Version" "Failed to create new flow version" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                        echo -e "${YELLOW}Continuing with DRAFT version...${NC}"
                    fi
                else
                    echo -e "${GREEN}Created new flow version: $FLOW_VERSION${NC}"
                    
                    # Update or create flow alias
                    echo -e "${BLUE}Checking for existing flow aliases...${NC}"
                    FLOW_ALIASES=$(aws bedrock list-flow-aliases \
                        --flow-identifier $FLOW_ID \
                        --region $REGION \
                        --query 'flowAliasSummaries[*].[flowAliasId,flowAliasName]' --output text \
                        2>/dev/null || echo "")
                        
                    if [ -z "$FLOW_ALIASES" ]; then
                        # Create new alias
                        echo -e "${BLUE}Creating new flow alias...${NC}"
                        FLOW_ALIAS=$(aws bedrock create-flow-alias \
                            --flow-identifier $FLOW_ID \
                            --flow-alias-name "latest" \
                            --routing-configuration "[{\"flowVersion\":\"$FLOW_VERSION\"}]" \
                            --region $REGION \
                            --query 'flowAliasId' --output text \
                            2>/dev/null || echo "")
                            
                        if [ -z "$FLOW_ALIAS" ]; then
                            if ! handle_error "Flow Alias" "Failed to create flow alias" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                                echo -e "${YELLOW}Continuing without alias...${NC}"
                            fi
                        else
                            echo -e "${GREEN}Created flow alias: $FLOW_ALIAS${NC}"
                        fi
                    else
                        # Show available aliases
                        echo -e "${BLUE}Available aliases:${NC}"
                        echo "$FLOW_ALIASES" | nl
                        
                        read -p "Enter the number of the alias to update (0 to skip): " alias_num
                        
                        if [ "$alias_num" -eq 0 ]; then
                            echo -e "${YELLOW}Skipping alias update.${NC}"
                        else
                            # Get the selected alias
                            SELECTED_ALIAS=$(echo "$FLOW_ALIASES" | sed -n "${alias_num}p")
                            ALIAS_ID=$(echo "$SELECTED_ALIAS" | awk '{print $1}')
                            ALIAS_NAME=$(echo "$SELECTED_ALIAS" | awk '{print $2}')
                            
                            if [ -z "$ALIAS_ID" ]; then
                                echo -e "${RED}Invalid alias selection.${NC}"
                            else
                                # Update the alias
                                echo -e "${BLUE}Updating alias $ALIAS_NAME to use version $FLOW_VERSION...${NC}"
                                if ! aws bedrock update-flow-alias \
                                    --flow-identifier $FLOW_ID \
                                    --flow-alias-identifier $ALIAS_ID \
                                    --routing-configuration "[{\"flowVersion\":\"$FLOW_VERSION\"}]" \
                                    --region $REGION \
                                    2>/dev/null; then
                                    if ! handle_error "Flow Alias Update" "Failed to update flow alias" "Check flow status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows/$FLOW_ID"; then
                                        echo -e "${YELLOW}Continuing without alias update...${NC}"
                                    fi
                                else
                                    echo -e "${GREEN}Alias $ALIAS_NAME updated to use version $FLOW_VERSION.${NC}"
                                fi
                            fi
                        fi
                    fi
                }
            }
        }
    else
        echo -e "${YELLOW}Flow update cancelled.${NC}"
    fi

    # Clean up temporary files
    rm -rf temp_flow

    return 0
}

# Update agents
update_agents() {
    if ! check_prerequisites; then
        echo -e "${RED}Prerequisites check failed. Cannot update agents.${NC}"
        return 1
    fi

    echo -e "${BLUE}=== Updating storybook agents ===${NC}"
    echo -e "${BLUE}Project: $PROJECT_NAME${NC}"
    echo -e "${BLUE}Region: $REGION${NC}"

    # Get list of agents
    AGENTS=$(aws bedrock-agent list-agents --region $REGION --query 'agentSummaries[*].[agentId,agentName]' --output text 2>/dev/null || echo "Failed to list agents")

    if [[ "$AGENTS" == "Failed to list agents" || -z "$AGENTS" ]]; then
        if ! handle_error "Agent Listing" "Failed to list agents or no agents found" "Check if agents were created and if you have permission to list them" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/agents"; then
            return 1
        fi
    fi

    # Show list of agents
    echo -e "${BLUE}Available agents:${NC}"
    echo "$AGENTS" | nl

    read -p "Enter the number of the agent to update (0 to cancel): " agent_num

    if [ "$agent_num" -eq 0 ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return 0
    fi

    # Get the selected agent
    SELECTED_AGENT=$(echo "$AGENTS" | sed -n "${agent_num}p")
    AGENT_ID=$(echo "$SELECTED_AGENT" | awk '{print $1}')
    AGENT_NAME=$(echo "$SELECTED_AGENT" | awk '{print $2}')

    if [ -z "$AGENT_ID" ]; then
        echo -e "${RED}Invalid selection.${NC}"
        return 1
    fi

    echo -e "${BLUE}Selected agent: $AGENT_NAME (ID: $AGENT_ID)${NC}"

    # Get the agent details
    echo -e "${BLUE}Retrieving agent details...${NC}"
    AGENT_DETAILS=$(aws bedrock-agent get-agent --agent-id $AGENT_ID --region $REGION 2>/dev/null || echo "Failed to get agent details")

    if [[ "$AGENT_DETAILS" == "Failed to get agent details" ]]; then
        if ! handle_error "Agent Details" "Failed to get agent details" "Check if the agent exists and if you have permission to view it" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/agents/$AGENT_ID"; then
            return 1
        fi
    fi

    # Extract current instruction
    CURRENT_INSTRUCTION=$(echo "$AGENT_DETAILS" | jq -r '.agent.instruction')

    # Create a temporary file for editing the instruction
    mkdir -p temp_agent
    echo "$CURRENT_INSTRUCTION" > temp_agent/instruction.txt

    # Allow the user to edit the instruction
    echo -e "${BLUE}Opening agent instruction for editing. Save and exit when done.${NC}"
    if command -v nano &> /dev/null; then
        nano temp_agent/instruction.txt
    elif command -v vi &> /dev/null; then
        vi temp_agent/instruction.txt
    else
        echo -e "${RED}No text editor (nano or vi) found.${NC}"
        rm -rf temp_agent
        return 1
    fi

    # Read the updated instruction
    NEW_INSTRUCTION=$(cat temp_agent/instruction.txt)

    # Confirm update
    read -p "Update the agent with the modified instruction? (y/n): " confirm

    if [[ $confirm =~ ^[Yy]$ ]]; then
        # Update the agent
        echo -e "${BLUE}Updating agent...${NC}"
        if ! aws bedrock-agent update-agent \
            --agent-id $AGENT_ID \
            --instruction "$NEW_INSTRUCTION" \
            --region $REGION \
            2>/dev/null; then
            if ! handle_error "Agent Update" "Failed to update agent" "Check agent status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/agents/$AGENT_ID"; then
                rm -rf temp_agent
                return 1
            fi
        else
            echo -e "${GREEN}Agent updated successfully.${NC}"
            
            # Prepare the updated agent
            echo -e "${BLUE}Preparing updated agent...${NC}"
            if ! aws bedrock-agent prepare-agent \
                --agent-id $AGENT_ID \
                --region $REGION \
                2>/dev/null; then
                if ! handle_error "Agent Preparation" "Failed to prepare updated agent" "Check agent status in the Bedrock console" "https://console.aws.amazon.com/bedrock/home?region=$REGION#/agents/$AGENT_ID"; then
                    echo -e "${YELLOW}Continuing despite agent preparation failure...${NC}"
                fi
            else
                echo -e "${GREEN}Agent prepared successfully.${NC}"
                echo -e "${YELLOW}You may need to create a new agent alias for the updated agent.${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Agent update cancelled.${NC}"
    fi

    # Clean up temporary files
    rm -rf temp_agent

    return 0
}

# Delete the entire deployment
delete_deployment() {
    if ! check_prerequisites; then
        echo -e "${RED}Prerequisites check failed. Cannot delete deployment.${NC}"
        return 1
    fi

    echo -e "${RED}=== DANGER: Deleting storybook system ===${NC}"
    echo -e "${RED}Project: $PROJECT_NAME${NC}"
    echo -e "${RED}Region: $REGION${NC}"
    echo -e "${RED}THIS WILL DELETE ALL RESOURCES ASSOCIATED WITH THE PROJECT!${NC}"
    echo
    read -p "Type 'DELETE' to confirm deletion: " confirm

    if [ "$confirm" != "DELETE" ]; then
        echo -e "${YELLOW}Deletion cancelled.${NC}"
        return 0
    fi

    # Get account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
    BUCKET_NAME="${PROJECT_NAME}-assets-${ACCOUNT_ID}"

    echo -e "${BLUE}Starting deletion process...${NC}"

    # Delete Bedrock Agents
    echo -e "${BLUE}Deleting Bedrock Agents...${NC}"
    AGENTS=$(aws bedrock-agent list-agents --region $REGION --query 'agentSummaries[*].agentId' --output text 2>/dev/null || echo "")

    if [[ ! -z "$AGENTS" ]]; then
        for AGENT_ID in $AGENTS; do
            echo -e "${BLUE}Processing agent $AGENT_ID...${NC}"

            # Get aliases
            ALIASES=$(aws bedrock-agent list-agent-aliases --agent-id $AGENT_ID --region $REGION --query 'agentAliasSummaries[*].agentAliasId' --output text 2>/dev/null || echo "")

            if [[ ! -z "$ALIASES" ]]; then
                for ALIAS_ID in $ALIASES; do
                    echo -e "${BLUE}Deleting agent alias $ALIAS_ID...${NC}"
                    if ! aws bedrock-agent delete-agent-alias --agent-id $AGENT_ID --agent-alias-id $ALIAS_ID --region $REGION 2>/dev/null; then
                        echo -e "${YELLOW}Failed to delete agent alias $ALIAS_ID. You may need to delete it manually.${NC}"
                    fi
                    
                    # Add a short delay between deleting aliases
                    sleep 2
                done
            fi

            # Get collaborators
            COLLABORATORS=$(aws bedrock-agent list-agent-collaborators --agent-id $AGENT_ID --agent-version "DRAFT" --region $REGION --query 'collaborators[*].collaboratorId' --output text 2>/dev/null || echo "")

            if [[ ! -z "$COLLABORATORS" ]]; then
                for COLLABORATOR_ID in $COLLABORATORS; do
                    echo -e "${BLUE}Disassociating collaborator $COLLABORATOR_ID...${NC}"
                    if ! aws bedrock-agent disassociate-agent-collaborator --agent-id $AGENT_ID --agent-version "DRAFT" --collaborator-id $COLLABORATOR_ID --region $REGION 2>/dev/null; then
                        echo -e "${YELLOW}Failed to disassociate collaborator $COLLABORATOR_ID. You may need to disassociate it manually.${NC}"
                    fi
                    
                    # Add a short delay
                    sleep 2
                done
            fi

            # Get knowledge bases
            KBS=$(aws bedrock-agent list-agent-knowledge-bases --agent-id $AGENT_ID --agent-version "DRAFT" --region $REGION --query 'agentKnowledgeBases[*].knowledgeBaseId' --output text 2>/dev/null || echo "")

            if [[ ! -z "$KBS" ]]; then
                for KB_ID in $KBS; do
                    echo -e "${BLUE}Disassociating knowledge base $KB_ID...${NC}"
                    if ! aws bedrock-agent disassociate-agent-knowledge-base --agent-id $AGENT_ID --agent-version "DRAFT" --knowledge-base-id $KB_ID --region $REGION 2>/dev/null; then
                        echo -e "${YELLOW}Failed to disassociate knowledge base $KB_ID. You may need to disassociate it manually.${NC}"
                    fi
                    
                    # Add a short delay
                    sleep 2
                done
            fi

            # Delete the agent
            echo -e "${BLUE}Deleting agent $AGENT_ID...${NC}"
            if ! aws bedrock-agent delete-agent --agent-id $AGENT_ID --region $REGION 2>/dev/null; then
                echo -e "${YELLOW}Failed to delete agent $AGENT_ID. You may need to delete it manually.${NC}"
            fi
            
            # Add a delay between deleting agents
            sleep 5
        done
    else
        echo -e "${YELLOW}No agents found or unable to list agents.${NC}"
    fi

    # Delete Bedrock Flows
    echo -e "${BLUE}Deleting Bedrock Flows...${NC}"
    FLOWS=$(aws bedrock list-flows --region $REGION --query 'flowSummaries[*].flowId' --output text 2>/dev/null || echo "")

    if [[ ! -z "$FLOWS" ]]; then
        for FLOW_ID in $FLOWS; do
            echo -e "${BLUE}Processing flow $FLOW_ID...${NC}"

            # Get aliases
            ALIASES=$(aws bedrock list-flow-aliases --flow-identifier $FLOW_ID --region $REGION --query 'flowAliasSummaries[*].flowAliasId' --output text 2>/dev/null || echo "")

            if [[ ! -z "$ALIASES" ]]; then
                for ALIAS_ID in $ALIASES; do
                    echo -e "${BLUE}Deleting flow alias $ALIAS_ID...${NC}"
                    if ! aws bedrock delete-flow-alias --flow-identifier $FLOW_ID --flow-alias-identifier $ALIAS_ID --region $REGION 2>/dev/null; then
                        echo -e "${YELLOW}Failed to delete flow alias $ALIAS_ID. You may need to delete it manually.${NC}"
                    fi
                    
                    # Add a short delay between deleting aliases
                    sleep 2
                done
            fi

            # Get versions
            VERSIONS=$(aws bedrock list-flow-versions --flow-identifier $FLOW_ID --region $REGION --query 'flowVersionSummaries[*].flowVersion' --output text 2>/dev/null || echo "")

            if [[ ! -z "$VERSIONS" ]]; then
                for VERSION in $VERSIONS; do
                    if [[ "$VERSION" != "DRAFT" ]]; then
                        echo -e "${BLUE}Deleting flow version $VERSION...${NC}"
                        if ! aws bedrock delete-flow-version --flow-identifier $FLOW_ID --flow-version $VERSION --region $REGION 2>/dev/null; then
                            echo -e "${YELLOW}Failed to delete flow version $VERSION. You may need to delete it manually.${NC}"
                        fi
                        
                        # Add a short delay between deleting versions
                        sleep 2
                    fi
                done
            fi

            # Delete the flow
            echo -e "${BLUE}Deleting flow $FLOW_ID...${NC}"
            if ! aws bedrock delete-flow --flow-identifier $FLOW_ID --region $REGION 2>/dev/null; then
                echo -e "${YELLOW}Failed to delete flow $FLOW_ID. You may need to delete it manually.${NC}"
            }
            
            # Add a delay between deleting flows
            sleep 5
        done
    else
        echo -e "${YELLOW}No flows found or unable to list flows.${NC}"
    fi

    # Delete knowledge bases
    echo -e "${BLUE}Deleting Knowledge Bases...${NC}"
    KBS=$(aws bedrock-agent list-knowledge-bases --region $REGION --query 'knowledgeBaseSummaries[*].knowledgeBaseId' --output text 2>/dev/null || echo "")

    if [[ ! -z "$KBS" ]]; then
        for KB_ID in $KBS; do
            echo -e "${BLUE}Processing knowledge base $KB_ID...${NC}"

            # Get data sources
            DS=$(aws bedrock-agent list-data-sources --knowledge-base-id $KB_ID --region $REGION --query 'dataSourceSummaries[*].dataSourceId' --output text 2>/dev/null || echo "")

            if [[ ! -z "$DS" ]]; then
                for DS_ID in $DS; do
                    echo -e "${BLUE}Deleting data source $DS_ID...${NC}"
                    if ! aws bedrock-agent delete-data-source --knowledge-base-id $KB_ID --data-source-id $DS_ID --region $REGION 2>/dev/null; then
                        echo -e "${YELLOW}Failed to delete data source $DS_ID. You may need to delete it manually.${NC}"
                    fi
                    
                    # Add a short delay
                    sleep 2
                done
            fi

            # Delete the knowledge base
            echo -e "${BLUE}Deleting knowledge base $KB_ID...${NC}"
            if ! aws bedrock-agent delete-knowledge-base --knowledge-base-id $KB_ID --region $REGION 2>/dev/null; then
                echo -e "${YELLOW}Failed to delete knowledge base $KB_ID. You may need to delete it manually.${NC}"
            fi
            
            # Add a delay
            sleep 5
        done
    else
        echo -e "${YELLOW}No knowledge bases found or unable to list knowledge bases.${NC}"
    fi

    # Delete vector collections
    echo -e "${BLUE}Deleting OpenSearch collections...${NC}"
    COLLECTIONS=$(aws opensearchserverless list-collections --region $REGION --query 'collectionSummaries[?starts_with(name, `'${PROJECT_NAME}'`)].id' --output text 2>/dev/null || echo "")

    if [[ ! -z "$COLLECTIONS" ]]; then
        for COLLECTION_ID in $COLLECTIONS; do
            echo -e "${BLUE}Deleting collection $COLLECTION_ID...${NC}"
            if ! aws opensearchserverless delete-collection --id $COLLECTION_ID --region $REGION 2>/dev/null; then
                echo -e "${YELLOW}Failed to delete collection $COLLECTION_ID. You may need to delete it manually.${NC}"
            fi
            
            # Add a delay
            sleep 5
        done
    else
        echo -e "${YELLOW}No OpenSearch collections found or unable to list collections.${NC}"
    fi

    # Delete API Gateway
    echo -e "${BLUE}Deleting API Gateway...${NC}"
    API_IDS=$(aws apigateway get-rest-apis --region $REGION --query "items[?name=='${PROJECT_NAME}-api'].id" --output text 2>/dev/null || echo "")

    if [[ ! -z "$API_IDS" ]]; then
        for API_ID in $API_IDS; do
            echo -e "${BLUE}Deleting API Gateway $API_ID...${NC}"

            # Add retry logic with exponential backoff
            RETRY_COUNT=0
            MAX_RETRIES=5
            INITIAL_WAIT=2

            while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
                if aws apigateway delete-rest-api --rest-api-id $API_ID --region $REGION 2>/dev/null; then
                    echo -e "${GREEN}Successfully deleted API Gateway $API_ID${NC}"
                    break
                else
                    RETRY_COUNT=$((RETRY_COUNT + 1))
                    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
                        echo -e "${YELLOW}Failed to delete API Gateway $API_ID after $MAX_RETRIES attempts. You may need to delete it manually.${NC}"
                    else
                        WAIT_TIME=$((INITIAL_WAIT * 2 ** (RETRY_COUNT - 1)))
                        echo -e "${YELLOW}Attempt $RETRY_COUNT failed. Waiting $WAIT_TIME seconds before retry...${NC}"
                        sleep $WAIT_TIME
                    fi
                fi
            done

            # Add a delay between deleting different APIs to avoid rate limiting
            sleep 5
        done
    else
        echo -e "${YELLOW}No API Gateways found or unable to list APIs.${NC}"
    fi

    # Delete Lambda functions
    echo -e "${BLUE}Deleting Lambda functions...${NC}"
    
    # Delete orchestrator Lambda
    if ! aws lambda delete-function --function-name ${PROJECT_NAME}-orchestrator --region $REGION 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete Lambda function ${PROJECT_NAME}-orchestrator. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted Lambda function ${PROJECT_NAME}-orchestrator${NC}"
    fi
    
    # Delete data processor Lambda
    if ! aws lambda delete-function --function-name ${PROJECT_NAME}-data-processor --region $REGION 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete Lambda function ${PROJECT_NAME}-data-processor. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted Lambda function ${PROJECT_NAME}-data-processor${NC}"
    fi

    # Delete DynamoDB tables
    echo -e "${BLUE}Deleting DynamoDB tables...${NC}"
    TABLES="storybook-projects storybook-chapters storybook-characters storybook-agents"
    for TABLE in $TABLES; do
        if ! aws dynamodb delete-table --table-name $TABLE --region $REGION 2>/dev/null; then
            echo -e "${YELLOW}Failed to delete DynamoDB table $TABLE. You may need to delete it manually.${NC}"
        else
            echo -e "${GREEN}Deleted DynamoDB table $TABLE${NC}"
        fi
    done

    # Empty and delete S3 bucket with versioning support
    echo -e "${BLUE}Emptying and deleting S3 bucket $BUCKET_NAME...${NC}"

    # Check if bucket exists
    if aws s3api head-bucket --bucket $BUCKET_NAME 2>/dev/null; then
        # Delete all versions and delete markers
        echo -e "${BLUE}Removing all object versions...${NC}"
        VERSIONS=$(aws s3api list-object-versions \
            --bucket $BUCKET_NAME \
            --output=json \
            --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' 2>/dev/null)

        if [ $? -eq 0 ] && [ ! -z "$VERSIONS" ] && [ "$VERSIONS" != "{}" ]; then
            echo "$VERSIONS" > delete_versions.json
            if ! aws s3api delete-objects \
                --bucket $BUCKET_NAME \
                --delete file://delete_versions.json 2>/dev/null; then
                echo -e "${YELLOW}Failed to delete some object versions. You may need to clean up the bucket manually.${NC}"
            fi
            rm delete_versions.json
        fi

        # Delete any delete markers
        echo -e "${BLUE}Removing delete markers...${NC}"
        DELETE_MARKERS=$(aws s3api list-object-versions \
            --bucket $BUCKET_NAME \
            --output=json \
            --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' 2>/dev/null)

        if [ $? -eq 0 ] && [ ! -z "$DELETE_MARKERS" ] && [ "$DELETE_MARKERS" != "{}" ]; then
            echo "$DELETE_MARKERS" > delete_markers.json
            if ! aws s3api delete-objects \
                --bucket $BUCKET_NAME \
                --delete file://delete_markers.json 2>/dev/null; then
                echo -e "${YELLOW}Failed to delete some markers. You may need to clean up the bucket manually.${NC}"
            fi
            rm delete_markers.json
        fi

        # Remove any remaining objects (as fallback)
        if ! aws s3 rm s3://$BUCKET_NAME --recursive --force 2>/dev/null; then
            echo -e "${YELLOW}Failed to empty S3 bucket completely. You may need to delete remaining objects manually.${NC}"
        fi

        # Delete the bucket
        echo -e "${BLUE}Deleting S3 bucket...${NC}"
        if ! aws s3api delete-bucket --bucket $BUCKET_NAME --region $REGION 2>/dev/null; then
            echo -e "${YELLOW}Failed to delete S3 bucket $BUCKET_NAME. You may need to delete it manually.${NC}"
        else
            echo -e "${GREEN}Deleted S3 bucket $BUCKET_NAME${NC}"
        fi
    else
        echo -e "${YELLOW}S3 bucket $BUCKET_NAME not found or cannot be accessed.${NC}"
    fi

    # Delete IAM roles and policies
    echo -e "${BLUE}Deleting IAM roles and policies...${NC}"

    # Agent role
    AGENT_ROLE_NAME="${PROJECT_NAME}-agentrole"
    AGENT_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-agent-policy"
    AGENT_KB_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-agent-kb-policy"
    
    if ! aws iam detach-role-policy --role-name $AGENT_ROLE_NAME --policy-arn $AGENT_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to detach agent policy from agent role.${NC}"
    fi
    
    if ! aws iam detach-role-policy --role-name $AGENT_ROLE_NAME --policy-arn $AGENT_KB_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to detach agent KB policy from agent role.${NC}"
    fi
    
    if ! aws iam delete-role --role-name $AGENT_ROLE_NAME 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete IAM role $AGENT_ROLE_NAME. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted IAM role $AGENT_ROLE_NAME${NC}"
    fi

    # Flow role
    FLOW_ROLE_NAME="${PROJECT_NAME}-flowrole"
    FLOW_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-flow-policy"
    
    if ! aws iam detach-role-policy --role-name $FLOW_ROLE_NAME --policy-arn $FLOW_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to detach flow policy from flow role.${NC}"
    fi
    
    if ! aws iam delete-role --role-name $FLOW_ROLE_NAME 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete IAM role $FLOW_ROLE_NAME. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted IAM role $FLOW_ROLE_NAME${NC}"
    fi

    # Lambda role
    LAMBDA_ROLE_NAME="${PROJECT_NAME}-lambdarole"
    LAMBDA_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-lambda-policy"
    
    if ! aws iam detach-role-policy --role-name $LAMBDA_ROLE_NAME --policy-arn $LAMBDA_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to detach Lambda policy from Lambda role.${NC}"
    fi
    
    if ! aws iam detach-role-policy --role-name $LAMBDA_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null; then
        echo -e "${YELLOW}Failed to detach Lambda execution policy from Lambda role.${NC}"
    fi
    
    if ! aws iam delete-role --role-name $LAMBDA_ROLE_NAME 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete IAM role $LAMBDA_ROLE_NAME. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted IAM role $LAMBDA_ROLE_NAME${NC}"
    fi

    # Knowledge Base role
    KB_ROLE_NAME="${PROJECT_NAME}-kbrole"
    KB_WEBCRAWL_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-kb-webcrawl-policy"
    
    if ! aws iam detach-role-policy --role-name $KB_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess 2>/dev/null; then
        echo -e "${YELLOW}Failed to detach S3 read policy from Knowledge Base role.${NC}"
    fi

    if ! aws iam detach-role-policy --role-name $KB_ROLE_NAME --policy-arn $KB_WEBCRAWL_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to detach web crawling policy from Knowledge Base role.${NC}"
    fi

    if ! aws iam delete-role --role-name $KB_ROLE_NAME 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete IAM role $KB_ROLE_NAME. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted IAM role $KB_ROLE_NAME${NC}"
    fi

    # Delete custom policies
    if ! aws iam delete-policy --policy-arn $FLOW_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete flow policy. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted custom flow policy${NC}"
    fi
    
    if ! aws iam delete-policy --policy-arn $LAMBDA_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete Lambda policy. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted custom Lambda policy${NC}"
    fi

    if ! aws iam delete-policy --policy-arn $AGENT_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete agent policy. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted custom agent policy${NC}"
    fi

    if ! aws iam delete-policy --policy-arn $AGENT_KB_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete agent KB policy. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted custom agent KB policy${NC}"
    fi

    if ! aws iam delete-policy --policy-arn $KB_WEBCRAWL_POLICY_ARN 2>/dev/null; then
        echo -e "${YELLOW}Failed to delete KB web crawling policy. You may need to delete it manually.${NC}"
    else
        echo -e "${GREEN}Deleted custom KB web crawling policy${NC}"
    fi

    # Delete checkpoint files
    echo -e "${BLUE}Removing checkpoint files...${NC}"
    rm -rf "$CHECKPOINT_DIR"

    echo -e "${GREEN}Deletion completed.${NC}"
    echo -e "${YELLOW}Some resources might still be in the process of being deleted.${NC}"
    echo -e "${YELLOW}Check the AWS Console to ensure all resources have been properly deleted:${NC}"
    echo -e "1. DynamoDB tables: https://console.aws.amazon.com/dynamodb/home?region=$REGION#tables:"
    echo -e "2. IAM roles: https://console.aws.amazon.com/iamv2/home?region=$REGION#/roles"
    echo -e "3. Lambda functions: https://console.aws.amazon.com/lambda/home?region=$REGION#/functions"
    echo -e "4. API Gateway: https://console.aws.amazon.com/apigateway/main/apis?region=$REGION"
    echo -e "5. S3 buckets: https://s3.console.aws.amazon.com/s3/home?region=$REGION"
    echo -e "6. Bedrock flows: https://console.aws.amazon.com/bedrock/home?region=$REGION#/flows"
    echo -e "7. Bedrock agents: https://console.aws.amazon.com/bedrock/home?region=$REGION#/agents"
    echo -e "8. Knowledge Bases: https://console.aws.amazon.com/bedrock/home?region=$REGION#/knowledge-bases"

    return 0
}

# Main menu
show_menu() {
    clear
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}      Storybook AWS Bedrock Deployment      ${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo -e "Project: ${GREEN}$PROJECT_NAME${NC}"
    echo -e "Region: ${GREEN}$REGION${NC}"
    echo -e "Model: ${GREEN}$CLAUDE_MODEL${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo
    echo "1. Create New Deployment"
    echo "2. Update Existing Flows"
    echo "3. Update Existing Agents"
    echo "4. Delete Deployment"
    echo "5. Configuration Settings"
    echo "6. Exit"
    echo
    read -p "Select an option (1-6): " choice

    case $choice in
        1)
            deploy_system
            read -p "Press Enter to return to the main menu..."
            ;;
        2)
            update_flows
            read -p "Press Enter to return to the main menu..."
            ;;
        3)
            update_agents
            read -p "Press Enter to return to the main menu..."
            ;;
        4)
            delete_deployment
            read -p "Press Enter to return to the main menu..."
            ;;
        5)
            update_config
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice."
            read -p "Press Enter to continue..."
            ;;
    esac

# Initialize and run script
load_config
setup_checkpoints  # This ensures the checkpoint directory exists before we try to create any checkpoints
while true; do
    show_menu
done
