#!/bin/bash

# AWS Bedrock Hierarchical Multi-Agent Manuscript Editing System
# Minimized version that pulls resources from GitHub

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Load configuration from config.sh
source config.sh

# Global variables
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Resource names
FLOW_BUCKET_NAME="storybook-flow-${ACCOUNT_ID}"
ROLE_NAME="storybook-role"
DYNAMODB_TABLE="storybook-state"
KNOWLEDGE_BASE_NAME="storybook-knowledge"
LAMBDA_RESEARCH_NAME="storybook-research"
LAMBDA_WORKFLOW_NAME="storybook-workflow"

# Agent IDs and Alias IDs storage
declare -A AGENT_IDS
declare -A AGENT_ALIAS_IDS

# Debug function to print AWS CLI commands before execution
log() {
    echo -e "${YELLOW}LOG: $@${NC}"
}

# Function to download a file from GitHub
download_from_github() {
    local path=$1
    local output=$2

    log "Downloading $path from GitHub..."
    curl -s -o "$output" "${GITHUB_REPO}/$path"

    if [ $? -ne 0 ]; then
        log "Failed to download $path from GitHub"
        return 1
    fi

    log "Downloaded $path successfully"
    return 0
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."

    if ! command -v aws &> /dev/null; then
        log "AWS CLI is not installed. Please install it and try again."
        exit 1
    fi

    AWS_VERSION=$(aws --version | cut -d' ' -f1 | cut -d'/' -f2)
    if [[ $(echo -e "$AWS_VERSION\n2.24" | sort -V | head -n1) != "2.24" ]]; then
        log "AWS CLI version is $AWS_VERSION, but 2.24 or higher is required."
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log "jq is not installed. Please install it and try again."
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        log "curl is not installed. Please install it and try again."
        exit 1
    fi

    log "All dependencies found."
}

# Check if the script is run with necessary permissions
check_permissions() {
    if [ "$EUID" -ne 0 ]; then
        log "Please run as root or with sudo."
        exit 1
    fi
}

# Check if AWS CLI is configured with the correct profile
check_aws_profile() {
    if ! aws sts get-caller-identity &> /dev/null; then
        log "AWS CLI is not configured correctly. Please configure it and try again."
        exit 1
    fi
}

# Cleanup function to remove temporary files
cleanup() {
    log "Cleaning up temporary files..."
    rm -f trust-policy.json editor-policy-template.json prompt-list.txt
    rm -rf prompts
    rm -f agent-definitions.json collaborations.json knowledge_base_info.txt
    rm -f lambda/research_function.zip lambda/workflow_function.zip
    log "Cleanup completed."
}

# Retry mechanism for AWS CLI commands
retry_aws_command() {
    local retries=3
    local count=0
    local delay=5

    until [ $count -ge $retries ]; do
        "$@" && break
        count=$((count + 1))
        log "Retrying in $delay seconds..."
        sleep $delay
    done

    if [ $count -ge $retries ]; then
        log "Command failed after $retries attempts."
        return 1
    fi
}

# Delete IAM role if it exists
delete_iam_role_if_exists() {
    local role_name=$1
    log "Checking if IAM role $role_name exists..."

    if aws iam get-role --role-name $role_name &>/dev/null; then
        log "Role $role_name exists. Deleting..."

        # Get all attached policies
        local policies=$(aws iam list-attached-role-policies --role-name $role_name --query 'AttachedPolicies[*].PolicyArn' --output text)

        # Detach all policies
        for policy in $policies; do
            log "Detaching policy $policy from role $role_name..."
            retry_aws_command aws iam detach-role-policy --role-name $role_name --policy-arn $policy 2>/dev/null || true
        done

        # Try to delete the custom policy if it exists
        retry_aws_command aws iam delete-policy --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy 2>/dev/null || true

        # Delete the role
        retry_aws_command aws iam delete-role --role-name $role_name 2>/dev/null || true

        log "Role $role_name deleted."
        sleep 10
    else
        log "Role $role_name does not exist. Proceeding with creation."
    fi
}

# Create IAM role for the system
create_iam_role() {
    log "Creating IAM role for the Manuscript Editing System..."
    delete_iam_role_if_exists $ROLE_NAME

    # Download policies from GitHub
    download_from_github "config/trust-policy.json" "trust-policy.json"
    download_from_github "config/editor-policy-template.json" "editor-policy-template.json"

    # Replace placeholders in the policy
    sed -i "s/FLOW_BUCKET_NAME/${FLOW_BUCKET_NAME}/g; s/REGION/${REGION}/g; s/ACCOUNT_ID/${ACCOUNT_ID}/g; s/DYNAMODB_TABLE/${DYNAMODB_TABLE}/g; s/LAMBDA_RESEARCH_NAME/${LAMBDA_RESEARCH_NAME}/g; s/LAMBDA_WORKFLOW_NAME/${LAMBDA_WORKFLOW_NAME}/g" editor-policy-template.json

    # Create the role
    retry_aws_command aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json

    # Create and attach policy
    retry_aws_command aws iam create-policy --policy-name ManuscriptEditorPolicy --policy-document file://editor-policy-template.json 2>/dev/null || true
    retry_aws_command aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy

    log "Waiting for role to propagate..."
    sleep 10
    log "IAM role created successfully."

    rm -f trust-policy.json editor-policy-template.json
}

# Create S3 bucket and DynamoDB functions (minimized version)
delete_s3_bucket_if_exists() {
    local bucket_name=$1
    if aws s3api head-bucket --bucket $bucket_name 2>/dev/null; then
        log "S3 bucket $bucket_name exists. Deleting..."
        retry_aws_command aws s3 rm s3://${bucket_name} --recursive --region $REGION 2>/dev/null || true
        retry_aws_command aws s3api delete-bucket --bucket ${bucket_name} --region $REGION 2>/dev/null || true
        sleep 5
    fi
}

create_s3_bucket() {
    log "Creating S3 bucket for system resources..."
    delete_s3_bucket_if_exists $FLOW_BUCKET_NAME
    retry_aws_command aws s3 mb s3://${FLOW_BUCKET_NAME} --region $REGION
    retry_aws_command aws s3api put-bucket-versioning --bucket ${FLOW_BUCKET_NAME} --versioning-configuration Status=Enabled
}

delete_dynamodb_table_if_exists() {
    if aws dynamodb describe-table --table-name ${DYNAMODB_TABLE} --region ${REGION} &>/dev/null; then
        log "DynamoDB table ${DYNAMODB_TABLE} exists. Deleting..."
        retry_aws_command aws dynamodb delete-table --table-name ${DYNAMODB_TABLE} --region ${REGION}
        retry_aws_command aws dynamodb wait table-not-exists --table-name ${DYNAMODB_TABLE} --region ${REGION}
    fi
}

create_dynamodb_table() {
    log "Creating DynamoDB table for workflow state..."
    delete_dynamodb_table_if_exists
    retry_aws_command aws dynamodb create-table --table-name ${DYNAMODB_TABLE} --attribute-definitions AttributeName=ManuscriptId,AttributeType=S --key-schema AttributeName=ManuscriptId,KeyType=HASH --billing-mode PAY_PER_REQUEST --region ${REGION}
    log "Waiting for DynamoDB table creation to complete..."
    retry_aws_command aws dynamodb wait table-exists --table-name ${DYNAMODB_TABLE} --region ${REGION}
}

# Upload agent prompts to S3 (pulled from GitHub)
upload_prompts() {
    log "Downloading agent prompts from GitHub and uploading to S3..."
    mkdir -p prompts

    # Download prompt list file
    download_from_github "config/prompt-list.txt" "prompt-list.txt"

    # Read the prompt list and download each prompt
    while IFS= read -r line; do
        agent_name=$(echo "$line" | cut -d':' -f1)
        file_path=$(echo "$line" | cut -d':' -f2)
        download_from_github "prompts/${file_path}" "prompts/${file_path}"
    done < prompt-list.txt

    # Upload all prompts to S3
    retry_aws_command aws s3 sync prompts/ s3://${FLOW_BUCKET_NAME}/prompts/ --region $REGION
    log "Agent prompts uploaded successfully."

    rm -f prompt-list.txt
    rm -rf prompts
}

# Create the knowledge base
create_knowledge_base() {
    log "Creating knowledge base for manuscript context..."

    # Check if knowledge base exists and delete if necessary
    KB_EXISTS=$(aws bedrock list-knowledge-bases --region $REGION | jq -r ".knowledgeBaseSummaries[] | select(.name == \"$KNOWLEDGE_BASE_NAME\") | .knowledgeBaseId" 2>/dev/null)
    if [ ! -z "$KB_EXISTS" ]; then
        log "Knowledge base $KNOWLEDGE_BASE_NAME already exists. Deleting..."
        retry_aws_command aws bedrock delete-knowledge-base --knowledge-base-id $KB_EXISTS --region $REGION
        sleep 30
    fi

    # Download config templates
    download_from_github "config/kb-config-template.json" "kb-config-template.json"
    download_from_github "config/access-policy-template.json" "access-policy-template.json"

    # Replace placeholders
    sed -i "s/KNOWLEDGE_BASE_NAME/${KNOWLEDGE_BASE_NAME}/g; s/REGION/${REGION}/g; s/ACCOUNT_ID/${ACCOUNT_ID}/g; s/ROLE_NAME/${ROLE_NAME}/g" kb-config-template.json
    sed -i "s/ACCOUNT_ID/${ACCOUNT_ID}/g; s/ROLE_NAME/${ROLE_NAME}/g" access-policy-template.json

    # Create OpenSearch collection
    log "Creating OpenSearch Serverless collection..."
    retry_aws_command aws opensearchserverless create-collection --name storybook --type VECTORSEARCH --region $REGION
    log "Waiting for OpenSearch collection to be created..."
    retry_aws_command aws opensearchserverless wait collection-active --name storybook --region $REGION

    # Create access policy
    retry_aws_command aws opensearchserverless create-access-policy --cli-input-json file://access-policy-template.json --region $REGION

    # Create knowledge base
    KB_RESPONSE=$(retry_aws_command aws bedrock create-knowledge-base --cli-input-json file://kb-config-template.json --region $REGION)
    KB_ID=$(echo $KB_RESPONSE | jq -r '.knowledgeBase.knowledgeBaseId')
    log "Knowledge base created with ID: $KB_ID"

    # Wait for knowledge base to be ready
    log "Waiting for knowledge base to be active..."
    while true; do
        KB_STATUS=$(aws bedrock get-knowledge-base --knowledge-base-id $KB_ID --region $REGION | jq -r '.knowledgeBase.status')
        if [ "$KB_STATUS" == "ACTIVE" ]; then break; fi
        log "Knowledge base status: $KB_STATUS. Waiting..."
        sleep 10
    done

    # Save knowledge base ID
    echo "KB_ID=$KB_ID" > knowledge_base_info.txt

    # Clean up
    rm -f kb-config-template.json access-policy-template.json
}

# Create Lambda functions
create_lambda_functions() {
    log "Creating Lambda functions..."
    mkdir -p lambda

    # Download Lambda code from GitHub
    download_from_github "lambda/research_function.py" "lambda/research_function.py"
    download_from_github "lambda/workflow_function.py" "lambda/workflow_function.py"

    # Create deployment packages
    cd lambda
    zip -q research_function.zip research_function.py
    zip -q workflow_function.zip workflow_function.py
    cd ..

    # Delete existing functions if they exist
    if aws lambda get-function --function-name $LAMBDA_RESEARCH_NAME --region $REGION &>/dev/null; then
        retry_aws_command aws lambda delete-function --function-name $LAMBDA_RESEARCH_NAME --region $REGION
    fi

    if aws lambda get-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION &>/dev/null; then
        retry_aws_command aws lambda delete-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION
    fi

    # Create the Lambda functions
    retry_aws_command aws lambda create-function --function-name $LAMBDA_RESEARCH_NAME --runtime python3.9 --handler research_function.lambda_handler --role arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME} --zip-file fileb://lambda/research_function.zip --timeout 120 --environment "Variables={STATE_TABLE=$DYNAMODB_TABLE}" --region $REGION

    retry_aws_command aws lambda create-function --function-name $LAMBDA_WORKFLOW_NAME --runtime python3.9 --handler workflow_function.lambda_handler --role arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME} --zip-file fileb://lambda/workflow_function.zip --timeout 120 --environment "Variables={STATE_TABLE=$DYNAMODB_TABLE}" --region $REGION

    rm -rf lambda
}

# Agent management functions (these need to remain in the script)
delete_agent_if_exists() {
    local agent_name=$1
    local existing_id=""
    local action=""
    local new_name=""

    log "Checking if agent '$agent_name' already exists..."
    LIST_RESPONSE=$(aws bedrock-agent list-agents --region $REGION 2>/dev/null)

    if [ $? -eq 0 ]; then
        existing_id=$(echo $LIST_RESPONSE | jq -r ".agents[] | select(.displayName == \"$agent_name\") | .agentId" 2>/dev/null)

        if [ ! -z "$existing_id" ] && [ "$existing_id" != "null" ]; then
            log "Found existing agent '$agent_name' with ID: $existing_id."
            log "1. Delete existing agent and create a new one with the same name"
            log "2. Use a different name for the new agent"
            log "3. Cancel operation"
            read -p "Choose an option (1/2/3): " choice

            case $choice in
                1)
                    action="delete"
                    log "Deleting existing agent '$agent_name'..."
                    ALIAS_RESPONSE=$(aws bedrock-agent list-agent-aliases --agent-id $existing_id --region $REGION 2>/dev/null)

                    if [ $? -eq 0 ]; then
                        alias_ids=$(echo $ALIAS_RESPONSE | jq -r '.agentAliases[].agentAliasId' 2>/dev/null)
                        for alias_id in $alias_ids; do
                            retry_aws_command aws bedrock-agent delete-agent-alias --agent-id $existing_id --agent-alias-id $alias_id --region $REGION 2>/dev/null || true
                        done
                    fi
                    retry_aws_command aws bedrock-agent delete-agent --agent-id $existing_id --skip-resource-in-use-check --region $REGION 2>/dev/null || true
                    sleep 5
                    ;;
                2)
                    action="rename"
                    read -p "Enter a new name for the agent: " new_name
                    log "Will create agent with new name: '$new_name'"
                    ;;
                *)
                    action="cancel"
                    log "Operation cancelled."
                    ;;
            esac

            echo "action=$action" > /tmp/agent_action.txt
            echo "new_name=$new_name" >> /tmp/agent_action.txt

            if [ "$action" == "delete" ]; then return 0
            elif [ "$action" == "rename" ]; then return 2
            else return 1; fi
        fi
    fi
    return 0
}

wait_for_agent_creatable() {
    local agent_id=$1
    local max_attempts=30
    local delay=10
    local attempt=0

    if [ -z "$agent_id" ] || [ "$agent_id" == "null" ]; then
        log "Invalid agent ID provided."
        return 1
    fi

    log "Waiting for agent to be in a valid state..."

    while ((attempt < max_attempts)); do
        GET_RESPONSE=$(aws bedrock-agent get-agent --agent-id $agent_id --region $REGION 2>/dev/null)
        if [ $? -ne 0 ]; then
            log "Error getting agent status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)"
            sleep $delay
            attempt=$((attempt+1))
            continue
        fi

        agent_status=$(echo $GET_RESPONSE | jq -r '.agent.agentStatus' 2>/dev/null)
        if [ -z "$agent_status" ] || [ "$agent_status" == "null" ]; then
            log "Could not extract agent status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)"
            sleep $delay
            attempt=$((attempt+1))
            continue
        fi

        if [[ "$agent_status" != "CREATING" ]]; then
            log "Agent is now in a valid state: $agent_status"
            return 0
        fi

        log "Agent status: $agent_status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)"
        sleep $delay
        attempt=$((attempt+1))
    done

    log "Timed out waiting for agent to be in a valid state."
    return 1
}

wait_for_agent() {
    local agent_id=$1
    local max_attempts=30
    local delay=10
    local attempt=0

    if [ -z "$agent_id" ] || [ "$agent_id" == "null" ]; then
        log "Invalid agent ID provided."
        return 1
    fi

    log "Waiting for agent to be ready..."

    while ((attempt < max_attempts)); do
        GET_RESPONSE=$(aws bedrock-agent get-agent --agent-id $agent_id --region $REGION 2>/dev/null)
        if [ $? -ne 0 ]; then
            log "Error getting agent status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)"
            sleep $delay
            attempt=$((attempt+1))
            continue
        fi

        agent_status=$(echo $GET_RESPONSE | jq -r '.agent.agentStatus' 2>/dev/null)
        if [[ "$agent_status" == "PREPARED" || "$agent_status" == "AVAILABLE" ]]; then
            log "Agent is now available with status: $agent_status"
            return 0
        fi

        log "Agent status: $agent_status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)"
        sleep $delay
        attempt=$((attempt+1))
    done

    log "Timed out waiting for agent to be ready. Last status: $agent_status"
    return 1
}

# Create agent function (using prompt from S3)
create_agent() {
    local agent_name=$1
    local agent_desc=$2
    local prompt_s3_path=$3
    local retry_count=0
    local max_retries=3
    local original_name=$agent_name

    log "Creating agent: $agent_name..."

    while [ $retry_count -lt $max_retries ]; do
        delete_agent_if_exists "$agent_name"
        local delete_result=$?

        if [ $delete_result -eq 1 ]; then
            log "Agent creation cancelled by user."
            return 1
        elif [ $delete_result -eq 2 ]; then
            if [ -f /tmp/agent_action.txt ]; then
                source /tmp/agent_action.txt
                if [ ! -z "$new_name" ]; then
                    agent_name="$new_name"
                    log "Using new name: $agent_name"
                fi
                rm -f /tmp/agent_action.txt
            fi
        fi

        # Get prompt from S3
        retry_aws_command aws s3 cp s3://${FLOW_BUCKET_NAME}/${prompt_s3_path} prompt_content.md --region $REGION

        if [ ! -f prompt_content.md ]; then
            log "Failed to download prompt file from S3."
            return 1
        fi

        prompt_content=$(cat prompt_content.md)

        # Create agent with Claude 3 Sonnet and collaboration mode enabled
        RESPONSE=$(retry_aws_command aws bedrock-agent create-agent \
            --agent-name "$agent_name" \
            --agent-resource-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}" \
            --instruction "$prompt_content" \
            --foundation-model "$MODEL_ID" \
            --description "$agent_desc" \
            --idle-session-ttl-in-seconds 3600 \
            --agent-collaboration "ENABLED" \
            --region $REGION)

        if [ $? -ne 0 ]; then
            log "Failed to create agent $agent_name."
            retry_count=$((retry_count + 1))
            if [ $retry_count -ge $max_retries ]; then
                log "Maximum retries reached. Could not create agent."
                rm -f prompt_content.md
                return 1
            fi
            sleep 5
            continue
        fi

        # Extract agent ID
        AGENT_ID=$(echo $RESPONSE | jq -r '.agent.agentId')

        if [ -z "$AGENT_ID" ] || [ "$AGENT_ID" == "null" ]; then
            log "Failed to extract agent ID for $agent_name."
            retry_count=$((retry_count + 1))
            if [ $retry_count -ge $max_retries ]; then
                log "Maximum retries reached. Could not extract agent ID."
                rm -f prompt_content.md
                return 1
            fi
            sleep 5
            continue
        fi

        AGENT_IDS["$original_name"]=$AGENT_ID
        log "Created agent: $agent_name with ID: $AGENT_ID"

        # Wait for agent to be in a valid state for preparation
        wait_for_agent_creatable $AGENT_ID
        if [ $? -ne 0 ]; then
            log "Agent $agent_name is not in a valid state for preparation."
            retry_count=$((retry_count + 1))
            continue
        fi

        # Prepare the agent with retry logic
        local prepare_attempts=5
        local prepare_delay=15
        local prepare_attempt=0
        local prepare_success=false

        while ((prepare_attempt < prepare_attempts)) && [[ "$prepare_success" == "false" ]]; do
            log "Attempting to prepare agent (attempt $((prepare_attempt+1))/$prepare_attempts)..."
            if retry_aws_command aws bedrock-agent prepare-agent --agent-id $AGENT_ID --region $REGION 2>/dev/null; then
                prepare_success=true
                log "Successfully prepared agent."
            else
                log "Failed to prepare agent. Waiting $prepare_delay seconds before retry..."
                sleep $prepare_delay
                prepare_attempt=$((prepare_attempt+1))
            fi
        done

        if [[ "$prepare_success" == "false" ]]; then
            log "Failed to prepare agent after $prepare_attempts attempts."
            retry_count=$((retry_count + 1))
            continue
        fi

        # Wait for agent to be ready
        wait_for_agent $AGENT_ID
        if [ $? -ne 0 ]; then
            log "Agent $agent_name did not become ready."
            retry_count=$((retry_count + 1))
            continue
        fi

        # Create agent alias with the created version
        ALIAS_RESPONSE=$(retry_aws_command aws bedrock-agent create-agent-alias \
            --agent-id $AGENT_ID \
            --agent-alias-name "Production" \
            --region $REGION)

        if [ $? -ne 0 ]; then
            log "Failed to create alias for agent $agent_name."
            retry_count=$((retry_count + 1))
            continue
        fi

        # Extract alias ID
        ALIAS_ID=$(echo $ALIAS_RESPONSE | jq -r '.agentAlias.agentAliasId')
        if [ -z "$ALIAS_ID" ] || [ "$ALIAS_ID" == "null" ]; then
            log "Failed to extract alias ID for $agent_name."
            retry_count=$((retry_count + 1))
            continue
        fi

        # Store the alias ID
        AGENT_ALIAS_IDS["$original_name"]=$ALIAS_ID
        log "Created alias for $agent_name with ID: $ALIAS_ID"
        log "Agent $agent_name is ready"

        # Clean up
        rm -f prompt_content.md
        break
    done

    if [ $retry_count -ge $max_retries ]; then
        log "Maximum retries reached for agent $original_name."
        return 1
    fi
    return 0
}

# Helper function for collaboration
setup_collaboration() {
    local source_agent=$1
    local target_agent=$2
    local instruction=$3

    local source_id=${AGENT_IDS["$source_agent"]}
    local source_alias_id=${AGENT_ALIAS_IDS["$source_agent"]}
    local target_id=${AGENT_IDS["$target_agent"]}
    local target_alias_id=${AGENT_ALIAS_IDS["$target_agent"]}

    if [ -z "$source_id" ] || [ -z "$source_alias_id" ] || [ -z "$target_id" ] || [ -z "$target_alias_id" ]; then
        log "Cannot set up collaboration between $source_agent and $target_agent. Missing agent IDs."
        return 1
    fi

    log "Setting up collaboration: $source_agent -> $target_agent"

    retry_aws_command aws bedrock-agent associate-agent-collaborator \
        --agent-id $source_id \
        --agent-version "DRAFT" \
        --collaborator-name "$target_agent" \
        --agent-descriptor "aliasArn=arn:aws:bedrock:${REGION}:${ACCOUNT_ID}:agent-alias/${target_id}/${target_alias_id}" \
        --collaboration-instruction "$instruction" \
        --relay-conversation-history "TO_COLLABORATOR" \
        --region $REGION
}

# Create all agents using definitions from GitHub
create_agents() {
    log "Creating all agents for the manuscript editing system..."

    # Download agent definitions from GitHub
    download_from_github "config/agent-definitions.json" "agent-definitions.json"

    # Parse the agent definitions and create each agent
    AGENTS=$(jq -c '.agents[]' agent-definitions.json)
    for agent in $AGENTS; do
        NAME=$(echo $agent | jq -r '.name')
        DESC=$(echo $agent | jq -r '.description')
        PROMPT=$(echo $agent | jq -r '.prompt_path')

        create_agent "$NAME" "$DESC" "$PROMPT"
    done

    log "All agents created successfully."
    declare -p AGENT_IDS > agent_ids.txt
    declare -p AGENT_ALIAS_IDS > agent_alias_ids.txt
    rm -f agent-definitions.json
}

# Set up collaborations
setup_agent_collaborations() {
    log "Setting up collaborations between agents..."

    # Download collaboration configuration from GitHub
    download_from_github "config/collaborations.json" "collaborations.json"

    # Set up the collaborations based on the JSON file
    COLLABORATIONS=$(jq -c '.collaborations[]' collaborations.json)
    for collab in $COLLABORATIONS; do
        SOURCE=$(echo $collab | jq -r '.source')
        TARGET=$(echo $collab | jq -r '.target')
        INSTRUCTION=$(echo $collab | jq -r '.instruction')

        setup_collaboration "$SOURCE" "$TARGET" "$INSTRUCTION"
    done

    # Connect Executive Director with Knowledge Base if available
    if [ -f knowledge_base_info.txt ]; then
        source knowledge_base_info.txt
        if [ ! -z "$KB_ID" ]; then
            executive_director_id=${AGENT_IDS["ExecutiveDirector"]}
            log "Associating Knowledge Base with Executive Director agent..."
            retry_aws_command aws bedrock-agent associate-agent-knowledge-base \
                --agent-id $executive_director_id \
                --agent-version "DRAFT" \
                --knowledge-base-id $KB_ID \
                --description "Knowledge base for manuscript research and context" \
                --knowledge-base-state "ENABLED" \
                --region $REGION
        fi
    fi

    # Prepare the updated Executive Director agent with the collaborations
    log "Preparing updated Executive Director agent with collaborations..."
    executive_director_id=${AGENT_IDS["ExecutiveDirector"]}
    executive_director_alias_id=${AGENT_ALIAS_IDS["ExecutiveDirector"]}

    retry_aws_command aws bedrock-agent prepare-agent --agent-id $executive_director_id --region $REGION
    wait_for_agent $executive_director_id

    # Create a new version and update the alias
    VERSION_RESPONSE=$(retry_aws_command aws bedrock-agent create-agent-version --agent-id $executive_director_id --region $REGION)
    VERSION=$(echo $VERSION_RESPONSE | jq -r '.agentVersion')

    retry_aws_command aws bedrock-agent update-agent-alias \
        --agent-id $executive_director_id \
        --agent-alias-id $executive_director_alias_id \
        --routing-configuration "[{\"agentVersion\":\"$VERSION\"}]" \
        --region $REGION

    # Update Lambda function environment variables with agent IDs
    retry_aws_command aws lambda update-function-configuration \
        --function-name $LAMBDA_WORKFLOW_NAME \
        --environment "Variables={STATE_TABLE=$DYNAMODB_TABLE,EXECUTIVE_DIRECTOR_ID=$executive_director_id,EXECUTIVE_DIRECTOR_ALIAS_ID=$executive_director_alias_id}" \
        --region $REGION

    rm -f collaborations.json
}

# Main functions for resource creation/deletion
create_resources() {
    check_permissions
    check_dependencies
    check_aws_profile
    create_iam_role
    create_s3_bucket
    create_dynamodb_table
    upload_prompts
    create_knowledge_base
    create_lambda_functions
    create_agents
    setup_agent_collaborations
    cleanup

    log "All resources created successfully."
    log "Your Manuscript Editing system is ready to use."
}

delete_resources() {
    log "WARNING: This will delete all resources created for the Manuscript Editing system."
    read -p "Are you sure you want to proceed? (y/n): " confirm
    if [[ $confirm != "y" && $confirm != "Y" ]]; then
        log "Deletion cancelled."
        return
    fi

    # Load resource IDs if available
    if [ -f agent_ids.txt ]; then source agent_ids.txt; fi
    if [ -f agent_alias_ids.txt ]; then source agent_alias_ids.txt; fi
    if [ -f knowledge_base_info.txt ]; then source knowledge_base_info.txt; fi

    # Delete agents
    for agent_name in "${!AGENT_IDS[@]}"; do
        agent_id=${AGENT_IDS[$agent_name]}
        log "Deleting agent: $agent_name ($agent_id)"
        if [ ! -z "${AGENT_ALIAS_IDS[$agent_name]}" ]; then
            alias_id=${AGENT_ALIAS_IDS[$agent_name]}
            retry_aws_command aws bedrock-agent delete-agent-alias --agent-id $agent_id --agent-alias-id $alias_id --region $REGION 2>/dev/null || true
        fi
        retry_aws_command aws bedrock-agent delete-agent --agent-id $agent_id --skip-resource-in-use-check --region $REGION 2>/dev/null || true
    done

    # Delete knowledge base
    if [ ! -z "$KB_ID" ]; then
        retry_aws_command aws bedrock delete-knowledge-base --knowledge-base-id $KB_ID --region $REGION 2>/dev/null || true
    fi

    # Delete Lambda functions
    retry_aws_command aws lambda delete-function --function-name $LAMBDA_RESEARCH_NAME --region $REGION 2>/dev/null || true
    retry_aws_command aws lambda delete-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION 2>/dev/null || true

    # Delete DynamoDB table
    retry_aws_command aws dynamodb delete-table --table-name $DYNAMODB_TABLE --region $REGION 2>/dev/null || true

    # Delete S3 bucket
    retry_aws_command aws s3 rm s3://${FLOW_BUCKET_NAME} --recursive --region $REGION 2>/dev/null || true
    retry_aws_command aws s3api delete-bucket --bucket ${FLOW_BUCKET_NAME} --region $REGION 2>/dev/null || true

    # Delete OpenSearch Serverless collection
    retry_aws_command aws opensearchserverless delete-collection --name storybook --region $REGION 2>/dev/null || true
    retry_aws_command aws opensearchserverless delete-access-policy --name storybook-policy --type data --region $REGION 2>/dev/null || true

    # Delete IAM role
    retry_aws_command aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy 2>/dev/null || true
    retry_aws_command aws iam delete-policy --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy 2>/dev/null || true
    retry_aws_command aws iam delete-role --role-name $ROLE_NAME 2>/dev/null || true

    # Remove info files
    rm -f agent_ids.txt agent_alias_ids.txt knowledge_base_info.txt

    log "All resources deleted successfully."
}

# Show info and test functions
show_info() {
    log "Retrieving information about deployed resources..."

    # Display agent information
    if [ -f agent_ids.txt ]; then
        source agent_ids.txt
        log "Agent Information:"
        for agent_name in "${!AGENT_IDS[@]}"; do
            log "$agent_name: ${AGENT_IDS[$agent_name]}"
        done
    fi

    # Display other resource information
    if [ -f knowledge_base_info.txt ]; then
        source knowledge_base_info.txt
        log "Knowledge Base ID: $KB_ID"
    fi

    # Check resource existence
    if aws dynamodb describe-table --table-name $DYNAMODB_TABLE --region $REGION &>/dev/null; then
        log "DynamoDB Table: $DYNAMODB_TABLE exists"
    fi

    if aws lambda get-function --function-name $LAMBDA_RESEARCH_NAME --region $REGION &>/dev/null; then
        log "Lambda Function: $LAMBDA_RESEARCH_NAME exists"
    fi

    if aws lambda get-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION &>/dev/null; then
        log "Lambda Function: $LAMBDA_WORKFLOW_NAME exists"
    fi

    log "S3 Bucket: $FLOW_BUCKET_NAME"
}

test_system() {
    log "Testing the Manuscript Editing system..."

    # Check if workflow Lambda exists
    if ! aws lambda get-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION &>/dev/null; then
        log "Workflow Lambda function not found. Please deploy the system first."
        return 1
    fi

    # Download test manuscript from GitHub
    download_from_github "config/test_manuscript.json" "test_manuscript.json"

    # Invoke Lambda with the test manuscript
    log "Invoking workflow Lambda with test manuscript..."
    retry_aws_command aws lambda invoke --function-name $LAMBDA_WORKFLOW_NAME --payload file://test_manuscript.json --region $REGION output.json

    # Display the output
    log "Workflow Lambda invoked successfully. Output:"
    cat output.json

    # Extract manuscript ID for status check
    MANUSCRIPT_ID=$(cat output.json | jq -r '.manuscript_id')

    # Create status check template
    cat > check_status.json << EOF
{
  "action": "get_status",
  "manuscript_id": "$MANUSCRIPT_ID"
}
EOF

    log "To check the status of the editing process, use:"
    log "$(cat check_status.json)"

    # Clean up
    rm -f test_manuscript.json
}

# Usage function to display help information
usage() {
    echo -e "${BLUE}Usage: $0 [option]${NC}"
    echo -e "Options:"
    echo -e "  create   Create Manuscript Editing system"
    echo -e "  delete   Delete all resources"
    echo -e "  info     Show information about deployed resources"
    echo -e "  test     Test system with sample manuscript"
    echo -e "  help     Display this help message"
}

# Main menu
main_menu() {
    while true; do
        echo -e "\n${BLUE}=== AWS Bedrock Hierarchical Multi-Agent Manuscript Editing System ====${NC}"
        echo -e "1. Create Manuscript Editing system"
        echo -e "2. Delete all resources"
        echo -e "3. Show information about deployed resources"
        echo -e "4. Test system with sample manuscript"
        echo -e "5. Exit"

        read -p "Choose an option: " option

        case $option in
            1) create_resources ;;
            2) delete_resources ;;
            3) show_info ;;
            4) test_system ;;
            5) log "Exiting. Goodbye!"; exit 0 ;;
            *) log "Invalid option. Please try again." ;;
        esac
    done
}

# Start script
log "AWS Bedrock Hierarchical Multi-Agent Manuscript Editing System"
log "This script will deploy a complete system for editing manuscripts using AI agents"
main_menu
