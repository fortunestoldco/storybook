#!/bin/bash

# AWS Bedrock Hierarchical Multi-Agent Manuscript Editing System
# Minimized version that pulls resources from GitHub

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration - update this to your GitHub repository
GITHUB_REPO="https://raw.githubusercontent.com/your-username/storybook-editor/main"

# Global variables
REGION="us-east-1" # Default region - modify if needed
STACK_NAME="storybook"
MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0" # Default model
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
debug_aws_command() {
    echo -e "${YELLOW}AWS CLI Command: $@${NC}"
}

# Function to download a file from GitHub
download_from_github() {
    local path=$1
    local output=$2

    echo -e "${BLUE}Downloading $path from GitHub...${NC}"
    curl -s -o "$output" "${GITHUB_REPO}/$path"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download $path from GitHub${NC}"
        return 1
    fi

    echo -e "${GREEN}Downloaded $path successfully${NC}"
    return 0
}

# Check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    if ! command -v aws &> /dev/null; then
        echo -e "${RED}AWS CLI is not installed. Please install it and try again.${NC}"
        exit 1
    fi

    AWS_VERSION=$(aws --version | cut -d' ' -f1 | cut -d'/' -f2)
    if [[ $(echo -e "$AWS_VERSION\n2.24" | sort -V | head -n1) != "2.24" ]]; then
        echo -e "${RED}AWS CLI version is $AWS_VERSION, but 2.24 or higher is required.${NC}"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        echo -e "${RED}jq is not installed. Please install it and try again.${NC}"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        echo -e "${RED}curl is not installed. Please install it and try again.${NC}"
        exit 1
    fi

    echo -e "${GREEN}All dependencies found.${NC}"
}

# Delete IAM role if it exists
delete_iam_role_if_exists() {
    local role_name=$1
    echo -e "${BLUE}Checking if IAM role $role_name exists...${NC}"

    if aws iam get-role --role-name $role_name &>/dev/null; then
        echo -e "${YELLOW}Role $role_name exists. Deleting...${NC}"

        # Get all attached policies
        local policies=$(aws iam list-attached-role-policies --role-name $role_name --query 'AttachedPolicies[*].PolicyArn' --output text)

        # Detach all policies
        for policy in $policies; do
            echo -e "${YELLOW}Detaching policy $policy from role $role_name...${NC}"
            aws iam detach-role-policy --role-name $role_name --policy-arn $policy 2>/dev/null || true
        done

        # Try to delete the custom policy if it exists
        aws iam delete-policy --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy 2>/dev/null || true

        # Delete the role
        aws iam delete-role --role-name $role_name 2>/dev/null || true

        echo -e "${GREEN}Role $role_name deleted.${NC}"
        sleep 10
    else
        echo -e "${GREEN}Role $role_name does not exist. Proceeding with creation.${NC}"
    fi
}

# Create IAM role for the system
create_iam_role() {
    echo -e "${BLUE}Creating IAM role for the Manuscript Editing System...${NC}"
    delete_iam_role_if_exists $ROLE_NAME

    # Download policies from GitHub
    download_from_github "config/trust-policy.json" "trust-policy.json"
    download_from_github "config/editor-policy-template.json" "editor-policy-template.json"

    # Replace placeholders in the policy
    sed -i "s/FLOW_BUCKET_NAME/${FLOW_BUCKET_NAME}/g; s/REGION/${REGION}/g; s/ACCOUNT_ID/${ACCOUNT_ID}/g; s/DYNAMODB_TABLE/${DYNAMODB_TABLE}/g; s/LAMBDA_RESEARCH_NAME/${LAMBDA_RESEARCH_NAME}/g; s/LAMBDA_WORKFLOW_NAME/${LAMBDA_WORKFLOW_NAME}/g" editor-policy-template.json

    # Create the role
    aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json

    # Create and attach policy
    aws iam create-policy --policy-name ManuscriptEditorPolicy --policy-document file://editor-policy-template.json 2>/dev/null || true
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy

    echo -e "${YELLOW}Waiting for role to propagate...${NC}"
    sleep 10
    echo -e "${GREEN}IAM role created successfully.${NC}"

    rm -f trust-policy.json editor-policy-template.json
}

# Create S3 bucket and DynamoDB functions (minimized version)
delete_s3_bucket_if_exists() {
    local bucket_name=$1
    if aws s3api head-bucket --bucket $bucket_name 2>/dev/null; then
        echo -e "${YELLOW}S3 bucket $bucket_name exists. Deleting...${NC}"
        aws s3 rm s3://${bucket_name} --recursive --region $REGION 2>/dev/null || true
        aws s3api delete-bucket --bucket ${bucket_name} --region $REGION 2>/dev/null || true
        sleep 5
    fi
}

create_s3_bucket() {
    echo -e "${BLUE}Creating S3 bucket for system resources...${NC}"
    delete_s3_bucket_if_exists $FLOW_BUCKET_NAME
    aws s3 mb s3://${FLOW_BUCKET_NAME} --region $REGION
    aws s3api put-bucket-versioning --bucket ${FLOW_BUCKET_NAME} --versioning-configuration Status=Enabled
}

delete_dynamodb_table_if_exists() {
    if aws dynamodb describe-table --table-name ${DYNAMODB_TABLE} --region ${REGION} &>/dev/null; then
        echo -e "${YELLOW}DynamoDB table ${DYNAMODB_TABLE} exists. Deleting...${NC}"
        aws dynamodb delete-table --table-name ${DYNAMODB_TABLE} --region ${REGION}
        aws dynamodb wait table-not-exists --table-name ${DYNAMODB_TABLE} --region ${REGION}
    fi
}

create_dynamodb_table() {
    echo -e "${BLUE}Creating DynamoDB table for workflow state...${NC}"
    delete_dynamodb_table_if_exists
    aws dynamodb create-table --table-name ${DYNAMODB_TABLE} --attribute-definitions AttributeName=ManuscriptId,AttributeType=S --key-schema AttributeName=ManuscriptId,KeyType=HASH --billing-mode PAY_PER_REQUEST --region ${REGION}
    echo -e "${YELLOW}Waiting for DynamoDB table creation to complete...${NC}"
    aws dynamodb wait table-exists --table-name ${DYNAMODB_TABLE} --region ${REGION}
}

# Upload agent prompts to S3 (pulled from GitHub)
upload_prompts() {
    echo -e "${BLUE}Downloading agent prompts from GitHub and uploading to S3...${NC}"
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
    aws s3 sync prompts/ s3://${FLOW_BUCKET_NAME}/prompts/ --region $REGION
    echo -e "${GREEN}Agent prompts uploaded successfully.${NC}"

    rm -f prompt-list.txt
    rm -rf prompts
}

# Create the knowledge base
create_knowledge_base() {
    echo -e "${BLUE}Creating knowledge base for manuscript context...${NC}"

    # Check if knowledge base exists and delete if necessary
    KB_EXISTS=$(aws bedrock list-knowledge-bases --region $REGION | jq -r ".knowledgeBaseSummaries[] | select(.name == \"$KNOWLEDGE_BASE_NAME\") | .knowledgeBaseId" 2>/dev/null)
    if [ ! -z "$KB_EXISTS" ]; then
        echo -e "${YELLOW}Knowledge base $KNOWLEDGE_BASE_NAME already exists. Deleting...${NC}"
        aws bedrock delete-knowledge-base --knowledge-base-id $KB_EXISTS --region $REGION
        sleep 30
    fi

    # Download config templates
    download_from_github "config/kb-config-template.json" "kb-config-template.json"
    download_from_github "config/access-policy-template.json" "access-policy-template.json"

    # Replace placeholders
    sed -i "s/KNOWLEDGE_BASE_NAME/${KNOWLEDGE_BASE_NAME}/g; s/REGION/${REGION}/g; s/ACCOUNT_ID/${ACCOUNT_ID}/g; s/ROLE_NAME/${ROLE_NAME}/g" kb-config-template.json
    sed -i "s/ACCOUNT_ID/${ACCOUNT_ID}/g; s/ROLE_NAME/${ROLE_NAME}/g" access-policy-template.json

    # Create OpenSearch collection
    echo -e "${BLUE}Creating OpenSearch Serverless collection...${NC}"
    aws opensearchserverless create-collection --name storybook --type VECTORSEARCH --region $REGION
    echo -e "${YELLOW}Waiting for OpenSearch collection to be created...${NC}"
    aws opensearchserverless wait collection-active --name storybook --region $REGION

    # Create access policy
    aws opensearchserverless create-access-policy --cli-input-json file://access-policy-template.json --region $REGION

    # Create knowledge base
    KB_RESPONSE=$(aws bedrock create-knowledge-base --cli-input-json file://kb-config-template.json --region $REGION)
    KB_ID=$(echo $KB_RESPONSE | jq -r '.knowledgeBase.knowledgeBaseId')
    echo -e "${GREEN}Knowledge base created with ID: $KB_ID${NC}"

    # Wait for knowledge base to be ready
    echo -e "${YELLOW}Waiting for knowledge base to be active...${NC}"
    while true; do
        KB_STATUS=$(aws bedrock get-knowledge-base --knowledge-base-id $KB_ID --region $REGION | jq -r '.knowledgeBase.status')
        if [ "$KB_STATUS" == "ACTIVE" ]; then break; fi
        echo -e "${YELLOW}Knowledge base status: $KB_STATUS. Waiting...${NC}"
        sleep 10
    done

    # Save knowledge base ID
    echo "KB_ID=$KB_ID" > knowledge_base_info.txt

    # Clean up
    rm -f kb-config-template.json access-policy-template.json
}

# Create Lambda functions
create_lambda_functions() {
    echo -e "${BLUE}Creating Lambda functions...${NC}"
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
        aws lambda delete-function --function-name $LAMBDA_RESEARCH_NAME --region $REGION
    fi

    if aws lambda get-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION &>/dev/null; then
        aws lambda delete-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION
    fi

    # Create the Lambda functions
    aws lambda create-function --function-name $LAMBDA_RESEARCH_NAME --runtime python3.9 --handler research_function.lambda_handler --role arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME} --zip-file fileb://lambda/research_function.zip --timeout 120 --environment "Variables={STATE_TABLE=$DYNAMODB_TABLE}" --region $REGION

    aws lambda create-function --function-name $LAMBDA_WORKFLOW_NAME --runtime python3.9 --handler workflow_function.lambda_handler --role arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME} --zip-file fileb://lambda/workflow_function.zip --timeout 120 --environment "Variables={STATE_TABLE=$DYNAMODB_TABLE}" --region $REGION

    rm -rf lambda
}

# Agent management functions (these need to remain in the script)
delete_agent_if_exists() {
    local agent_name=$1
    local existing_id=""
    local action=""
    local new_name=""

    echo -e "${BLUE}Checking if agent '$agent_name' already exists...${NC}"
    LIST_RESPONSE=$(aws bedrock-agent list-agents --region $REGION 2>/dev/null)

    if [ $? -eq 0 ]; then
        existing_id=$(echo $LIST_RESPONSE | jq -r ".agents[] | select(.displayName == \"$agent_name\") | .agentId" 2>/dev/null)

        if [ ! -z "$existing_id" ] && [ "$existing_id" != "null" ]; then
            echo -e "${YELLOW}Found existing agent '$agent_name' with ID: $existing_id.${NC}"
            echo -e "1. Delete existing agent and create a new one with the same name"
            echo -e "2. Use a different name for the new agent"
            echo -e "3. Cancel operation"
            read -p "Choose an option (1/2/3): " choice

            case $choice in
                1)
                    action="delete"
                    echo -e "${YELLOW}Deleting existing agent '$agent_name'...${NC}"
                    ALIAS_RESPONSE=$(aws bedrock-agent list-agent-aliases --agent-id $existing_id --region $REGION 2>/dev/null)

                    if [ $? -eq 0 ]; then
                        alias_ids=$(echo $ALIAS_RESPONSE | jq -r '.agentAliases[].agentAliasId' 2>/dev/null)
                        for alias_id in $alias_ids; do
                            aws bedrock-agent delete-agent-alias --agent-id $existing_id --agent-alias-id $alias_id --region $REGION 2>/dev/null || true
                        done
                    fi
                    aws bedrock-agent delete-agent --agent-id $existing_id --skip-resource-in-use-check --region $REGION 2>/dev/null || true
                    sleep 5
                    ;;
                2)
                    action="rename"
                    read -p "Enter a new name for the agent: " new_name
                    echo -e "${GREEN}Will create agent with new name: '$new_name'${NC}"
                    ;;
                *)
                    action="cancel"
                    echo -e "${RED}Operation cancelled.${NC}"
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
        echo -e "${RED}Invalid agent ID provided.${NC}"
        return 1
    fi

    echo -e "${YELLOW}Waiting for agent to be in a valid state...${NC}"

    while ((attempt < max_attempts)); do
        GET_RESPONSE=$(aws bedrock-agent get-agent --agent-id $agent_id --region $REGION 2>/dev/null)
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Error getting agent status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)${NC}"
            sleep $delay
            attempt=$((attempt+1))
            continue
        fi

        agent_status=$(echo $GET_RESPONSE | jq -r '.agent.agentStatus' 2>/dev/null)
        if [ -z "$agent_status" ] || [ "$agent_status" == "null" ]; then
            echo -e "${YELLOW}Could not extract agent status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)${NC}"
            sleep $delay
            attempt=$((attempt+1))
            continue
        fi

        if [[ "$agent_status" != "CREATING" ]]; then
            echo -e "${GREEN}Agent is now in a valid state: $agent_status${NC}"
            return 0
        fi

        echo -e "${YELLOW}Agent status: $agent_status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)${NC}"
        sleep $delay
        attempt=$((attempt+1))
    done

    echo -e "${RED}Timed out waiting for agent to be in a valid state.${NC}"
    return 1
}

wait_for_agent() {
    local agent_id=$1
    local max_attempts=30
    local delay=10
    local attempt=0

    if [ -z "$agent_id" ] || [ "$agent_id" == "null" ]; then
        echo -e "${RED}Invalid agent ID provided.${NC}"
        return 1
    fi

    echo -e "${YELLOW}Waiting for agent to be ready...${NC}"

    while ((attempt < max_attempts)); do
        GET_RESPONSE=$(aws bedrock-agent get-agent --agent-id $agent_id --region $REGION 2>/dev/null)
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Error getting agent status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)${NC}"
            sleep $delay
            attempt=$((attempt+1))
            continue
        fi

        agent_status=$(echo $GET_RESPONSE | jq -r '.agent.agentStatus' 2>/dev/null)
        if [[ "$agent_status" == "PREPARED" || "$agent_status" == "AVAILABLE" ]]; then
            echo -e "${GREEN}Agent is now available with status: $agent_status${NC}"
            return 0
        fi

        echo -e "${YELLOW}Agent status: $agent_status. Waiting $delay seconds before retry (attempt $((attempt+1))/$max_attempts)${NC}"
        sleep $delay
        attempt=$((attempt+1))
    done

    echo -e "${RED}Timed out waiting for agent to be ready. Last status: $agent_status${NC}"
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

    echo -e "${BLUE}Creating agent: $agent_name...${NC}"

    while [ $retry_count -lt $max_retries ]; do
        delete_agent_if_exists "$agent_name"
        local delete_result=$?

        if [ $delete_result -eq 1 ]; then
            echo -e "${RED}Agent creation cancelled by user.${NC}"
            return 1
        elif [ $delete_result -eq 2 ]; then
            if [ -f /tmp/agent_action.txt ]; then
                source /tmp/agent_action.txt
                if [ ! -z "$new_name" ]; then
                    agent_name="$new_name"
                    echo -e "${BLUE}Using new name: $agent_name${NC}"
                fi
                rm -f /tmp/agent_action.txt
            fi
        fi

        # Get prompt from S3
        aws s3 cp s3://${FLOW_BUCKET_NAME}/${prompt_s3_path} prompt_content.md --region $REGION

        if [ ! -f prompt_content.md ]; then
            echo -e "${RED}Failed to download prompt file from S3.${NC}"
            return 1
        fi

        prompt_content=$(cat prompt_content.md)

        # Create agent with Claude 3 Sonnet and collaboration mode enabled
        RESPONSE=$(aws bedrock-agent create-agent \
            --agent-name "$agent_name" \
            --agent-resource-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}" \
            --instruction "$prompt_content" \
            --foundation-model "$MODEL_ID" \
            --description "$agent_desc" \
            --idle-session-ttl-in-seconds 3600 \
            --agent-collaboration "ENABLED" \
            --region $REGION)

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to create agent $agent_name.${NC}"
            retry_count=$((retry_count + 1))
            if [ $retry_count -ge $max_retries ]; then
                echo -e "${RED}Maximum retries reached. Could not create agent.${NC}"
                rm -f prompt_content.md
                return 1
            fi
            sleep 5
            continue
        fi

        # Extract agent ID
        AGENT_ID=$(echo $RESPONSE | jq -r '.agent.agentId')

        if [ -z "$AGENT_ID" ] || [ "$AGENT_ID" == "null" ]; then
            echo -e "${RED}Failed to extract agent ID for $agent_name.${NC}"
            retry_count=$((retry_count + 1))
            if [ $retry_count -ge $max_retries ]; then
                echo -e "${RED}Maximum retries reached. Could not extract agent ID.${NC}"
                rm -f prompt_content.md
                return 1
            fi
            sleep 5
            continue
        fi

        AGENT_IDS["$original_name"]=$AGENT_ID
        echo -e "${GREEN}Created agent: $agent_name with ID: $AGENT_ID${NC}"

        # Wait for agent to be in a valid state for preparation
        wait_for_agent_creatable $AGENT_ID
        if [ $? -ne 0 ]; then
            echo -e "${RED}Agent $agent_name is not in a valid state for preparation.${NC}"
            retry_count=$((retry_count + 1))
            continue
        fi

        # Prepare the agent with retry logic
        local prepare_attempts=5
        local prepare_delay=15
        local prepare_attempt=0
        local prepare_success=false

        while ((prepare_attempt < prepare_attempts)) && [[ "$prepare_success" == "false" ]]; do
            echo -e "${YELLOW}Attempting to prepare agent (attempt $((prepare_attempt+1))/$prepare_attempts)...${NC}"
            if aws bedrock-agent prepare-agent --agent-id $AGENT_ID --region $REGION 2>/dev/null; then
                prepare_success=true
                echo -e "${GREEN}Successfully prepared agent.${NC}"
            else
                echo -e "${YELLOW}Failed to prepare agent. Waiting $prepare_delay seconds before retry...${NC}"
                sleep $prepare_delay
                prepare_attempt=$((prepare_attempt+1))
            fi
        done

        if [[ "$prepare_success" == "false" ]]; then
            echo -e "${YELLOW}Failed to prepare agent after $prepare_attempts attempts.${NC}"
            retry_count=$((retry_count + 1))
            continue
        fi

        # Wait for agent to be ready
        wait_for_agent $AGENT_ID
        if [ $? -ne 0 ]; then
            echo -e "${RED}Agent $agent_name did not become ready.${NC}"
            retry_count=$((retry_count + 1))
            continue
        fi

        # Create agent alias with the created version
        ALIAS_RESPONSE=$(aws bedrock-agent create-agent-alias \
            --agent-id $AGENT_ID \
            --agent-alias-name "Production" \
            --region $REGION)

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to create alias for agent $agent_name.${NC}"
            retry_count=$((retry_count + 1))
            continue
        }

        # Extract alias ID
        ALIAS_ID=$(echo $ALIAS_RESPONSE | jq -r '.agentAlias.agentAliasId')
        if [ -z "$ALIAS_ID" ] || [ "$ALIAS_ID" == "null" ]; then
            echo -e "${RED}Failed to extract alias ID for $agent_name.${NC}"
            retry_count=$((retry_count + 1))
            continue
        fi

        # Store the alias ID
        AGENT_ALIAS_IDS["$original_name"]=$ALIAS_ID
        echo -e "${GREEN}Created alias for $agent_name with ID: $ALIAS_ID${NC}"
        echo -e "${GREEN}Agent $agent_name is ready${NC}"

        # Clean up
        rm -f prompt_content.md
        break
    done

    if [ $retry_count -ge $max_retries ]; then
        echo -e "${RED}Maximum retries reached for agent $original_name.${NC}"
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
        echo -e "${YELLOW}Cannot set up collaboration between $source_agent and $target_agent. Missing agent IDs.${NC}"
        return 1
    fi

    echo -e "${BLUE}Setting up collaboration: $source_agent -> $target_agent${NC}"

    aws bedrock-agent associate-agent-collaborator \
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
    echo -e "${BLUE}Creating all agents for the manuscript editing system...${NC}"

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

    echo -e "${GREEN}All agents created successfully.${NC}"
    declare -p AGENT_IDS > agent_ids.txt
    declare -p AGENT_ALIAS_IDS > agent_alias_ids.txt
    rm -f agent-definitions.json
}

# Set up collaborations
setup_agent_collaborations() {
    echo -e "${BLUE}Setting up collaborations between agents...${NC}"

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
            echo -e "${BLUE}Associating Knowledge Base with Executive Director agent...${NC}"
            aws bedrock-agent associate-agent-knowledge-base \
                --agent-id $executive_director_id \
                --agent-version "DRAFT" \
                --knowledge-base-id $KB_ID \
                --description "Knowledge base for manuscript research and context" \
                --knowledge-base-state "ENABLED" \
                --region $REGION
        fi
    fi

    # Prepare the updated Executive Director agent with the collaborations
    echo -e "${BLUE}Preparing updated Executive Director agent with collaborations...${NC}"
    executive_director_id=${AGENT_IDS["ExecutiveDirector"]}
    executive_director_alias_id=${AGENT_ALIAS_IDS["ExecutiveDirector"]}

    aws bedrock-agent prepare-agent --agent-id $executive_director_id --region $REGION
    wait_for_agent $executive_director_id

    # Create a new version and update the alias
    VERSION_RESPONSE=$(aws bedrock-agent create-agent-version --agent-id $executive_director_id --region $REGION)
    VERSION=$(echo $VERSION_RESPONSE | jq -r '.agentVersion')

    aws bedrock-agent update-agent-alias \
        --agent-id $executive_director_id \
        --agent-alias-id $executive_director_alias_id \
        --routing-configuration "[{\"agentVersion\":\"$VERSION\"}]" \
        --region $REGION

    # Update Lambda function environment variables with agent IDs
    aws lambda update-function-configuration \
        --function-name $LAMBDA_WORKFLOW_NAME \
        --environment "Variables={STATE_TABLE=$DYNAMODB_TABLE,EXECUTIVE_DIRECTOR_ID=$executive_director_id,EXECUTIVE_DIRECTOR_ALIAS_ID=$executive_director_alias_id}" \
        --region $REGION

    rm -f collaborations.json
}

# Main functions for resource creation/deletion
create_resources() {
    check_dependencies
    create_iam_role
    create_s3_bucket
    create_dynamodb_table
    upload_prompts
    create_knowledge_base
    create_lambda_functions
    create_agents
    setup_agent_collaborations

    echo -e "${GREEN}All resources created successfully.${NC}"
    echo -e "${BLUE}Your Manuscript Editing system is ready to use.${NC}"
}

delete_resources() {
    echo -e "${RED}WARNING: This will delete all resources created for the Manuscript Editing system.${NC}"
    read -p "Are you sure you want to proceed? (y/n): " confirm
    if [[ $confirm != "y" && $confirm != "Y" ]]; then
        echo -e "${BLUE}Deletion cancelled.${NC}"
        return
    fi

    # Load resource IDs if available
    if [ -f agent_ids.txt ]; then source agent_ids.txt; fi
    if [ -f agent_alias_ids.txt ]; then source agent_alias_ids.txt; fi
    if [ -f knowledge_base_info.txt ]; then source knowledge_base_info.txt; fi

    # Delete agents
    for agent_name in "${!AGENT_IDS[@]}"; do
        agent_id=${AGENT_IDS[$agent_name]}
        echo -e "Deleting agent: $agent_name ($agent_id)"
        if [ ! -z "${AGENT_ALIAS_IDS[$agent_name]}" ]; then
            alias_id=${AGENT_ALIAS_IDS[$agent_name]}
            aws bedrock-agent delete-agent-alias --agent-id $agent_id --agent-alias-id $alias_id --region $REGION 2>/dev/null || true
        fi
        aws bedrock-agent delete-agent --agent-id $agent_id --skip-resource-in-use-check --region $REGION 2>/dev/null || true
    done

    # Delete knowledge base
    if [ ! -z "$KB_ID" ]; then
        aws bedrock delete-knowledge-base --knowledge-base-id $KB_ID --region $REGION 2>/dev/null || true
    fi

    # Delete Lambda functions
    aws lambda delete-function --function-name $LAMBDA_RESEARCH_NAME --region $REGION 2>/dev/null || true
    aws lambda delete-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION 2>/dev/null || true

    # Delete DynamoDB table
    aws dynamodb delete-table --table-name $DYNAMODB_TABLE --region $REGION 2>/dev/null || true

    # Delete S3 bucket
    aws s3 rm s3://${FLOW_BUCKET_NAME} --recursive --region $REGION 2>/dev/null || true
    aws s3api delete-bucket --bucket ${FLOW_BUCKET_NAME} --region $REGION 2>/dev/null || true

    # Delete OpenSearch Serverless collection
    aws opensearchserverless delete-collection --name storybook --region $REGION 2>/dev/null || true
    aws opensearchserverless delete-access-policy --name storybook-policy --type data --region $REGION 2>/dev/null || true

    # Delete IAM role
    aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy 2>/dev/null || true
    aws iam delete-policy --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ManuscriptEditorPolicy 2>/dev/null || true
    aws iam delete-role --role-name $ROLE_NAME 2>/dev/null || true

    # Remove info files
    rm -f agent_ids.txt agent_alias_ids.txt knowledge_base_info.txt

    echo -e "${GREEN}All resources deleted successfully.${NC}"
}

# Show info and test functions
show_info() {
    echo -e "${BLUE}Retrieving information about deployed resources...${NC}"

    # Display agent information
    if [ -f agent_ids.txt ]; then
        source agent_ids.txt
        echo -e "${GREEN}Agent Information:${NC}"
        for agent_name in "${!AGENT_IDS[@]}"; do
            echo -e "$agent_name: ${AGENT_IDS[$agent_name]}"
        done
    fi

    # Display other resource information
    if [ -f knowledge_base_info.txt ]; then
        source knowledge_base_info.txt
        echo -e "${GREEN}Knowledge Base ID: $KB_ID${NC}"
    fi

    # Check resource existence
    if aws dynamodb describe-table --table-name $DYNAMODB_TABLE --region $REGION &>/dev/null; then
        echo -e "${GREEN}DynamoDB Table: $DYNAMODB_TABLE exists${NC}"
    fi

    if aws lambda get-function --function-name $LAMBDA_RESEARCH_NAME --region $REGION &>/dev/null; then
        echo -e "${GREEN}Lambda Function: $LAMBDA_RESEARCH_NAME exists${NC}"
    fi

    if aws lambda get-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION &>/dev/null; then
        echo -e "${GREEN}Lambda Function: $LAMBDA_WORKFLOW_NAME exists${NC}"
    fi

    echo -e "${GREEN}S3 Bucket: $FLOW_BUCKET_NAME${NC}"
}

test_system() {
    echo -e "${BLUE}Testing the Manuscript Editing system...${NC}"

    # Check if workflow Lambda exists
    if ! aws lambda get-function --function-name $LAMBDA_WORKFLOW_NAME --region $REGION &>/dev/null; then
        echo -e "${RED}Workflow Lambda function not found. Please deploy the system first.${NC}"
        return 1
    fi

    # Download test manuscript from GitHub
    download_from_github "config/test_manuscript.json" "test_manuscript.json"

    # Invoke Lambda with the test manuscript
    echo -e "${BLUE}Invoking workflow Lambda with test manuscript...${NC}"
    aws lambda invoke --function-name $LAMBDA_WORKFLOW_NAME --payload file://test_manuscript.json --region $REGION output.json

    # Display the output
    echo -e "${GREEN}Workflow Lambda invoked successfully. Output:${NC}"
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

    echo -e "\n${BLUE}To check the status of the editing process, use:${NC}"
    echo -e "${YELLOW}$(cat check_status.json)${NC}"

    # Clean up
    rm -f test_manuscript.json
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
            5) echo -e "${GREEN}Exiting. Goodbye!${NC}"; exit 0 ;;
            *) echo -e "${RED}Invalid option. Please try again.${NC}" ;;
        esac
    done
}

# Start script
echo -e "${BLUE}AWS Bedrock Hierarchical Multi-Agent Manuscript Editing System${NC}"
echo -e "${BLUE}This script will deploy a complete system for editing manuscripts using AI agents${NC}"
main_menu
