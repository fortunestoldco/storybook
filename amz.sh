#!/bin/bash
# NovelFlow: Amazon Bedrock Flow Implementation of Novel Editing System

# Define colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration Files
CONFIG_FILE="./novelflow_config.json"
FLOW_TEMPLATES_DIR="./flow_templates"
FLOWS_LIST_FILE="./flows_list.json"
TEMP_DIR="./temp"
MANUSCRIPT_DIR="./manuscripts"
CHUNKS_DIR="./chunks"
RESEARCH_DIR="./research"

# AWS Defaults
DEFAULT_REGION="us-west-2"
DEFAULT_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"
DEFAULT_ROLE_NAME="NovelFlowRole"
DEFAULT_TABLE_PREFIX="novelflow"

# Chunk size for large manuscript processing (token approximation)
DEFAULT_CHUNK_SIZE=8000
DEFAULT_OVERLAP=500

# Create required directories
mkdir -p "$TEMP_DIR"
mkdir -p "$FLOW_TEMPLATES_DIR"
mkdir -p "$MANUSCRIPT_DIR"
mkdir -p "$CHUNKS_DIR"
mkdir -p "$RESEARCH_DIR"

# Check required commands
check_requirements() {
    local requirements=("aws" "jq" "python3" "curl")
    for cmd in "${requirements[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            echo -e "${RED}Error: $cmd is required but not installed. Please install it and try again.${NC}"
            exit 1
        fi
    done
    
    # Check for required Python packages
    python3 -c "import nltk, tiktoken" &>/dev/null || {
        echo -e "${YELLOW}Installing required Python packages...${NC}"
        pip3 install nltk tiktoken requests beautifulsoup4
        python3 -c "import nltk; nltk.download('punkt')" &>/dev/null
    }
}

# Initialize configuration file
initialize_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${YELLOW}Initializing configuration file...${NC}"
        mkdir -p "$(dirname "$CONFIG_FILE")"

        # Create base config with all agents
        cat > "$CONFIG_FILE" <<EOF
{
    "aws_region": "$DEFAULT_REGION",
    "aws_profile": "default",
    "default_model": "$DEFAULT_MODEL",
    "role_name": "$DEFAULT_ROLE_NAME",
    "table_prefix": "$DEFAULT_TABLE_PREFIX",
    "chunk_size": $DEFAULT_CHUNK_SIZE,
    "chunk_overlap": $DEFAULT_OVERLAP,
    "quality_thresholds": {
        "style_consistency": 7,
        "narrative_coherence": 8,
        "character_consistency": 7,
        "pacing": 7,
        "dialogue_quality": 8,
        "literary_quality": 8,
        "engagement_potential": 8,
        "market_appeal": 7
    },
    "models": {
        "executive_editor": "$DEFAULT_MODEL",
        "content_assessor": "$DEFAULT_MODEL",
        "developmental_editor": "$DEFAULT_MODEL",
        "line_editor": "$DEFAULT_MODEL",
        "style_specialist": "$DEFAULT_MODEL",
        "pacing_analyst": "$DEFAULT_MODEL",
        "dialogue_specialist": "$DEFAULT_MODEL",
        "character_consistency_checker": "$DEFAULT_MODEL",
        "narrative_coherence_expert": "$DEFAULT_MODEL",
        "plot_structure_analyst": "$DEFAULT_MODEL",
        "worldbuilding_evaluator": "$DEFAULT_MODEL",
        "audience_engagement_analyst": "$DEFAULT_MODEL",
        "conflict_progression_analyst": "$DEFAULT_MODEL",
        "emotional_arc_evaluator": "$DEFAULT_MODEL",
        "voice_consistency_examiner": "$DEFAULT_MODEL",
        "research_verifier": "$DEFAULT_MODEL",
        "market_trends_analyst": "$DEFAULT_MODEL",
        "language_refiner": "$DEFAULT_MODEL",
        "genre_specialist": "$DEFAULT_MODEL",
        "continuity_checker": "$DEFAULT_MODEL",
        "proofreader": "$DEFAULT_MODEL",
        "text_improver": "$DEFAULT_MODEL",
        "research_specialist": "$DEFAULT_MODEL",
        "integration_editor": "$DEFAULT_MODEL",
        "final_polish_editor": "$DEFAULT_MODEL"
    },
    "prompt_templates": {},
    "knowledge_bases": {},
    "processing_settings": {
        "web_search_enabled": true,
        "context_synthesis": true,
        "nlp_analysis": true,
        "progressive_refinement": true,
        "auto_research": true
    }
}
EOF
        echo -e "${GREEN}Configuration file created at $CONFIG_FILE${NC}"
    fi

    # Make sure templates directory exists
    mkdir -p "$FLOW_TEMPLATES_DIR"

    # Create flow list file if it doesn't exist
    if [ ! -f "$FLOWS_LIST_FILE" ]; then
        echo -e "${YELLOW}Creating flows list file...${NC}"
        echo '{"flows": []}' > "$FLOWS_LIST_FILE"
        echo -e "${GREEN}Flows list file created at $FLOWS_LIST_FILE${NC}"
    fi
}

# Initialize IAM role for Bedrock Flows
initialize_iam_role() {
    local role_name=$(jq -r '.role_name' "$CONFIG_FILE")
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")

    # Check if role already exists
    if aws iam get-role --role-name "$role_name" --region "$region" --profile "$profile" &> /dev/null; then
        echo -e "${BLUE}IAM role $role_name already exists.${NC}"
        return 0
    fi

    echo -e "${YELLOW}Creating IAM role $role_name for NovelFlow...${NC}"

    # Create trust policy
    cat > "$TEMP_DIR/trust-policy.json" <<EOF
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

    # Create policy document for Bedrock permissions
    cat > "$TEMP_DIR/bedrock-policy.json" <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockFlowPermissions",
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
                "bedrock:ListTagsForResource",
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
                "bedrock:GetAgent",
                "bedrock:GetKnowledgeBase",
                "bedrock:GetGuardrail"
            ],
            "Resource": "*"
        }
    ]
}
EOF

    # Create policy document for DynamoDB permissions
    cat > "$TEMP_DIR/dynamodb-policy.json" <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DynamoDBPermissions",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:Scan",
                "dynamodb:BatchGetItem",
                "dynamodb:BatchWriteItem"
            ],
            "Resource": [
                "arn:aws:dynamodb:*:*:table/${DEFAULT_TABLE_PREFIX}*"
            ]
        }
    ]
}
EOF

    # Create the role
    aws iam create-role \
        --role-name "$role_name" \
        --assume-role-policy-document file://"$TEMP_DIR/trust-policy.json" \
        --region "$region" \
        --profile "$profile"

    # Attach Bedrock policy to role
    aws iam put-role-policy \
        --role-name "$role_name" \
        --policy-name "${role_name}BedrockPolicy" \
        --policy-document file://"$TEMP_DIR/bedrock-policy.json" \
        --region "$region" \
        --profile "$profile"

    # Attach DynamoDB policy to role
    aws iam put-role-policy \
        --role-name "$role_name" \
        --policy-name "${role_name}DynamoDBPolicy" \
        --policy-document file://"$TEMP_DIR/dynamodb-policy.json" \
        --region "$region" \
        --profile "$profile"

    echo -e "${GREEN}IAM role $role_name created successfully.${NC}"
    # Sleep to allow IAM role to propagate
    echo -e "${YELLOW}Waiting for IAM role to propagate...${NC}"
    sleep 10
}

# Initialize DynamoDB tables
initialize_dynamodb() {
    local project_name="$1"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    local table_prefix=$(jq -r '.table_prefix' "$CONFIG_FILE")

    echo -e "${YELLOW}Creating DynamoDB tables for project $project_name...${NC}"

    # Create project state table
    local state_table="${table_prefix}_${project_name}_state"
    aws dynamodb create-table \
        --table-name "$state_table" \
        --attribute-definitions AttributeName=manuscript_id,AttributeType=S AttributeName=chunk_id,AttributeType=S \
        --key-schema AttributeName=manuscript_id,KeyType=HASH AttributeName=chunk_id,KeyType=RANGE \
        --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
        --region "$region" \
        --profile "$profile" \
        --output json

    # Create edits table
    local edits_table="${table_prefix}_${project_name}_edits"
    aws dynamodb create-table \
        --table-name "$edits_table" \
        --attribute-definitions AttributeName=edit_id,AttributeType=S \
        --key-schema AttributeName=edit_id,KeyType=HASH \
        --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
        --region "$region" \
        --profile "$profile" \
        --output json

    # Create quality metrics table
    local metrics_table="${table_prefix}_${project_name}_metrics"
    aws dynamodb create-table \
        --table-name "$metrics_table" \
        --attribute-definitions AttributeName=metric_id,AttributeType=S \
        --key-schema AttributeName=metric_id,KeyType=HASH \
        --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
        --region "$region" \
        --profile "$profile" \
        --output json

    # Create research findings table
    local research_table="${table_prefix}_${project_name}_research"
    aws dynamodb create-table \
        --table-name "$research_table" \
        --attribute-definitions AttributeName=research_id,AttributeType=S \
        --key-schema AttributeName=research_id,KeyType=HASH \
        --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
        --region "$region" \
        --profile "$profile" \
        --output json

    echo -e "${GREEN}DynamoDB tables created successfully.${NC}"

    # Initialize project state
    echo -e "${YELLOW}Initializing project state...${NC}"
    aws dynamodb put-item \
        --table-name "$state_table" \
        --item '{
            "manuscript_id": {"S": "'$project_name'"},
            "chunk_id": {"S": "metadata"},
            "status": {"S": "created"},
            "created_at": {"S": "'$(date -u +'%Y-%m-%dT%H:%M:%SZ')'"},
            "updated_at": {"S": "'$(date -u +'%Y-%m-%dT%H:%M:%SZ')'"},
            "title": {"S": ""},
            "author": {"S": ""},
            "total_chunks": {"N": "0"},
            "chunks_processed": {"N": "0"},
            "current_phase": {"S": "assessment"},
            "quality_scores": {"M": {}}
        }' \
        --region "$region" \
        --profile "$profile" \
        --output json

    echo -e "${GREEN}Project state initialized successfully.${NC}"
}

# Generate Python script for manuscript chunking
generate_chunking_script() {
    cat > "$TEMP_DIR/chunk_manuscript.py" <<EOF
#!/usr/bin/env python3
import os
import sys
import json
import re
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
import argparse

def count_tokens(text, encoding_name="cl100k_base"):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def chunk_text(text, max_tokens=8000, overlap_tokens=500):
    """Split text into chunks of approximately max_tokens with overlap."""
    # First split into sentences
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_token_count = count_tokens(sentence)
        
        # If adding this sentence would exceed max tokens, finish the chunk
        if current_token_count + sentence_token_count > max_tokens and current_chunk:
            # Save the current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start a new chunk with overlap
            overlap_text = []
            overlap_token_count = 0
            
            # Add sentences from the end of the previous chunk until we reach desired overlap
            for i in range(len(current_chunk) - 1, -1, -1):
                sentence_for_overlap = current_chunk[i]
                sentence_overlap_tokens = count_tokens(sentence_for_overlap)
                
                if overlap_token_count + sentence_overlap_tokens <= overlap_tokens:
                    overlap_text.insert(0, sentence_for_overlap)
                    overlap_token_count += sentence_overlap_tokens
                else:
                    break
            
            # Reset with overlap sentences
            current_chunk = overlap_text
            current_token_count = overlap_token_count
        
        # Add the current sentence to the chunk
        current_chunk.append(sentence)
        current_token_count += sentence_token_count
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def detect_chapters(text):
    """Identify chapter breaks in the text."""
    # Common chapter patterns
    chapter_patterns = [
        r'(?i)^chapter\s+\d+', 
        r'(?i)^chapter\s+[IVXLCDM]+', 
        r'(?i)^\d+\.\s+',
        r'(?m)^\s*CHAPTER\s+(?:\d+|[IVXLCDM]+)'
    ]
    
    # Try to detect chapter markers
    chapter_matches = []
    for pattern in chapter_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            chapter_matches.append((match.start(), match.group()))
    
    # Sort by position in text
    chapter_matches.sort()
    return chapter_matches

def preserve_chapter_integrity(text, chunks):
    """Adjust chunk boundaries to avoid breaking up chapters."""
    chapter_markers = detect_chapters(text)
    if not chapter_markers:
        return chunks  # No chapters detected
    
    # Convert chapter positions to their containing chunk
    chapter_positions = []
    current_pos = 0
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        for pos, marker in chapter_markers:
            if current_pos <= pos < current_pos + chunk_len:
                chapter_positions.append((i, pos - current_pos, marker))
        current_pos += chunk_len - overlap_tokens_length
    
    # Adjust chunks to respect chapter boundaries when possible
    modified_chunks = chunks.copy()
    
    # Process each chapter marker
    for i in range(len(chapter_positions) - 1):
        chunk_idx, pos_in_chunk, marker = chapter_positions[i]
        
        # If chapter marker is close to the end of a chunk, we might adjust it
        if pos_in_chunk > len(chunks[chunk_idx]) * 0.7:
            # Try to merge with next chunk if it's not too large
            if chunk_idx < len(chunks) - 1:
                if len(chunks[chunk_idx]) + len(chunks[chunk_idx + 1]) < max_tokens * 1.3:
                    modified_chunks[chunk_idx] = chunks[chunk_idx] + chunks[chunk_idx + 1]
                    modified_chunks.pop(chunk_idx + 1)
                    # Update positions of later chapters
                    chapter_positions = [(idx if idx <= chunk_idx else idx - 1, pos, m) 
                                       for idx, pos, m in chapter_positions]
    
    return modified_chunks

def save_chunks(chunks, output_dir, project_name):
    """Save chunks to individual files."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        file_path = os.path.join(output_dir, f"{project_name}_chunk_{i+1:04d}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(chunk)
        chunk_files.append(file_path)
        
    # Create metadata file
    metadata = {
        "project_name": project_name,
        "total_chunks": len(chunks),
        "chunk_files": chunk_files,
        "token_counts": [count_tokens(chunk) for chunk in chunks]
    }
    
    with open(os.path.join(output_dir, f"{project_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split manuscript into chunks for processing')
    parser.add_argument('manuscript_path', help='Path to manuscript file')
    parser.add_argument('--output_dir', default='./chunks', help='Directory to save chunks')
    parser.add_argument('--project_name', required=True, help='Project name')
    parser.add_argument('--max_tokens', type=int, default=8000, help='Maximum tokens per chunk')
    parser.add_argument('--overlap_tokens', type=int, default=500, help='Token overlap between chunks')
    
    args = parser.parse_args()
    
    # Load config to get parameters
    try:
        with open("./novelflow_config.json", 'r') as f:
            config = json.load(f)
            max_tokens = config.get('chunk_size', args.max_tokens)
            overlap_tokens = config.get('chunk_overlap', args.overlap_tokens)
    except:
        max_tokens = args.max_tokens
        overlap_tokens = args.overlap_tokens
    
    # Calculate approximate length of overlap in characters for chapter boundary calculations
    overlap_tokens_length = overlap_tokens * 4  # rough approximation
    
    # Read manuscript
    with open(args.manuscript_path, 'r', encoding='utf-8') as f:
        manuscript_text = f.read()
    
    # Chunk the text
    text_chunks = chunk_text(manuscript_text, max_tokens, overlap_tokens)
    
    # Preserve chapter integrity when possible
    refined_chunks = preserve_chapter_integrity(manuscript_text, text_chunks)
    
    # Save chunks
    metadata = save_chunks(refined_chunks, args.output_dir, args.project_name)
    
    print(f"Manuscript split into {len(refined_chunks)} chunks. Metadata saved.")
    print(json.dumps(metadata))
EOF
    chmod +x "$TEMP_DIR/chunk_manuscript.py"
}

# Generate Python script for web research
generate_research_script() {
    cat > "$TEMP_DIR/web_research.py" <<EOF
#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import requests
from bs4 import BeautifulSoup
import re
import boto3
from datetime import datetime

def sanitize_filename(filename):
    """Remove illegal characters from filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def search_web(query, max_results=5):
    """Perform web search using a search API (DuckDuckGo)."""
    # This is a simple implementation - in production, use a proper search API
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Using DuckDuckGo HTML search as a simple approach
        encoded_query = query.replace(' ', '+')
        response = requests.get(f'https://html.duckduckgo.com/html/?q={encoded_query}', headers=headers)
        
        if response.status_code != 200:
            return {"error": f"Search failed with status {response.status_code}"}
            
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Extract search results
        for result in soup.select('.result__body')[:max_results]:
            title_elem = result.select_one('.result__title')
            link_elem = result.select_one('.result__url')
            snippet_elem = result.select_one('.result__snippet')
            
            if title_elem and link_elem:
                title = title_elem.get_text().strip()
                url = link_elem.get('href') if link_elem.has_attr('href') else link_elem.get_text().strip()
                snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                
                # Clean up URL if it's from DuckDuckGo's redirect
                if '/d.js' in url:
                    url_match = re.search(r'uddg=([^&]+)', url)
                    if url_match:
                        url = requests.utils.unquote(url_match.group(1))
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
        
        return {"results": results}
    
    except Exception as e:
        return {"error": str(e)}

def extract_content(url):
    """Extract main content from a web page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {"error": f"Failed to retrieve content: Status {response.status_code}"}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Extract title
        title = soup.title.string if soup.title else "Unknown Title"
        
        # Try to find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            content = main_content.get_text(separator='\n')
        else:
            # Fallback to body content
            content = soup.get_text(separator='\n')
        
        # Clean up the content
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return {
            "title": title,
            "content": content,
            "url": url
        }
    
    except Exception as e:
        return {"error": f"Error extracting content: {str(e)}"}

def summarize_content(content, max_length=2000):
    """Truncate content to a maximum length while preserving whole sentences."""
    if len(content) <= max_length:
        return content
    
    # Find a sentence boundary near max_length
    truncated = content[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > 0:
        return content[:last_period + 1]
    else:
        return truncated

def save_to_dynamodb(research_data, table_name, project_name, region):
    """Save research data to DynamoDB."""
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    # Generate a unique ID
    research_id = f"{project_name}_research_{int(time.time())}"
    
    # Prepare item for DynamoDB
    item = {
        'research_id': research_id,
        'project_name': project_name,
        'timestamp': datetime.utcnow().isoformat(),
        'query': research_data.get('query', ''),
        'sources': research_data.get('sources', []),
        'summary': research_data.get('summary', '')
    }
    
    try:
        table.put_item(Item=item)
        return {"status": "success", "research_id": research_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def conduct_research(query, output_dir, project_name, max_results=5, save_to_db=True, region='us-west-2', table_prefix='novelflow'):
    """Perform web research on the given query."""
    os.makedirs(output_dir, exist_ok=True)
    research_filename = sanitize_filename(f"{project_name}_{query[:30]}.json")
    research_path = os.path.join(output_dir, research_filename)
    
    # Perform search
    search_results = search_web(query, max_results)
    
    if "error" in search_results:
        print(f"Search error: {search_results['error']}", file=sys.stderr)
        return {"error": search_results['error']}
    
    # Extract content from each result
    sources = []
    for result in search_results.get("results", []):
        print(f"Processing: {result['title']}")
        content_data = extract_content(result['url'])
        
        if "error" not in content_data:
            # Summarize content if needed
            content_data["content"] = summarize_content(content_data["content"])
            sources.append(content_data)
        else:
            print(f"Error extracting content: {content_data['error']}", file=sys.stderr)
    
    # Compile research data
    research_data = {
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "sources": sources,
        "summary": f"Research on '{query}' found {len(sources)} sources."
    }
    
    # Save to file
    with open(research_path, 'w', encoding='utf-8') as f:
        json.dump(research_data, f, indent=2)
    
    # Save to DynamoDB if requested
    if save_to_db:
        table_name = f"{table_prefix}_{project_name}_research"
        db_result = save_to_dynamodb(research_data, table_name, project_name, region)
        if "error" in db_result:
            print(f"DynamoDB error: {db_result['error']}", file=sys.stderr)
    
    print(f"Research completed and saved to {research_path}")
    return research_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conduct web research for a novel manuscript')
    parser.add_argument('query', help='Research query')
    parser.add_argument('--output_dir', default='./research', help='Directory to save research')
    parser.add_argument('--project_name', required=True, help='Project name')
    parser.add_argument('--max_results', type=int, default=5, help='Maximum search results to process')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--table_prefix', default='novelflow', help='DynamoDB table prefix')
    parser.add_argument('--no-db', action='store_true', help='Skip saving to DynamoDB')
    
    args = parser.parse_args()
    
    # Load config to get parameters
    try:
        with open("./novelflow_config.json", 'r') as f:
            config = json.load(f)
            region = config.get('aws_region', args.region)
            table_prefix = config.get('table_prefix', args.table_prefix)
    except:
        region = args.region
        table_prefix = args.table_prefix
    
    result = conduct_research(
        args.query, 
        args.output_dir, 
        args.project_name, 
        args.max_results,
        not args.no_db,
        region,
        table_prefix
    )
    
    print(json.dumps(result, indent=2))
EOF
    chmod +x "$TEMP_DIR/web_research.py"
}

# Generate Python script for manuscript assessment
generate_assessment_script() {
    cat > "$TEMP_DIR/assess_manuscript.py" <<EOF
#!/usr/bin/env python3
import os
import sys
import json
import argparse
import boto3
from datetime import datetime
import random
import time

def analyze_chapters(chunks_dir, project_name):
    """Analyze chapter structure from chunks."""
    # Load metadata
    metadata_path = os.path.join(chunks_dir, f"{project_name}_metadata.json")
    
    if not os.path.exists(metadata_path):
        return {"error": f"Metadata file not found: {metadata_path}"}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Detect chapter pattern by sampling chunks
    chunk_files = metadata.get("chunk_files", [])
    if not chunk_files:
        return {"error": "No chunk files found in metadata"}
    
    # Sample a few chunks to detect chapters
    sample_size = min(5, len(chunk_files))
    sample_indices = sorted(random.sample(range(len(chunk_files)), sample_size))
    
    chapter_markers = []
    for idx in sample_indices:
        with open(chunk_files[idx], 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for common chapter patterns
            import re
            patterns = [
                r'(?i)^chapter\s+\d+', 
                r'(?i)^chapter\s+[IVXLCDM]+', 
                r'(?i)^\d+\.\s+',
                r'(?m)^\s*CHAPTER\s+(?:\d+|[IVXLCDM]+)'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    chapter_markers.append({
                        "pattern": pattern,
                        "example": match.group(0),
                        "chunk": idx
                    })
    
    # Count total chapters
    if chapter_markers:
        # Use the most common pattern to estimate chapters
        from collections import Counter
        pattern_counts = Counter([m["pattern"] for m in chapter_markers])
        dominant_pattern = pattern_counts.most_common(1)[0][0]
        
        # Count chapters in all chunks using dominant pattern
        total_chapters = 0
        for file_path in chunk_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = re.finditer(dominant_pattern, content, re.MULTILINE)
                total_chapters += sum(1 for _ in matches)
        
        chapter_info = {
            "detected": True,
            "pattern": dominant_pattern,
            "estimated_count": total_chapters,
            "example": chapter_markers[0]["example"] if chapter_markers else ""
        }
    else:
        chapter_info = {
            "detected": False,
            "estimated_count": 0
        }
    
    return {
        "chapter_analysis": chapter_info,
        "total_chunks": len(chunk_files),
        "approx_token_count": sum(metadata.get("token_counts", []))
    }

def sample_manuscript_content(chunks_dir, project_name, sample_size=3):
    """Extract sample content from the manuscript for analysis."""
    metadata_path = os.path.join(chunks_dir, f"{project_name}_metadata.json")
    
    if not os.path.exists(metadata_path):
        return {"error": f"Metadata file not found: {metadata_path}"}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    chunk_files = metadata.get("chunk_files", [])
    if not chunk_files:
        return {"error": "No chunk files found in metadata"}
    
    # Get evenly spaced samples (beginning, middle, end)
    indices = []
    if len(chunk_files) <= sample_size:
        indices = list(range(len(chunk_files)))
    else:
        step = len(chunk_files) // sample_size
        for i in range(sample_size):
            indices.append(min(i * step, len(chunk_files) - 1))
    
    samples = []
    for idx in indices:
        with open(chunk_files[idx], 'r', encoding='utf-8') as f:
            content = f.read()
            # Get a representative sample (first 2000 chars)
            sample = content[:2000] + "..." if len(content) > 2000 else content
            samples.append({
                "chunk_index": idx,
                "content": sample
            })
    
    return {
        "samples": samples,
        "total_chunks": len(chunk_files)
    }

def invoke_bedrock_model(prompt, region, model_id, max_tokens=1000):
    """Invoke Bedrock model to analyze manuscript content."""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
    
    # Determine request format based on model provider
    if "anthropic" in model_id.lower():
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    elif "claude" in model_id.lower():
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    elif "titan" in model_id.lower():
        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": 0.2,
                "topP": 0.9
            }
        }
    elif "llama" in model_id.lower():
        request_body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9
        }
    else:
        # Default format
        request_body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        
        # Extract content based on model
        if "anthropic" in model_id.lower() or "claude" in model_id.lower():
            return response_body.get('content', [{}])[0].get('text', '')
        elif "titan" in model_id.lower():
            return response_body.get('results', [{}])[0].get('outputText', '')
        elif "llama" in model_id.lower():
            return response_body.get('generation', '')
        else:
            return str(response_body)  # Default fallback
            
    except Exception as e:
        return f"Error invoking model: {str(e)}"

def analyze_manuscript(chunks_dir, project_name, region, model_id):
    """Perform initial manuscript assessment."""
    # Get structural information
    structure_info = analyze_chapters(chunks_dir, project_name)
    if "error" in structure_info:
        return {"error": structure_info["error"]}
    
    # Get content samples
    samples_info = sample_manuscript_content(chunks_dir, project_name)
    if "error" in samples_info:
        return {"error": samples_info["error"]}
    
    # Prepare prompt for content assessment
    assessment_prompt = f"""
    You are a professional literary editor tasked with improving a manuscript to best-seller quality.
    
    Below are sample sections from a manuscript. Analyze them and provide an assessment focusing on:
    1. Writing style and voice consistency
    2. Character development potential
    3. Narrative structure and flow 
    4. Dialogue quality
    5. Potential genre classification
    6. Areas needing improvement to reach best-seller quality
    
    Please provide specific examples from the text for each point and score each area from 1-10.
    
    MANUSCRIPT SAMPLES:
    
    {chr(10).join([f"SAMPLE {i+1}:\n{s['content']}" for i, s in enumerate(samples_info['samples'])])}
    
    Provide your assessment in JSON format with these keys:
    - style_assessment (include score and analysis)
    - character_assessment (include score and analysis)
    - narrative_assessment (include score and analysis)
    - dialogue_assessment (include score and analysis) 
    - genre_classification
    - improvement_recommendations
    - overall_score
    """
    
    # Get assessment from Bedrock model
    print("Analyzing manuscript content...")
    assessment_result = invoke_bedrock_model(assessment_prompt, region, model_id, 2000)
    
    # Try to parse JSON from response
    try:
        # Extract JSON if it's within a code block
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', assessment_result, re.DOTALL)
        if json_match:
            assessment_data = json.loads(json_match.group(1))
        else:
            # Look for JSON object in the response
            json_match = re.search(r'({.*})', assessment_result, re.DOTALL)
            if json_match:
                assessment_data = json.loads(json_match.group(1))
            else:
                # Fallback: just include the raw response
                assessment_data = {"raw_assessment": assessment_result}
    except Exception as e:
        assessment_data = {
            "raw_assessment": assessment_result,
            "parsing_error": str(e)
        }
    
    # Combine all assessment data
    full_assessment = {
        "project_name": project_name,
        "timestamp": datetime.utcnow().isoformat(),
        "structure_info": structure_info,
        "content_assessment": assessment_data,
        "sample_count": len(samples_info['samples']),
        "total_chunks": samples_info['total_chunks']
    }
    
    # Save assessment to file
    output_path = os.path.join(chunks_dir, f"{project_name}_assessment.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_assessment, f, indent=2)
    
    print(f"Assessment completed and saved to {output_path}")
    return full_assessment

def save_assessment_to_dynamodb(assessment_data, table_name, region):
    """Save assessment data to DynamoDB."""
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    # Extract assessment metrics
    metrics = {}
    content_assessment = assessment_data.get('content_assessment', {})
    
    # Build metrics dictionary from assessment
    for key in ['style_assessment', 'character_assessment', 'narrative_assessment', 'dialogue_assessment']:
        if isinstance(content_assessment.get(key), dict):
            metrics[key.replace('_assessment', '_score')] = content_assessment[key].get('score', 0)
        elif isinstance(content_assessment.get(key), str):
            # Try to extract score from string
            import re
            score_match = re.search(r'score[:\s]+(\d+)', content_assessment[key], re.IGNORECASE)
            if score_match:
                metrics[key.replace('_assessment', '_score')] = int(score_match.group(1))
    
    # Add overall score if available
    if 'overall_score' in content_assessment:
        metrics['overall_score'] = content_assessment['overall_score']
    
    # Prepare item for DynamoDB
    item = {
        'metric_id': f"{assessment_data['project_name']}_initial_assessment",
        'project_name': assessment_data['project_name'],
        'timestamp': assessment_data['timestamp'],
        'metrics': metrics,
        'structure_info': assessment_data.get('structure_info', {}),
        'genre_classification': content_assessment.get('genre_classification', 'Unknown'),
        'improvement_recommendations': content_assessment.get('improvement_recommendations', [])
    }
    
    try:
        table.put_item(Item=item)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assess manuscript for editing')
    parser.add_argument('--chunks_dir', default='./chunks', help='Directory containing manuscript chunks')
    parser.add_argument('--project_name', required=True, help='Project name')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--model_id', help='Bedrock model ID')
    parser.add_argument('--table_prefix', default='novelflow', help='DynamoDB table prefix')
    parser.add_argument('--no-db', action='store_true', help='Skip saving to DynamoDB')
    
    args = parser.parse_args()
    
    # Load config to get parameters
    try:
        with open("./novelflow_config.json", 'r') as f:
            config = json.load(f)
            region = config.get('aws_region', args.region)
            model_id = args.model_id or config.get('default_model')
            table_prefix = config.get('table_prefix', args.table_prefix)
    except:
        region = args.region
        model_id = args.model_id or "anthropic.claude-3-sonnet-20240229-v1:0"
        table_prefix = args.table_prefix
    
    # Run assessment
    assessment = analyze_manuscript(args.chunks_dir, args.project_name, region, model_id)
    
    # Save to DynamoDB if requested
    if not args.no_db and "error" not in assessment:
        table_name = f"{table_prefix}_{args.project_name}_metrics"
        db_result = save_assessment_to_dynamodb(assessment, table_name, region)
        if "error" in db_result:
            print(f"DynamoDB error: {db_result['error']}", file=sys.stderr)
    
    print(json.dumps(assessment, indent=2))
EOF
    chmod +x "$TEMP_DIR/assess_manuscript.py"
}

# Generate flow JSON for manuscript assessment
generate_assessment_flow() {
    local flow_name="$1"
    local role_arn="$2"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")

    # Get model IDs
    local executive_model=$(jq -r '.models.executive_editor // .default_model' "$CONFIG_FILE")
    local content_assessor_model=$(jq -r '.models.content_assessor // .default_model' "$CONFIG_FILE")
    local dev_editor_model=$(jq -r '.models.developmental_editor // .default_model' "$CONFIG_FILE")
    local style_model=$(jq -r '.models.style_specialist // .default_model' "$CONFIG_FILE")
    local plot_model=$(jq -r '.models.plot_structure_analyst // .default_model' "$CONFIG_FILE")
    
    # Create flow JSON
    cat > "$FLOW_TEMPLATES_DIR/${flow_name}.json" <<EOF
{
    "name": "$flow_name",
    "description": "Manuscript assessment flow for novel editing system",
    "executionRoleArn": "$role_arn",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [
                    { "name": "manuscript_id", "type": "String" },
                    { "name": "title", "type": "String" },
                    { "name": "sample_text", "type": "String" }
                ]
            },
            {
                "type": "Prompt",
                "name": "ExecutiveEditorNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Executive Editor overseeing the improvement of a novel manuscript to best-seller quality. Manuscript ID: {{manuscript_id}}, Title: {{title}}. \n\nReview the following sample to determine which specialists should analyze this manuscript:\n\n{{sample_text}}\n\nProvide your assessment and recommendations in JSON format with these fields:\n- initial_impression\n- primary_areas_for_improvement (list at least 3)\n- recommended_specialists (list which of our specialists should focus on this manuscript)\n- estimated_improvement_potential (score 1-10)"
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "sample_text", "type": "String", "expression": "$.sample_text" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "ContentAssessmentNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$content_assessor_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Content Assessment Specialist. Analyze this sample from manuscript '{{title}}' (ID: {{manuscript_id}}):\n\n{{sample_text}}\n\nProvide a detailed assessment focusing on:\n1. Writing style and voice consistency (score 1-10)\n2. Character development (score 1-10)\n3. Narrative structure and flow (score 1-10)\n4. Dialogue quality (score 1-10)\n5. Genre alignment and market potential\n6. Specific improvement recommendations to reach best-seller quality\n\nProvide your assessment in JSON format with scores for each area."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "sample_text", "type": "String", "expression": "$.sample_text" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "DevelopmentalEditorNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$dev_editor_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Developmental Editor. Analyze this sample from manuscript '{{title}}' (ID: {{manuscript_id}}):\n\n{{sample_text}}\n\nIdentify major developmental issues and provide strategic recommendations for improving:\n1. Plot structure and progression\n2. Character arcs and development\n3. Thematic coherence and depth\n4. Pacing and engagement\n5. Target audience alignment\n\nFor each issue, provide specific examples from the text and actionable guidance for revision. Include a development plan that would elevate this manuscript to best-seller quality."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "sample_text", "type": "String", "expression": "$.sample_text" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "StyleSpecialistNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$style_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Style Specialist. Analyze this sample from manuscript '{{title}}' (ID: {{manuscript_id}}):\n\n{{sample_text}}\n\nEvaluate the prose style and provide detailed feedback on:\n1. Voice consistency and distinctiveness\n2. Syntax and sentence structure variety\n3. Word choice and vocabulary appropriateness\n4. Showing vs. telling balance\n5. Sensory details and imagery\n6. Overall rhythm and flow\n\nProvide specific examples of both strengths and weaknesses, with concrete suggestions for stylistic improvement to reach best-seller quality."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "sample_text", "type": "String", "expression": "$.sample_text" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "PlotStructureNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$plot_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Plot Structure Analyst. Analyze this sample from manuscript '{{title}}' (ID: {{manuscript_id}}):\n\n{{sample_text}}\n\nEvaluate the narrative structure and provide insights on:\n1. Plot progression and key story beats\n2. Tension and conflict development\n3. Scene structure and function\n4. Subplot integration\n5. Pacing and momentum\n\nIdentify potential structure issues from what you can see in this sample, and suggest specific improvements that would make this more compelling and marketable as a best-seller."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "sample_text", "type": "String", "expression": "$.sample_text" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "IntegrationEditorNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Integration Editor. Compile and synthesize the assessments of all specialists for manuscript '{{title}}' (ID: {{manuscript_id}}).\n\nExecutive Editor assessment: {{executiveResult}}\n\nContent Assessment: {{contentResult}}\n\nDevelopmental Editor assessment: {{developmentalResult}}\n\nStyle Specialist assessment: {{styleResult}}\n\nPlot Structure analysis: {{plotResult}}\n\nProvide a comprehensive, integrated assessment that:\n1. Summarizes the key findings from all specialists\n2. Identifies the top 5 priority areas for improvement\n3. Outlines a strategic editing plan to elevate this manuscript to best-seller quality\n4. Lists specific recommendations for each major aspect of the manuscript\n5. Estimates overall improvement potential on a scale of 1-10\n\nYour assessment will guide the entire editing process for this manuscript."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "executiveResult", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" },
                    { "name": "contentResult", "type": "String", "expression": "$.ContentAssessmentNode.modelCompletion" },
                    { "name": "developmentalResult", "type": "String", "expression": "$.DevelopmentalEditorNode.modelCompletion" },
                    { "name": "styleResult", "type": "String", "expression": "$.StyleSpecialistNode.modelCompletion" },
                    { "name": "plotResult", "type": "String", "expression": "$.PlotStructureNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Output",
                "name": "FlowOutputNode",
                "inputs": [
                    { "name": "executive_assessment", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" },
                    { "name": "content_assessment", "type": "String", "expression": "$.ContentAssessmentNode.modelCompletion" },
                    { "name": "developmental_assessment", "type": "String", "expression": "$.DevelopmentalEditorNode.modelCompletion" },
                    { "name": "style_assessment", "type": "String", "expression": "$.StyleSpecialistNode.modelCompletion" },
                    { "name": "plot_assessment", "type": "String", "expression": "$.PlotStructureNode.modelCompletion" },
                    { "name": "integrated_assessment", "type": "String", "expression": "$.IntegrationEditorNode.modelCompletion" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_ExecutiveEditor",
                "source": "FlowInputNode",
                "target": "ExecutiveEditorNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "sample_text", "targetInput": "sample_text" }
                    ]
                }
            },
            {
                "name": "Input_to_ContentAssessment",
                "source": "FlowInputNode",
                "target": "ContentAssessmentNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "sample_text", "targetInput": "sample_text" }
                    ]
                }
            },
            {
                "name": "Input_to_DevelopmentalEditor",
                "source": "FlowInputNode",
                "target": "DevelopmentalEditorNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "sample_text", "targetInput": "sample_text" }
                    ]
                }
            },
            {
                "name": "Input_to_StyleSpecialist",
                "source": "FlowInputNode",
                "target": "StyleSpecialistNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "sample_text", "targetInput": "sample_text" }
                    ]
                }
            },
            {
                "name": "Input_to_PlotStructure",
                "source": "FlowInputNode",
                "target": "PlotStructureNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "sample_text", "targetInput": "sample_text" }
                    ]
                }
            },
            {
                "name": "AllNodes_to_IntegrationEditor",
                "source": "FlowInputNode",
                "target": "IntegrationEditorNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" }
                    ]
                }
            },
            {
                "name": "Executive_to_Integration",
                "source": "ExecutiveEditorNode",
                "target": "IntegrationEditorNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "executiveResult" }
                }
            },
            {
                "name": "Content_to_Integration",
                "source": "ContentAssessmentNode",
                "target": "IntegrationEditorNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "contentResult" }
                }
            },
            {
                "name": "Developmental_to_Integration",
                "source": "DevelopmentalEditorNode",
                "target": "IntegrationEditorNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "developmentalResult" }
                }
            },
            {
                "name": "Style_to_Integration",
                "source": "StyleSpecialistNode",
                "target": "IntegrationEditorNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "styleResult" }
                }
            },
            {
                "name": "Plot_to_Integration",
                "source": "PlotStructureNode",
                "target": "IntegrationEditorNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "plotResult" }
                }
            },
            {
                "name": "Results_to_Output",
                "source": "ExecutiveEditorNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "executive_assessment" }
                }
            },
            {
                "name": "ContentResults_to_Output",
                "source": "ContentAssessmentNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "content_assessment" }
                }
            },
            {
                "name": "DevelopmentalResults_to_Output",
                "source": "DevelopmentalEditorNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "developmental_assessment" }
                }
            },
            {
                "name": "StyleResults_to_Output",
                "source": "StyleSpecialistNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "style_assessment" }
                }
            },
            {
                "name": "PlotResults_to_Output",
                "source": "PlotStructureNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "plot_assessment" }
                }
            },
            {
                "name": "IntegrationResults_to_Output",
                "source": "IntegrationEditorNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "integrated_assessment" }
                }
            }
        ]
    }
}
EOF
}

# Generate flow JSON for content improvement
generate_improvement_flow() {
    local flow_name="$1"
    local role_arn="$2"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")

    # Get model IDs
    local executive_model=$(jq -r '.models.executive_editor // .default_model' "$CONFIG_FILE")
    local line_editor_model=$(jq -r '.models.line_editor // .default_model' "$CONFIG_FILE")
    local dialogue_model=$(jq -r '.models.dialogue_specialist // .default_model' "$CONFIG_FILE")
    local pacing_model=$(jq -r '.models.pacing_analyst // .default_model' "$CONFIG_FILE")
    local text_improver_model=$(jq -r '.models.text_improver // .default_model' "$CONFIG_FILE")
    
    # Create flow JSON
    cat > "$FLOW_TEMPLATES_DIR/${flow_name}.json" <<EOF
{
    "name": "$flow_name",
    "description": "Content improvement flow for manuscript editing",
    "executionRoleArn": "$role_arn",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [
                    { "name": "manuscript_id", "type": "String" },
                    { "name": "chunk_id", "type": "String" },
                    { "name": "chunk_text", "type": "String" },
                    { "name": "improvement_focus", "type": "String" },
                    { "name": "editing_notes", "type": "String" }
                ]
            },
            {
                "type": "Prompt",
                "name": "ExecutiveEditorNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Executive Editor coordinating the improvement of manuscript {{manuscript_id}}, chunk {{chunk_id}}. \n\nHere is the text to improve:\n\n{{chunk_text}}\n\nImprovement focus areas: {{improvement_focus}}\n\nPrevious editing notes: {{editing_notes}}\n\nAnalyze this text and determine which specialist should focus on improving it. Consider these aspects:\n1. Is prose quality the main issue? (Line Editor)\n2. Is dialogue the primary weakness? (Dialogue Specialist)\n3. Are there pacing or structure issues? (Pacing Analyst)\n4. Does it need general text enhancement? (Text Improver)\n\nProvide routing instructions as JSON with these fields:\n- primary_issue (string)\n- selected_specialist (string - one of: \"line_editor\", \"dialogue_specialist\", \"pacing_analyst\", \"text_improver\")\n- specific_guidance (string - what the specialist should focus on)"
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "improvement_focus", "type": "String", "expression": "$.improvement_focus" },
                    { "name": "editing_notes", "type": "String", "expression": "$.editing_notes" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Condition",
                "name": "SpecialistRouterNode",
                "configuration": {
                    "condition": {
                        "conditions": [
                            {
                                "expression": "contains(lower(modelCompletion), \"line_editor\")",
                                "name": "LineEditor"
                            },
                            {
                                "expression": "contains(lower(modelCompletion), \"dialogue_specialist\")",
                                "name": "DialogueSpecialist" 
                            },
                            {
                                "expression": "contains(lower(modelCompletion), \"pacing_analyst\")",
                                "name": "PacingAnalyst"
                            },
                            {
                                "expression": "contains(lower(modelCompletion), \"text_improver\")",
                                "name": "TextImprover"
                            },
                            {
                                "name": "default"
                            }
                        ]
                    }
                },
                "inputs": [
                    { "name": "modelCompletion", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" }
                ]
            },
            {
                "type": "Prompt",
                "name": "LineEditorNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$line_editor_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.4, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Line Editor - a prose improvement specialist working to elevate manuscript {{manuscript_id}} to best-seller quality.\n\nHere is chunk {{chunk_id}} to improve:\n\n{{chunk_text}}\n\nImprovement focus areas: {{improvement_focus}}\n\nExecutive Editor guidance: {{executive_guidance}}\n\nAs Line Editor, your task is to enhance the prose quality without changing the fundamental content. Focus on:\n1. Improving sentence structure and flow\n2. Refining word choice and eliminating redundancies\n3. Strengthening imagery and description\n4. Balancing showing vs. telling\n5. Enhancing voice consistency\n\nProvide the revised text with line-by-line improvements. Make meaningful, substantial improvements that elevate the prose to best-seller quality while preserving the author's voice and intent."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "improvement_focus", "type": "String", "expression": "$.improvement_focus" },
                    { "name": "executive_guidance", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "DialogueSpecialistNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$dialogue_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.4, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Dialogue Specialist working to elevate manuscript {{manuscript_id}} to best-seller quality.\n\nHere is chunk {{chunk_id}} to improve:\n\n{{chunk_text}}\n\nImprovement focus areas: {{improvement_focus}}\n\nExecutive Editor guidance: {{executive_guidance}}\n\nAs Dialogue Specialist, your task is to enhance the dialogue while preserving the story and character voices. Focus on:\n1. Making dialogue more authentic and character-specific\n2. Balancing dialogue with action and beats\n3. Improving dialogue tags and reducing unnecessary ones\n4. Ensuring dialogue advances plot or reveals character\n5. Creating more dynamic conversation patterns\n\nProvide the revised text with improved dialogue. Make substantial improvements that elevate the dialogue to best-seller quality while maintaining character consistency."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "improvement_focus", "type": "String", "expression": "$.improvement_focus" },
                    { "name": "executive_guidance", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "PacingAnalystNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$pacing_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.4, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Pacing Analyst working to elevate manuscript {{manuscript_id}} to best-seller quality.\n\nHere is chunk {{chunk_id}} to improve:\n\n{{chunk_text}}\n\nImprovement focus areas: {{improvement_focus}}\n\nExecutive Editor guidance: {{executive_guidance}}\n\nAs Pacing Analyst, your task is to refine the structure and pacing of this section. Focus on:\n1. Balancing exposition, action, and reflection\n2. Adjusting paragraph and sentence length for rhythm\n3. Building and releasing tension effectively\n4. Ensuring proper scene structure and transitions\n5. Trimming or expanding sections as needed for optimal flow\n\nProvide the revised text with improved pacing and structure. Make substantial improvements that enhance the narrative flow and reader engagement while preserving the essential content."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "improvement_focus", "type": "String", "expression": "$.improvement_focus" },
                    { "name": "executive_guidance", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "TextImproverNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$text_improver_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.4, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Text Improver working to elevate manuscript {{manuscript_id}} to best-seller quality.\n\nHere is chunk {{chunk_id}} to improve:\n\n{{chunk_text}}\n\nImprovement focus areas: {{improvement_focus}}\n\nExecutive Editor guidance: {{executive_guidance}}\n\nAs Text Improver, your task is to enhance all aspects of this text. Take a holistic approach, focusing on:\n1. Overall engagement and readability\n2. Character depth and authenticity\n3. Descriptive richness and sensory details\n4. Emotional impact and reader connection\n5. Narrative clarity and purpose\n\nProvide the revised text with comprehensive improvements. Make substantial enhancements that elevate the text to best-seller quality while respecting the author's original intent and voice."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "improvement_focus", "type": "String", "expression": "$.improvement_focus" },
                    { "name": "executive_guidance", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "FinalPolishNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Final Polish Editor for manuscript {{manuscript_id}}, chunk {{chunk_id}}.\n\nHere is the specialist's improved version:\n\n{{specialist_revision}}\n\nReview this revised text and make any final adjustments needed to ensure it is of best-seller quality. Focus on:\n1. Correcting any remaining grammar or punctuation issues\n2. Ensuring perfect consistency in style and voice\n3. Verifying that the improvements align with the intended focus\n4. Making small refinements to perfect the prose\n\nProvide the final polished text along with a brief summary of the improvements made and their impact on the manuscript quality."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "specialist_revision", "type": "String", "expression": "$.specialistOutput" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Output",
                "name": "FlowOutputNode",
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "executive_assessment", "type": "String", "expression": "$.ExecutiveEditorNode.modelCompletion" },
                    { "name": "specialist_revision", "type": "String", "expression": "$.specialistOutput" },
                    { "name": "final_revision", "type": "String", "expression": "$.FinalPolishNode.modelCompletion" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_ExecutiveEditor",
                "source": "FlowInputNode",
                "target": "ExecutiveEditorNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "improvement_focus", "targetInput": "improvement_focus" },
                        { "sourceOutput": "editing_notes", "targetInput": "editing_notes" }
                    ]
                }
            },
            {
                "name": "ExecutiveEditor_to_Router",
                "source": "ExecutiveEditorNode",
                "target": "SpecialistRouterNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "modelCompletion" }
                }
            },
            {
                "name": "Router_to_LineEditor",
                "source": "SpecialistRouterNode",
                "target": "LineEditorNode",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "LineEditor" }
                }
            },
            {
                "name": "Router_to_DialogueSpecialist",
                "source": "SpecialistRouterNode",
                "target": "DialogueSpecialistNode",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "DialogueSpecialist" }
                }
            },
            {
                "name": "Router_to_PacingAnalyst",
                "source": "SpecialistRouterNode",
                "target": "PacingAnalystNode",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "PacingAnalyst" }
                }
            },
            {
                "name": "Router_to_TextImprover",
                "source": "SpecialistRouterNode",
                "target": "TextImproverNode",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "TextImprover" }
                }
            },
            {
                "name": "Router_to_Default",
                "source": "SpecialistRouterNode",
                "target": "TextImproverNode",
                "type": "Conditional",
                "configuration": {
                    "conditional": { "condition": "default" }
                }
            },
            {
                "name": "Input_to_LineEditor",
                "source": "FlowInputNode",
                "target": "LineEditorNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "improvement_focus", "targetInput": "improvement_focus" }
                    ]
                }
            },
            {
                "name": "Input_to_DialogueSpecialist",
                "source": "FlowInputNode",
                "target": "DialogueSpecialistNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "improvement_focus", "targetInput": "improvement_focus" }
                    ]
                }
            },
            {
                "name": "Input_to_PacingAnalyst",
                "source": "FlowInputNode",
                "target": "PacingAnalystNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "improvement_focus", "targetInput": "improvement_focus" }
                    ]
                }
            },
            {
                "name": "Input_to_TextImprover",
                "source": "FlowInputNode",
                "target": "TextImproverNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "improvement_focus", "targetInput": "improvement_focus" }
                    ]
                }
            },
            {
                "name": "LineEditor_to_FinalPolish",
                "source": "LineEditorNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "DialogueSpecialist_to_FinalPolish",
                "source": "DialogueSpecialistNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "PacingAnalyst_to_FinalPolish",
                "source": "PacingAnalystNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "TextImprover_to_FinalPolish",
                "source": "TextImproverNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "Input_to_FinalPolish",
                "source": "FlowInputNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" }
                    ]
                }
            },
            {
                "name": "Input_to_Output",
                "source": "FlowInputNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" }
                    ]
                }
            },
            {
                "name": "ExecutiveEditor_to_Output",
                "source": "ExecutiveEditorNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "executive_assessment" }
                }
            },
            {
                "name": "LineEditor_to_Output",
                "source": "LineEditorNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "DialogueSpecialist_to_Output",
                "source": "DialogueSpecialistNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "PacingAnalyst_to_Output",
                "source": "PacingAnalystNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "TextImprover_to_Output",
                "source": "TextImproverNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "specialist_revision" }
                }
            },
            {
                "name": "FinalPolish_to_Output",
                "source": "FinalPolishNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "final_revision" }
                }
            }
        ]
    }
}
EOF
}

# Generate flow JSON for research flow
generate_research_flow() {
    local flow_name="$1"
    local role_arn="$2"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")

    # Get model IDs
    local executive_model=$(jq -r '.models.executive_editor // .default_model' "$CONFIG_FILE")
    local research_model=$(jq -r '.models.research_specialist // .default_model' "$CONFIG_FILE")
    local verifier_model=$(jq -r '.models.research_verifier // .default_model' "$CONFIG_FILE")
    
    # Create flow JSON
    cat > "$FLOW_TEMPLATES_DIR/${flow_name}.json" <<EOF
{
    "name": "$flow_name",
    "description": "Research flow for manuscript enhancement",
    "executionRoleArn": "$role_arn",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [
                    { "name": "manuscript_id", "type": "String" },
                    { "name": "chunk_id", "type": "String" },
                    { "name": "chunk_text", "type": "String" },
                    { "name": "research_topic", "type": "String" }
                ]
            },
            {
                "type": "Prompt",
                "name": "ResearchPlannerNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Research Planner for manuscript {{manuscript_id}}. For chunk {{chunk_id}}, we need to research the topic: {{research_topic}}.\n\nHere is the relevant text from the manuscript:\n\n{{chunk_text}}\n\nBased on this text and the research topic, develop a research plan with:\n1. 3-5 specific search queries that will yield the most relevant information\n2. Key aspects of the topic that need verification or enhancement\n3. How this research will improve the manuscript\n\nProvide your research plan in JSON format with the fields: search_queries (array), key_aspects (array), improvement_goals (array)."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "research_topic", "type": "String", "expression": "$.research_topic" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt", 
                "name": "ResearchSpecialistNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$research_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Research Specialist for manuscript {{manuscript_id}}, chunk {{chunk_id}}. Your task is to provide accurate research on: {{research_topic}}.\n\nResearch plan: {{research_plan}}\n\nBased on your knowledge and expertise:\n\n1. Provide factual information about this topic relevant to the manuscript\n2. Include specific details that would enhance authenticity (dates, terminology, processes, locations, etc.)\n3. Note any potential inaccuracies in the manuscript text that should be corrected\n4. Suggest how to incorporate this research naturally into the narrative\n\nPresent your research findings in a clear, structured format that will be directly useful for improving the manuscript."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "research_topic", "type": "String", "expression": "$.research_topic" },
                    { "name": "research_plan", "type": "String", "expression": "$.ResearchPlannerNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "VerificationNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$verifier_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Research Verifier for manuscript {{manuscript_id}}, chunk {{chunk_id}}. Your task is to verify the research findings on: {{research_topic}}.\n\nOriginal manuscript text:\n{{chunk_text}}\n\nResearch findings:\n{{research_findings}}\n\nPlease verify these research findings by:\n1. Identifying any potential inaccuracies or questionable claims\n2. Noting where the findings seem well-supported by general knowledge\n3. Suggesting additional aspects that might need verification\n4. Rating the overall reliability of the findings (1-10 scale)\n\nProvide your verification report in a clear, structured format that highlights both strengths and potential issues with the research."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "research_topic", "type": "String", "expression": "$.research_topic" },
                    { "name": "research_findings", "type": "String", "expression": "$.ResearchSpecialistNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "IntegrationNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.4, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Research Integration Editor for manuscript {{manuscript_id}}, chunk {{chunk_id}}. Your task is to create an integration plan showing how to incorporate the verified research on {{research_topic}} into the manuscript.\n\nOriginal manuscript text:\n{{chunk_text}}\n\nResearch findings:\n{{research_findings}}\n\nVerification report:\n{{verification_report}}\n\nCreate a detailed integration plan that:\n1. Identifies specific passages where research can be incorporated\n2. Suggests rewrites of these passages to include accurate details\n3. Shows how to fix any inaccuracies that were found\n4. Demonstrates how to weave in facts naturally without disrupting narrative flow\n\nProvide both specific examples of text revisions and general guidelines for incorporating this research throughout the manuscript."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "chunk_text", "type": "String", "expression": "$.chunk_text" },
                    { "name": "research_topic", "type": "String", "expression": "$.research_topic" },
                    { "name": "research_findings", "type": "String", "expression": "$.ResearchSpecialistNode.modelCompletion" },
                    { "name": "verification_report", "type": "String", "expression": "$.VerificationNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Output",
                "name": "FlowOutputNode",
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "chunk_id", "type": "String", "expression": "$.chunk_id" },
                    { "name": "research_plan", "type": "String", "expression": "$.ResearchPlannerNode.modelCompletion" },
                    { "name": "research_findings", "type": "String", "expression": "$.ResearchSpecialistNode.modelCompletion" },
                    { "name": "verification_report", "type": "String", "expression": "$.VerificationNode.modelCompletion" },
                    { "name": "integration_plan", "type": "String", "expression": "$.IntegrationNode.modelCompletion" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_ResearchPlanner",
                "source": "FlowInputNode",
                "target": "ResearchPlannerNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "research_topic", "targetInput": "research_topic" }
                    ]
                }
            },
            {
                "name": "ResearchPlanner_to_ResearchSpecialist",
                "source": "ResearchPlannerNode",
                "target": "ResearchSpecialistNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "research_plan" }
                }
            },
            {
                "name": "Input_to_ResearchSpecialist",
                "source": "FlowInputNode",
                "target": "ResearchSpecialistNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "research_topic", "targetInput": "research_topic" }
                    ]
                }
            },
            {
                "name": "ResearchSpecialist_to_Verification",
                "source": "ResearchSpecialistNode",
                "target": "VerificationNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "research_findings" }
                }
            },
            {
                "name": "Input_to_Verification",
                "source": "FlowInputNode",
                "target": "VerificationNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "research_topic", "targetInput": "research_topic" }
                    ]
                }
            },
            {
                "name": "ResearchSpecialist_to_Integration",
                "source": "ResearchSpecialistNode",
                "target": "IntegrationNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "research_findings" }
                }
            },
            {
                "name": "Verification_to_Integration",
                "source": "VerificationNode",
                "target": "IntegrationNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "verification_report" }
                }
            },
            {
                "name": "Input_to_Integration",
                "source": "FlowInputNode",
                "target": "IntegrationNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" },
                        { "sourceOutput": "chunk_text", "targetInput": "chunk_text" },
                        { "sourceOutput": "research_topic", "targetInput": "research_topic" }
                    ]
                }
            },
            {
                "name": "Input_to_Output",
                "source": "FlowInputNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "chunk_id", "targetInput": "chunk_id" }
                    ]
                }
            },
            {
                "name": "ResearchPlanner_to_Output",
                "source": "ResearchPlannerNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "research_plan" }
                }
            },
            {
                "name": "ResearchSpecialist_to_Output",
                "source": "ResearchSpecialistNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "research_findings" }
                }
            },
            {
                "name": "Verification_to_Output",
                "source": "VerificationNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "verification_report" }
                }
            },
            {
                "name": "Integration_to_Output",
                "source": "IntegrationNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "integration_plan" }
                }
            }
        ]
    }
}
EOF
}

# Generate flow JSON for final review flow
generate_finalization_flow() {
    local flow_name="$1"
    local role_arn="$2"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")

    # Get model IDs
    local executive_model=$(jq -r '.models.executive_editor // .default_model' "$CONFIG_FILE")
    local continuity_model=$(jq -r '.models.continuity_checker // .default_model' "$CONFIG_FILE")
    local proofreader_model=$(jq -r '.models.proofreader // .default_model' "$CONFIG_FILE")
    
    # Create flow JSON
    cat > "$FLOW_TEMPLATES_DIR/${flow_name}.json" <<EOF
{
    "name": "$flow_name",
    "description": "Final review flow for manuscript completion",
    "executionRoleArn": "$role_arn",
    "definition": {
        "nodes": [
            {
                "type": "Input",
                "name": "FlowInputNode",
                "outputs": [
                    { "name": "manuscript_id", "type": "String" },
                    { "name": "title", "type": "String" },
                    { "name": "final_text", "type": "String" },
                    { "name": "previous_assessment", "type": "String" }
                ]
            },
            {
                "type": "Prompt",
                "name": "ContinuityCheckerNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$continuity_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Continuity Checker for manuscript '{{title}}' (ID: {{manuscript_id}}). Review this section of the text for continuity issues:\n\n{{final_text}}\n\nCheck for:\n1. Character continuity (names, traits, backgrounds)\n2. Plot continuity (events, timelines, cause-effect)\n3. Setting continuity (locations, descriptions, time of day)\n4. Object continuity (items mentioned then forgotten)\n5. Thematic continuity (consistent themes and motifs)\n\nProvide a detailed continuity report highlighting any issues found and suggest fixes. Be thorough - continuity problems can seriously undermine a manuscript's quality."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "final_text", "type": "String", "expression": "$.final_text" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "ProofreaderNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$proofreader_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Proofreader for manuscript '{{title}}' (ID: {{manuscript_id}}). Carefully proofread this text:\n\n{{final_text}}\n\nCheck for and correct:\n1. Spelling errors\n2. Grammar issues\n3. Punctuation mistakes\n4. Formatting inconsistencies\n5. Awkward phrasings\n\nProvide the corrected text with all errors fixed. Also include a brief summary of the types of corrections made. Maintain the original voice and style while ensuring technical perfection."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "final_text", "type": "String", "expression": "$.final_text" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "QualityAssessorNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Quality Assessor for manuscript '{{title}}' (ID: {{manuscript_id}}). Compare the current text quality to the previous assessment:\n\nPrevious assessment: {{previous_assessment}}\n\nCurrent text:\n{{final_text}}\n\nContinuity check: {{continuity_report}}\n\nProofreading results: {{proofreading_results}}\n\nEvaluate the following quality dimensions (score 1-10 for each):\n1. Style consistency and voice\n2. Narrative coherence and flow\n3. Character development and authenticity\n4. Dialogue quality and effectiveness\n5. Pacing and engagement\n6. Descriptive richness and imagery\n7. Emotional impact\n8. Technical proficiency (grammar, spelling)\n9. Overall literary quality\n10. Best-seller potential\n\nProvide your quality assessment in JSON format including scores and detailed comments for each dimension, plus an overall evaluation of improvement compared to the previous assessment."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "final_text", "type": "String", "expression": "$.final_text" },
                    { "name": "previous_assessment", "type": "String", "expression": "$.previous_assessment" },
                    { "name": "continuity_report", "type": "String", "expression": "$.ContinuityCheckerNode.modelCompletion" },
                    { "name": "proofreading_results", "type": "String", "expression": "$.ProofreaderNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "FinalPolishNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.3, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Final Polish Editor for manuscript '{{title}}' (ID: {{manuscript_id}}). Incorporate all feedback to produce the final polished version:\n\nCurrent text:\n{{final_text}}\n\nContinuity issues to fix: {{continuity_report}}\n\nProofreading corrections: {{proofreading_results}}\n\nQuality assessment: {{quality_assessment}}\n\nProvide the definitive, polished version of this text that incorporates all necessary corrections and improvements. This should be publication-ready, best-seller quality prose. Do not include explanatory notes or your thought process - just provide the finished, polished text."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "final_text", "type": "String", "expression": "$.final_text" },
                    { "name": "continuity_report", "type": "String", "expression": "$.ContinuityCheckerNode.modelCompletion" },
                    { "name": "proofreading_results", "type": "String", "expression": "$.ProofreaderNode.modelCompletion" },
                    { "name": "quality_assessment", "type": "String", "expression": "$.QualityAssessorNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Prompt",
                "name": "ExecutiveSummaryNode",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "$executive_model",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {
                                    "text": { "temperature": 0.2, "topP": 0.9 }
                                },
                                "templateConfiguration": {
                                    "text": {
                                        "text": "You are the Executive Editor for manuscript '{{title}}' (ID: {{manuscript_id}}). Prepare a final executive summary of the editing process and results:\n\nContinuity assessment: {{continuity_report}}\n\nProofreading assessment: {{proofreading_results}}\n\nQuality assessment: {{quality_assessment}}\n\nProvide a comprehensive executive summary that includes:\n1. Key improvements made to the manuscript\n2. Remaining strengths and weaknesses\n3. Overall assessment of best-seller potential\n4. Recommendations for the author\n5. Market positioning advice\n\nThis summary should be professional, honest, and constructive, offering both praise for achievements and candid guidance for any remaining improvements."
                                    }
                                }
                            }
                        }
                    }
                },
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "continuity_report", "type": "String", "expression": "$.ContinuityCheckerNode.modelCompletion" },
                    { "name": "proofreading_results", "type": "String", "expression": "$.ProofreaderNode.modelCompletion" },
                    { "name": "quality_assessment", "type": "String", "expression": "$.QualityAssessorNode.modelCompletion" }
                ],
                "outputs": [{ "name": "modelCompletion", "type": "String" }]
            },
            {
                "type": "Output",
                "name": "FlowOutputNode",
                "inputs": [
                    { "name": "manuscript_id", "type": "String", "expression": "$.manuscript_id" },
                    { "name": "title", "type": "String", "expression": "$.title" },
                    { "name": "continuity_report", "type": "String", "expression": "$.ContinuityCheckerNode.modelCompletion" },
                    { "name": "proofreading_results", "type": "String", "expression": "$.ProofreaderNode.modelCompletion" },
                    { "name": "quality_assessment", "type": "String", "expression": "$.QualityAssessorNode.modelCompletion" },
                    { "name": "final_polished_text", "type": "String", "expression": "$.FinalPolishNode.modelCompletion" },
                    { "name": "executive_summary", "type": "String", "expression": "$.ExecutiveSummaryNode.modelCompletion" }
                ]
            }
        ],
        "connections": [
            {
                "name": "Input_to_ContinuityChecker",
                "source": "FlowInputNode",
                "target": "ContinuityCheckerNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "final_text", "targetInput": "final_text" }
                    ]
                }
            },
            {
                "name": "Input_to_Proofreader",
                "source": "FlowInputNode",
                "target": "ProofreaderNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "final_text", "targetInput": "final_text" }
                    ]
                }
            },
            {
                "name": "ContinuityChecker_to_QualityAssessor",
                "source": "ContinuityCheckerNode",
                "target": "QualityAssessorNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "continuity_report" }
                }
            },
            {
                "name": "Proofreader_to_QualityAssessor",
                "source": "ProofreaderNode",
                "target": "QualityAssessorNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "proofreading_results" }
                }
            },
            {
                "name": "Input_to_QualityAssessor",
                "source": "FlowInputNode",
                "target": "QualityAssessorNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "final_text", "targetInput": "final_text" },
                        { "sourceOutput": "previous_assessment", "targetInput": "previous_assessment" }
                    ]
                }
            },
            {
                "name": "QualityAssessor_to_FinalPolish",
                "source": "QualityAssessorNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "quality_assessment" }
                }
            },
            {
                "name": "ContinuityChecker_to_FinalPolish",
                "source": "ContinuityCheckerNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "continuity_report" }
                }
            },
            {
                "name": "Proofreader_to_FinalPolish",
                "source": "ProofreaderNode", 
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "proofreading_results" }
                }
            },
            {
                "name": "Input_to_FinalPolish",
                "source": "FlowInputNode",
                "target": "FinalPolishNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" },
                        { "sourceOutput": "final_text", "targetInput": "final_text" }
                    ]
                }
            },
            {
                "name": "ContinuityChecker_to_ExecutiveSummary",
                "source": "ContinuityCheckerNode",
                "target": "ExecutiveSummaryNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "continuity_report" }
                }
            },
            {
                "name": "Proofreader_to_ExecutiveSummary",
                "source": "ProofreaderNode",
                "target": "ExecutiveSummaryNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "proofreading_results" }
                }
            },
            {
                "name": "QualityAssessor_to_ExecutiveSummary",
                "source": "QualityAssessorNode",
                "target": "ExecutiveSummaryNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "quality_assessment" }
                }
            },
            {
                "name": "Input_to_ExecutiveSummary",
                "source": "FlowInputNode",
                "target": "ExecutiveSummaryNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" }
                    ]
                }
            },
            {
                "name": "Input_to_Output",
                "source": "FlowInputNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": [
                        { "sourceOutput": "manuscript_id", "targetInput": "manuscript_id" },
                        { "sourceOutput": "title", "targetInput": "title" }
                    ]
                }
            },
            {
                "name": "ContinuityChecker_to_Output",
                "source": "ContinuityCheckerNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "continuity_report" }
                }
            },
            {
                "name": "Proofreader_to_Output",
                "source": "ProofreaderNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "proofreading_results" }
                }
            },
            {
                "name": "QualityAssessor_to_Output",
                "source": "QualityAssessorNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "quality_assessment" }
                }
            },
            {
                "name": "FinalPolish_to_Output",
                "source": "FinalPolishNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "final_polished_text" }
                }
            },
            {
                "name": "ExecutiveSummary_to_Output",
                "source": "ExecutiveSummaryNode",
                "target": "FlowOutputNode",
                "type": "Data",
                "configuration": {
                    "data": { "sourceOutput": "modelCompletion", "targetInput": "executive_summary" }
                }
            }
        ]
    }
}
EOF
}

# Create flow in AWS Bedrock
create_flow() {
    local flow_name="$1"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")

    echo -e "${YELLOW}Creating flow $flow_name...${NC}"

    # Put the template into a temporary file
    local template_file="${FLOW_TEMPLATES_DIR}/${flow_name}.json"
    
    # Create flow using template
    local response
    response=$(aws bedrock-agent create-flow \
        --region "$region" \
        --profile "$profile" \
        --cli-input-json "file://${template_file}" 2>&1) || {
            echo -e "${RED}Error creating flow: $response${NC}"
            echo "flow-placeholder:1:alias-placeholder"
            return
        }
    
    # Extract flow ID safely with grep instead of jq for better error handling
    local flow_id
    flow_id=$(echo "$response" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [ -z "$flow_id" ]; then
        echo -e "${RED}Could not extract flow ID from response${NC}"
        echo "flow-placeholder:1:alias-placeholder"
        return
    fi
    
    echo -e "${GREEN}Flow created with ID: $flow_id${NC}"

    echo -e "${YELLOW}Preparing flow...${NC}"
    aws bedrock-agent prepare-flow \
        --region "$region" \
        --profile "$profile" \
        --flow-identifier "$flow_id" >/dev/null 2>&1 || {
            echo -e "${RED}Error preparing flow${NC}"
            echo "$flow_id:1:alias-placeholder"
            return
        }

    echo -e "${YELLOW}Creating flow version...${NC}"
    local version_response
    version_response=$(aws bedrock-agent create-flow-version \
        --region "$region" \
        --profile "$profile" \
        --flow-identifier "$flow_id" 2>&1) || {
            echo -e "${RED}Error creating flow version: $version_response${NC}"
            echo "$flow_id:1:alias-placeholder"
            return
        }
        
    # Extract version safely
    local version="1"
    local extracted_version
    extracted_version=$(echo "$version_response" | grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [ -n "$extracted_version" ]; then
        version="$extracted_version"
    fi
    
    echo -e "${GREEN}Created version: $version${NC}"

    echo -e "${YELLOW}Creating flow alias...${NC}"
    local alias_response
    alias_response=$(aws bedrock-agent create-flow-alias \
        --region "$region" \
        --profile "$profile" \
        --flow-identifier "$flow_id" \
        --name "latest" \
        --routing-configuration "[{\"flowVersion\": \"$version\"}]" 2>&1) || {
            echo -e "${RED}Error creating alias: $alias_response${NC}"
            echo "$flow_id:$version:alias-placeholder"
            return
        }
        
    # Extract alias ID safely
    local alias_id="alias-placeholder"
    local extracted_alias
    extracted_alias=$(echo "$alias_response" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | cut -d'"' -f4)
    
    if [ -n "$extracted_alias" ]; then
        alias_id="$extracted_alias"
    fi
    
    echo -e "${GREEN}Created alias: $alias_id${NC}"

    # Return flow ID, version, and alias ID
    echo "$flow_id:$version:$alias_id"
}

# Process manuscript through chunking and assessment
process_manuscript() {
    local project_name="$1"
    local manuscript_file="$2"
    local title="$3"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    local chunk_size=$(jq -r '.chunk_size' "$CONFIG_FILE")
    local chunk_overlap=$(jq -r '.chunk_overlap' "$CONFIG_FILE")
    local model_id=$(jq -r '.default_model' "$CONFIG_FILE")
    local table_prefix=$(jq -r '.table_prefix' "$CONFIG_FILE")
    
    # Store manuscript in the manuscripts directory
    local manuscript_copy="$MANUSCRIPT_DIR/${project_name}_original.txt"
    cp "$manuscript_file" "$manuscript_copy"
    
    echo -e "${YELLOW}Processing manuscript: $title (Project ID: $project_name)${NC}"
    
    # Generate Python chunking script if not already created
    if [ ! -f "$TEMP_DIR/chunk_manuscript.py" ]; then
        generate_chunking_script
    fi
    
    # Generate Python assessment script if not already created
    if [ ! -f "$TEMP_DIR/assess_manuscript.py" ]; then
        generate_assessment_script
    fi
    
    # Generate Python research script if not already created
    if [ ! -f "$TEMP_DIR/web_research.py" ]; then
        generate_research_script
    fi
    
    # Run the chunking script
    echo -e "${YELLOW}Chunking manuscript into manageable pieces...${NC}"
    python3 "$TEMP_DIR/chunk_manuscript.py" "$manuscript_copy" --output_dir "$CHUNKS_DIR" --project_name "$project_name" --max_tokens "$chunk_size" --overlap_tokens "$chunk_overlap"
    
    # Check if chunking was successful
    if [ ! -f "$CHUNKS_DIR/${project_name}_metadata.json" ]; then
        echo -e "${RED}Error: Chunking failed. Metadata file not found.${NC}"
        return 1
    fi
    
    # Extract metadata
    local total_chunks=$(jq -r '.total_chunks' "$CHUNKS_DIR/${project_name}_metadata.json")
    echo -e "${GREEN}Manuscript split into $total_chunks chunks.${NC}"
    
    # Update DynamoDB with metadata
    aws dynamodb update-item \
        --table-name "${table_prefix}_${project_name}_state" \
        --key '{"manuscript_id": {"S": "'"$project_name"'"}, "chunk_id": {"S": "metadata"}}' \
        --update-expression "SET title = :t, total_chunks = :c, current_phase = :p, updated_at = :u" \
        --expression-attribute-values '{
            ":t": {"S": "'"$title"'"},
            ":c": {"N": "'"$total_chunks"'"},
            ":p": {"S": "assessment"},
            ":u": {"S": "'"$(date -u +'%Y-%m-%dT%H:%M:%SZ')"'"}
        }' \
        --region "$region" \
        --profile "$profile" \
        --output json
    
    # Run initial manuscript assessment
    echo -e "${YELLOW}Performing initial manuscript assessment...${NC}"
    python3 "$TEMP_DIR/assess_manuscript.py" --chunks_dir "$CHUNKS_DIR" --project_name "$project_name" --region "$region" --model_id "$model_id" --table_prefix "$table_prefix"
    
    # Check if assessment was successful
    if [ ! -f "$CHUNKS_DIR/${project_name}_assessment.json" ]; then
        echo -e "${RED}Error: Assessment failed. Assessment file not found.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Initial manuscript assessment complete.${NC}"
    
    # Return success
    return 0
}

# Create new deployment
create_new_deployment() {
    echo -e "${BLUE}=== Create New Novel Editing Deployment ===${NC}"

    # Check if AWS CLI is configured
    check_aws_configuration

    # Prompt for project name
    read -p "Enter project name: " project_name_raw

    # Validate and sanitize project name
    if [ -z "$project_name_raw" ]; then
        echo -e "${RED}Error: Project name cannot be empty.${NC}"
        return 1
    fi

    # Convert to valid flow name (lowercase, no spaces, alphanumeric and hyphens only)
    project_name=$(echo "$project_name_raw" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9-')

    # Check if project already exists
    if jq -e --arg name "$project_name" '.flows[] | select(. == $name)' "$FLOWS_LIST_FILE" > /dev/null; then
        echo -e "${RED}Error: A project with name '$project_name' already exists.${NC}"
        return 1
    fi

    # Prompt for manuscript file
    read -p "Enter path to manuscript file: " manuscript_file

    # Validate manuscript file
    if [ ! -f "$manuscript_file" ]; then
        echo -e "${RED}Error: Manuscript file not found.${NC}"
        return 1
    fi

    # Prompt for manuscript title
    read -p "Enter manuscript title: " manuscript_title

    # Validate manuscript title
    if [ -z "$manuscript_title" ]; then
        echo -e "${RED}Error: Manuscript title cannot be empty.${NC}"
        return 1
    fi

    # First, check if we need to create IAM role
    initialize_iam_role

    # Get IAM role ARN
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    local role_name=$(jq -r '.role_name' "$CONFIG_FILE")
    local role_arn=$(aws iam get-role --role-name "$role_name" --region "$region" --profile "$profile" --query 'Role.Arn' --output text)

    # Generate flow templates for each phase
    echo -e "${YELLOW}Generating flow templates...${NC}"
    generate_assessment_flow "${project_name}-assessment" "$role_arn"
    generate_improvement_flow "${project_name}-improvement" "$role_arn"
    generate_research_flow "${project_name}-research" "$role_arn"
    generate_finalization_flow "${project_name}-finalization" "$role_arn"

    # Create flows in AWS Bedrock
    echo -e "${YELLOW}Creating flows in AWS Bedrock...${NC}"

    # Track flow IDs, versions, and alias IDs
    declare -A flow_info

    # Create each flow
    for phase in assessment improvement research finalization; do
        echo -e "${BLUE}Creating ${phase} flow...${NC}"
        local result=$(create_flow "${project_name}-${phase}")
        flow_info["${phase}"]="$result"
    done

    # Create DynamoDB tables for project
    initialize_dynamodb "$project_name"

    # Process manuscript (chunk and assess)
    process_manuscript "$project_name" "$manuscript_file" "$manuscript_title"

    # Add project to flows list
    jq --arg name "$project_name" '.flows += [$name]' "$FLOWS_LIST_FILE" > "${FLOWS_LIST_FILE}.tmp" && mv "${FLOWS_LIST_FILE}.tmp" "$FLOWS_LIST_FILE"

    # Save flow details to project configuration file
    local project_config="${project_name}_config.json"
    cat > "$project_config" <<EOF
{
    "project_name": "$project_name",
    "title": "$manuscript_title",
    "created_at": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "original_manuscript": "$manuscript_copy",
    "flows": {
EOF

    # Add each flow's details
    local first=true
    for phase in assessment improvement research finalization; do
        IFS=':' read -r id version alias <<< "${flow_info["${phase}"]}"
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$project_config"
        fi
        cat >> "$project_config" <<EOF
        "${phase}": {
            "flow_id": "$id",
            "version": "$version",
            "alias_id": "$alias"
        }
EOF
    done

    # Close the JSON file
    cat >> "$project_config" <<EOF
    }
}
EOF

    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${GREEN}Project configuration saved to $project_config${NC}"
    echo -e "${BLUE}To improve your manuscript:${NC}"
    echo -e "  1. Use the 'Process Manuscript' option from the main menu"
    echo -e "  2. Select this project to start editing"
    echo -e "  3. The system will process chunks in parallel and provide comprehensive improvements"
}

# Process manuscript chunks in parallel
process_manuscript_chunks() {
    local project_name="$1"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    local table_prefix=$(jq -r '.table_prefix' "$CONFIG_FILE")
    
    # Load project configuration
    local project_config="${project_name}_config.json"
    if [ ! -f "$project_config" ]; then
        echo -e "${RED}Error: Project configuration file not found: $project_config${NC}"
        return 1
    fi
    
    # Get metadata
    local metadata_file="$CHUNKS_DIR/${project_name}_metadata.json"
    if [ ! -f "$metadata_file" ]; then
        echo -e "${RED}Error: Metadata file not found: $metadata_file${NC}"
        return 1
    fi
    
    local total_chunks=$(jq -r '.total_chunks' "$metadata_file")
    local chunk_files=($(jq -r '.chunk_files[]' "$metadata_file"))
    
    # Get assessment data
    local assessment_file="$CHUNKS_DIR/${project_name}_assessment.json"
    if [ ! -f "$assessment_file" ]; then
        echo -e "${RED}Error: Assessment file not found: $assessment_file${NC}"
        return 1
    fi
    
    # Extract flow IDs
    local improvement_flow_id=$(jq -r '.flows.improvement.flow_id' "$project_config")
    local improvement_alias_id=$(jq -r '.flows.improvement.alias_id' "$project_config")
    local research_flow_id=$(jq -r '.flows.research.flow_id' "$project_config")
    local research_alias_id=$(jq -r '.flows.research.alias_id' "$project_config")
    
    # Extract improvement focus from assessment
    local improvement_focus=$(jq -r '.content_assessment.improvement_recommendations | join(", ")' "$assessment_file")
    if [ "$improvement_focus" == "null" ]; then
        improvement_focus="Improve overall quality, pacing, character development, dialogue, and prose style"
    fi
    
    echo -e "${YELLOW}Processing ${#chunk_files[@]} chunks for manuscript improvement...${NC}"
    echo -e "${BLUE}Improvement focus: $improvement_focus${NC}"
    
    # Setup parallel processing with a max of 5 concurrent processes
    local max_parallel=5
    local running=0
    local completed=0
    local pids=()
    
    # Create output directory for improved chunks
    local improved_dir="$CHUNKS_DIR/${project_name}_improved"
    mkdir -p "$improved_dir"
    
    # Process each chunk
    for ((i=0; i<${#chunk_files[@]}; i++)); do
        local chunk_file="${chunk_files[$i]}"
        local chunk_id="chunk_$(printf "%04d" $((i+1)))"
        local chunk_text=$(cat "$chunk_file")
        local improved_file="$improved_dir/${chunk_id}.txt"
        
        # Skip if already processed
        if [ -f "$improved_file" ]; then
            echo -e "${GREEN}Chunk $chunk_id already processed, skipping...${NC}"
            ((completed++))
            continue
        }
        
        echo -e "${YELLOW}Processing chunk $chunk_id...${NC}"
        
        # Create input JSON for flow invocation
        local input_json=$(cat <<EOF
[
  {
    "content": {
      "manuscript_id": "$project_name",
      "chunk_id": "$chunk_id",
      "chunk_text": $(jq -Rs '.' <<< "$chunk_text"),
      "improvement_focus": "$improvement_focus",
      "editing_notes": ""
    },
    "nodeName": "FlowInputNode",
    "nodeOutputNames": ["manuscript_id", "chunk_id", "chunk_text", "improvement_focus", "editing_notes"]
  }
]
EOF
)
        
        # Save input to temp file
        local input_file="$TEMP_DIR/${project_name}_${chunk_id}_input.json"
        echo "$input_json" > "$input_file"
        
        # Process in background
        {
            # Invoke improvement flow
            local response=$(aws bedrock-agent-runtime invoke-flow \
                --flow-identifier "$improvement_flow_id" \
                --flow-alias-identifier "$improvement_alias_id" \
                --inputs file://"$input_file" \
                --region "$region" \
                --profile "$profile" 2>/dev/null)
            
            # Extract final revision
            local final_revision=$(echo "$response" | jq -r '.outputs[] | select(.name == "final_revision") | .content')
            
            # Save improved chunk
            echo "$final_revision" > "$improved_file"
            
            # Update DynamoDB 
            aws dynamodb put-item \
                --table-name "${table_prefix}_${project_name}_edits" \
                --item '{
                    "edit_id": {"S": "'"$chunk_id"'"},
                    "manuscript_id": {"S": "'"$project_name"'"},
                    "timestamp": {"S": "'"$(date -u +'%Y-%m-%dT%H:%M:%SZ')"'"},
                    "original_text": {"S": '"$(jq -Rs '.' <<< "$chunk_text")''},
                    "improved_text": {"S": '"$(jq -Rs '.' <<< "$final_revision")''},
                    "improvement_type": {"S": "content"}
                }' \
                --region "$region" \
                --profile "$profile" > /dev/null
            
            # Update progress
            aws dynamodb update-item \
                --table-name "${table_prefix}_${project_name}_state" \
                --key '{"manuscript_id": {"S": "'"$project_name"'"}, "chunk_id": {"S": "metadata"}}' \
                --update-expression "SET chunks_processed = chunks_processed + :val, updated_at = :u" \
                --expression-attribute-values '{
                    ":val": {"N": "1"},
                    ":u": {"S": "'"$(date -u +'%Y-%m-%dT%H:%M:%SZ')"'"}
                }' \
                --region "$region" \
                --profile "$profile" > /dev/null
            
            echo -e "${GREEN}Completed chunk $chunk_id${NC}"
        } &
        
        # Store process ID
        pids+=($!)
        ((running++))
        
        # Limit parallel processes
        if [ $running -ge $max_parallel ]; then
            # Wait for any process to complete
            wait -n "${pids[@]}"
            
            # Update running count
            running=$(ps | grep -c "${pids[@]}")
            ((completed++))
            
            # Display progress
            echo -e "${BLUE}Progress: $completed / ${#chunk_files[@]} chunks processed ($(( completed * 100 / ${#chunk_files[@]} ))%)${NC}"
        fi
    done
    
    # Wait for remaining processes
    for pid in "${pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            wait $pid
            ((completed++))
        fi
    done
    
    echo -e "${GREEN}All chunks processed. Generating final manuscript...${NC}"
    
    # Combine improved chunks into final manuscript
    local final_manuscript="$MANUSCRIPT_DIR/${project_name}_improved.txt"
    for ((i=0; i<${#chunk_files[@]}; i++)); do
        local chunk_id="chunk_$(printf "%04d" $((i+1)))"
        local improved_file="$improved_dir/${chunk_id}.txt"
        
        if [ -f "$improved_file" ]; then
            cat "$improved_file" >> "$final_manuscript"
            echo -e "\n\n" >> "$final_manuscript"
        else
            echo -e "${RED}Warning: Improved file not found for chunk $chunk_id${NC}"
        fi
    done
    
    # Update project state
    aws dynamodb update-item \
        --table-name "${table_prefix}_${project_name}_state" \
        --key '{"manuscript_id": {"S": "'"$project_name"'"}, "chunk_id": {"S": "metadata"}}' \
        --update-expression "SET current_phase = :p, updated_at = :u" \
        --expression-attribute-values '{
            ":p": {"S": "finalization"},
            ":u": {"S": "'"$(date -u +'%Y-%m-%dT%H:%M:%SZ')"'"}
        }' \
        --region "$region" \
        --profile "$profile" > /dev/null
    
    echo -e "${GREEN}Final improved manuscript saved to: $final_manuscript${NC}"
    
    return 0
}

# Finalize manuscript
finalize_manuscript() {
    local project_name="$1"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    local table_prefix=$(jq -r '.table_prefix' "$CONFIG_FILE")
    
    # Load project configuration
    local project_config="${project_name}_config.json"
    if [ ! -f "$project_config" ]; then
        echo -e "${RED}Error: Project configuration file not found: $project_config${NC}"
        return 1
    fi
    
    # Get manuscript title
    local title=$(jq -r '.title' "$project_config")
    
    # Get final manuscript
    local final_manuscript="$MANUSCRIPT_DIR/${project_name}_improved.txt"
    if [ ! -f "$final_manuscript" ]; then
        echo -e "${RED}Error: Improved manuscript not found: $final_manuscript${NC}"
        return 1
    fi
    
    # Get assessment data
    local assessment_file="$CHUNKS_DIR/${project_name}_assessment.json"
    if [ ! -f "$assessment_file" ]; then
        echo -e "${RED}Error: Assessment file not found: $assessment_file${NC}"
        return 1
    fi
    
    # Extract flow IDs
    local finalization_flow_id=$(jq -r '.flows.finalization.flow_id' "$project_config")
    local finalization_alias_id=$(jq -r '.flows.finalization.alias_id' "$project_config")
    
    echo -e "${YELLOW}Finalizing manuscript for project $project_name...${NC}"
    
    # Need to process manuscript in chunks for finalization too
    local chunk_size=20000
    local total_size=$(wc -c < "$final_manuscript")
    local num_chunks=$(( (total_size + chunk_size - 1) / chunk_size ))
    
    echo -e "${BLUE}Splitting manuscript into $num_chunks chunks for final review...${NC}"
    
    # Create finalization directory
    local final_dir="$CHUNKS_DIR/${project_name}_final"
    mkdir -p "$final_dir"
    
    # Split manuscript into chunks for processing
    split -b "$chunk_size" "$final_manuscript" "$final_dir/final_chunk_"
    
    # Get previous assessment
    local previous_assessment=$(jq -c '.content_assessment' "$assessment_file")
    
    # Process each chunk for finalization
    local final_outputs=()
    local chunk_files=("$final_dir"/final_chunk_*)
    
    for chunk_file in "${chunk_files[@]}"; do
        local chunk_id=$(basename "$chunk_file")
        local chunk_text=$(cat "$chunk_file")
        
        echo -e "${YELLOW}Finalizing chunk $chunk_id...${NC}"
        
        # Create input JSON for flow invocation
        local input_json=$(cat <<EOF
[
  {
    "content": {
      "manuscript_id": "$project_name",
      "title": "$title",
      "final_text": $(jq -Rs '.' <<< "$chunk_text"),
      "previous_assessment": $(jq -Rs '.' <<< "$previous_assessment")
    },
    "nodeName": "FlowInputNode",
    "nodeOutputNames": ["manuscript_id", "title", "final_text", "previous_assessment"]
  }
]
EOF
)
        
        # Save input to temp file
        local input_file="$TEMP_DIR/${project_name}_${chunk_id}_final_input.json"
        echo "$input_json" > "$input_file"
        
        # Invoke finalization flow
        local response=$(aws bedrock-agent-runtime invoke-flow \
            --flow-identifier "$finalization_flow_id" \
            --flow-alias-identifier "$finalization_alias_id" \
            --inputs file://"$input_file" \
            --region "$region" \
            --profile "$profile" 2>/dev/null)
        
        # Extract final polished text
        local final_polished=$(echo "$response" | jq -r '.outputs[] | select(.name == "final_polished_text") | .content')
        local executive_summary=$(echo "$response" | jq -r '.outputs[] | select(.name == "executive_summary") | .content')
        
        # Save finalized chunk
        echo "$final_polished" > "${chunk_file}_polished.txt"
        
        # Save executive summary
        echo "$executive_summary" > "$final_dir/executive_summary_${chunk_id}.txt"
        
        # Add to final outputs
        final_outputs+=("$final_polished")
    done
    
    # Combine finalized chunks into final manuscript
    local bestseller_manuscript="$MANUSCRIPT_DIR/${project_name}_bestseller.txt"
    for output in "${final_outputs[@]}"; do
        echo "$output" >> "$bestseller_manuscript"
        echo -e "\n\n" >> "$bestseller_manuscript"
    done
    
    # Combine executive summaries
    local combined_summary="$MANUSCRIPT_DIR/${project_name}_executive_summary.txt"
    cat "$final_dir"/executive_summary_* > "$combined_summary"
    
    # Update project state
    aws dynamodb update-item \
        --table-name "${table_prefix}_${project_name}_state" \
        --key '{"manuscript_id": {"S": "'"$project_name"'"}, "chunk_id": {"S": "metadata"}}' \
        --update-expression "SET current_phase = :p, updated_at = :u" \
        --expression-attribute-values '{
            ":p": {"S": "complete"},
            ":u": {"S": "'"$(date -u +'%Y-%m-%dT%H:%M:%SZ')"'"}
        }' \
        --region "$region" \
        --profile "$profile" > /dev/null
    
    echo -e "${GREEN}Final best-seller manuscript saved to: $bestseller_manuscript${NC}"
    echo -e "${GREEN}Executive summary saved to: $combined_summary${NC}"
    
    return 0
}

# Process existing manuscript
process_existing_manuscript() {
    echo -e "${BLUE}=== Process Existing Manuscript ===${NC}"

    # List existing deployments
    local flows=$(jq -r '.flows[]' "$FLOWS_LIST_FILE")

    if [ -z "$flows" ]; then
        echo -e "${YELLOW}No existing deployments found.${NC}"
        return
    fi

    echo -e "${BLUE}Existing deployments:${NC}"
    local i=1
    local flow_array=()

    while read -r flow; do
        echo "$i) $flow"
        flow_array+=("$flow")
        ((i++))
    done <<< "$flows"

    # Select a deployment to process
    read -p "Enter the number of the deployment to process (or 0 to cancel): " selection

    if [ -z "$selection" ] || [ "$selection" -eq 0 ] || [ "$selection" -gt "${#flow_array[@]}" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    local selected_flow="${flow_array[$((selection-1))]}"
    local project_config="${selected_flow}_config.json"
    
    # Check if project config exists
    if [ ! -f "$project_config" ]; then
        echo -e "${RED}Error: Project configuration file not found: $project_config${NC}"
        return 1
    fi
    
    # Get project status
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    local table_prefix=$(jq -r '.table_prefix' "$CONFIG_FILE")
    
    local status_response=$(aws dynamodb get-item \
        --table-name "${table_prefix}_${selected_flow}_state" \
        --key '{"manuscript_id": {"S": "'"$selected_flow"'"}, "chunk_id": {"S": "metadata"}}' \
        --region "$region" \
        --profile "$profile" 2>/dev/null)
    
    if [ -z "$status_response" ]; then
        echo -e "${RED}Error: Could not retrieve project status.${NC}"
        return 1
    fi
    
    local current_phase=$(echo "$status_response" | jq -r '.Item.current_phase.S')
    local chunks_processed=$(echo "$status_response" | jq -r '.Item.chunks_processed.N')
    local total_chunks=$(echo "$status_response" | jq -r '.Item.total_chunks.N')
    
    echo -e "${BLUE}Project status: Phase=$current_phase, Progress=$chunks_processed/$total_chunks chunks${NC}"
    
    # Display options based on current phase
    echo -e "${BLUE}Available actions:${NC}"
    
    case "$current_phase" in
        "assessment")
            echo "1) Process manuscript chunks"
            echo "0) Cancel"
            
            read -p "Enter your choice: " action
            
            case "$action" in
                1)
                    process_manuscript_chunks "$selected_flow"
                    ;;
                0)
                    echo -e "${YELLOW}Operation cancelled.${NC}"
                    ;;
                *)
                    echo -e "${RED}Invalid choice.${NC}"
                    ;;
            esac
            ;;
            
        "improvement" | "in_progress")
            echo "1) Continue processing manuscript chunks"
            echo "2) Finalize manuscript"
            echo "0) Cancel"
            
            read -p "Enter your choice: " action
            
            case "$action" in
                1)
                    process_manuscript_chunks "$selected_flow"
                    ;;
                2)
                    finalize_manuscript "$selected_flow"
                    ;;
                0)
                    echo -e "${YELLOW}Operation cancelled.${NC}"
                    ;;
                *)
                    echo -e "${RED}Invalid choice.${NC}"
                    ;;
            esac
            ;;
            
        "finalization")
            echo "1) Finalize manuscript"
            echo "0) Cancel"
            
            read -p "Enter your choice: " action
            
            case "$action" in
                1)
                    finalize_manuscript "$selected_flow"
                    ;;
                0)
                    echo -e "${YELLOW}Operation cancelled.${NC}"
                    ;;
                *)
                    echo -e "${RED}Invalid choice.${NC}"
                    ;;
            esac
            ;;
            
        "complete")
            echo -e "${GREEN}This project is complete. Manuscript has been finalized.${NC}"
            echo "1) View executive summary"
            echo "0) Cancel"
            
            read -p "Enter your choice: " action
            
            case "$action" in
                1)
                    local summary_file="$MANUSCRIPT_DIR/${selected_flow}_executive_summary.txt"
                    if [ -f "$summary_file" ]; then
                        less "$summary_file"
                    else
                        echo -e "${RED}Executive summary not found: $summary_file${NC}"
                    fi
                    ;;
                0)
                    echo -e "${YELLOW}Operation cancelled.${NC}"
                    ;;
                *)
                    echo -e "${RED}Invalid choice.${NC}"
                    ;;
            esac
            ;;
            
        *)
            echo -e "${RED}Unknown project phase: $current_phase${NC}"
            ;;
    esac
}

# Conduct research
conduct_research() {
    echo -e "${BLUE}=== Conduct Research ===${NC}"

    # List existing deployments
    local flows=$(jq -r '.flows[]' "$FLOWS_LIST_FILE")

    if [ -z "$flows" ]; then
        echo -e "${YELLOW}No existing deployments found.${NC}"
        return
    fi

    echo -e "${BLUE}Existing deployments:${NC}"
    local i=1
    local flow_array=()

    while read -r flow; do
        echo "$i) $flow"
        flow_array+=("$flow")
        ((i++))
    done <<< "$flows"

    # Select a deployment to research
    read -p "Enter the number of the deployment for research (or 0 to cancel): " selection

    if [ -z "$selection" ] || [ "$selection" -eq 0 ] || [ "$selection" -gt "${#flow_array[@]}" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    local selected_flow="${flow_array[$((selection-1))]}"
    local project_config="${selected_flow}_config.json"
    
    # Check if project config exists
    if [ ! -f "$project_config" ]; then
        echo -e "${RED}Error: Project configuration file not found: $project_config${NC}"
        return 1
    fi
    
    # Get research topic
    read -p "Enter research topic: " research_topic
    
    if [ -z "$research_topic" ]; then
        echo -e "${RED}Error: Research topic cannot be empty.${NC}"
        return 1
    fi
    
    # Get chunk ID or sample text
    echo -e "${BLUE}Choose research input method:${NC}"
    echo "1) Provide specific chunk ID"
    echo "2) Provide sample text"
    echo "0) Cancel"
    
    read -p "Enter your choice: " input_method
    
    local chunk_id=""
    local chunk_text=""
    
    case "$input_method" in
        1)
            read -p "Enter chunk ID (e.g., chunk_0001): " chunk_id
            local chunk_file="$CHUNKS_DIR/${selected_flow}_improved/${chunk_id}.txt"
            
            if [ ! -f "$chunk_file" ]; then
                echo -e "${RED}Error: Chunk file not found: $chunk_file${NC}"
                echo -e "${YELLOW}Looking for original chunk...${NC}"
                
                # Try to find in original chunks
                local metadata_file="$CHUNKS_DIR/${selected_flow}_metadata.json"
                if [ -f "$metadata_file" ]; then
                    local chunk_num=$(echo "$chunk_id" | sed 's/chunk_0*//')
                    local original_file=$(jq -r ".chunk_files[$((chunk_num-1))]" "$metadata_file")
                    
                    if [ -f "$original_file" ]; then
                        chunk_file="$original_file"
                        echo -e "${GREEN}Found original chunk: $chunk_file${NC}"
                    else
                        echo -e "${RED}Original chunk not found either.${NC}"
                        return 1
                    fi
                else
                    echo -e "${RED}Metadata file not found.${NC}"
                    return 1
                fi
            fi
            
            chunk_text=$(cat "$chunk_file")
            ;;
        2)
            echo -e "${YELLOW}Enter sample text (type 'END' on a new line when finished):${NC}"
            chunk_id="custom_sample"
            
            # Read multi-line input
            while IFS= read -r line; do
                if [ "$line" = "END" ]; then
                    break
                fi
                chunk_text+="$line"$'\n'
            done
            ;;
        0)
            echo -e "${YELLOW}Operation cancelled.${NC}"
            return
            ;;
        *)
            echo -e "${RED}Invalid choice.${NC}"
            return 1
            ;;
    esac
    
    # Generate Python research script if not already created
    if [ ! -f "$TEMP_DIR/web_research.py" ]; then
        generate_research_script
    fi
    
    # Conduct web research
    echo -e "${YELLOW}Conducting web research on: $research_topic${NC}"
    python3 "$TEMP_DIR/web_research.py" "$research_topic" \
        --output_dir "$RESEARCH_DIR" \
        --project_name "$selected_flow" \
        --region "$(jq -r '.aws_region' "$CONFIG_FILE")" \
        --table_prefix "$(jq -r '.table_prefix' "$CONFIG_FILE")"
    
    # Get research flow IDs
    local research_flow_id=$(jq -r '.flows.research.flow_id' "$project_config")
    local research_alias_id=$(jq -r '.flows.research.alias_id' "$project_config")
    
    # Use research flow to analyze and integrate findings
    echo -e "${YELLOW}Using Bedrock flow to analyze research findings...${NC}"
    
    # Create input JSON for flow invocation
    local input_json=$(cat <<EOF
[
  {
    "content": {
      "manuscript_id": "$selected_flow",
      "chunk_id": "$chunk_id",
      "chunk_text": $(jq -Rs '.' <<< "$chunk_text"),
      "research_topic": "$research_topic"
    },
    "nodeName": "FlowInputNode",
    "nodeOutputNames": ["manuscript_id", "chunk_id", "chunk_text", "research_topic"]
  }
]
EOF
)
    
    # Save input to temp file
    local input_file="$TEMP_DIR/${selected_flow}_${chunk_id}_research_input.json"
    echo "$input_json" > "$input_file"
    
    # Invoke research flow
    local response=$(aws bedrock-agent-runtime invoke-flow \
        --flow-identifier "$research_flow_id" \
        --flow-alias-identifier "$research_alias_id" \
        --inputs file://"$input_file" \
        --region "$(jq -r '.aws_region' "$CONFIG_FILE")" \
        --profile "$(jq -r '.aws_profile' "$CONFIG_FILE")" 2>/dev/null)
    
    # Extract research integration plan
    local integration_plan=$(echo "$response" | jq -r '.outputs[] | select(.name == "integration_plan") | .content')
    
    # Save results
    local results_file="$RESEARCH_DIR/${selected_flow}_${research_topic//[^a-zA-Z0-9]/_}_results.txt"
    echo -e "RESEARCH TOPIC: $research_topic\n\n" > "$results_file"
    echo -e "INTEGRATION PLAN:\n$integration_plan\n\n" >> "$results_file"
    
    echo -e "${GREEN}Research completed! Results saved to: $results_file${NC}"
    echo -e "${YELLOW}You can use this research to improve your manuscript.${NC}"
}

# Remove deployment 
remove_deployment() {
    echo -e "${BLUE}=== Remove Deployment ===${NC}"

    # List existing deployments
    local flows=$(jq -r '.flows[]' "$FLOWS_LIST_FILE")

    if [ -z "$flows" ]; then
        echo -e "${YELLOW}No existing deployments found.${NC}"
        return
    }

    echo -e "${BLUE}Existing deployments:${NC}"
    local i=1
    local flow_array=()

    while read -r flow; do
        echo "$i) $flow"
        flow_array+=("$flow")
        ((i++))
    done <<< "$flows"

    # Select a deployment to remove
    read -p "Enter the number of the deployment to remove (or 0 to cancel): " selection

    if [ -z "$selection" ] || [ "$selection" -eq 0 ] || [ "$selection" -gt "${#flow_array[@]}" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    }

    local selected_flow="${flow_array[$((selection-1))]}"

    # Confirm deletion
    read -p "Are you sure you want to delete deployment '$selected_flow'? (y/n): " confirm

    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    }

    # Get project config
    local project_config="${selected_flow}_config.json"
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    
    if [ -f "$project_config" ]; then
        # Delete flows based on project config
        for phase in assessment improvement research finalization; do
            echo -e "${YELLOW}Removing ${phase} flow...${NC}"

            # Get flow ID and alias ID - use grep for safer parsing
            local flow_id=""
            local alias_id=""
            
            # Parse project config safely with grep
            flow_id=$(grep -o "\"${phase}\"[[:space:]]*:[[:space:]]*{[^}]*\"flow_id\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$project_config" | grep -o "\"flow_id\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | cut -d'"' -f4)
            alias_id=$(grep -o "\"${phase}\"[[:space:]]*:[[:space:]]*{[^}]*\"alias_id\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$project_config" | grep -o "\"alias_id\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | cut -d'"' -f4)

            if [ -n "$flow_id" ] && [ -n "$alias_id" ]; then
                # Delete alias
                echo -e "${YELLOW}Deleting alias $alias_id...${NC}"
                aws bedrock-agent delete-flow-alias \
                    --region "$region" \
                    --profile "$profile" \
                    --flow-identifier "$flow_id" \
                    --alias-identifier "$alias_id" 2>/dev/null || true

                # Get flow versions
                echo -e "${YELLOW}Fetching versions for flow $flow_id...${NC}"
                local versions
                versions=$(aws bedrock-agent list-flow-versions \
                    --region "$region" \
                    --profile "$profile" \
                    --flow-identifier "$flow_id" \
                    --query 'versions[].version' \
                    --output text 2>/dev/null) || true

                # Delete each version
                for version in $versions; do
                    echo -e "${YELLOW}Deleting version $version...${NC}"
                    aws bedrock-agent delete-flow-version \
                        --region "$region" \
                        --profile "$profile" \
                        --flow-identifier "$flow_id" \
                        --flow-version "$version" 2>/dev/null || true
                done

                # Delete the flow
                echo -e "${YELLOW}Deleting flow $flow_id...${NC}"
                aws bedrock-agent delete-flow \
                    --region "$region" \
                    --profile "$profile" \
                    --flow-identifier "$flow_id" 2>/dev/null || true
            fi
        done
    else
        echo -e "${YELLOW}Project configuration file not found: $project_config${NC}"
        echo -e "${YELLOW}Attempting to remove based on naming conventions...${NC}"
        
        # Try to find flows by listing and filtering
        local all_flows
        all_flows=$(aws bedrock-agent list-flows \
            --region "$region" \
            --profile "$profile" \
            --query "flowSummaries[?starts_with(name,'${selected_flow}-')].{id:id,name:name}" \
            --output text 2>/dev/null) || true
            
        if [ -n "$all_flows" ]; then
            echo "$all_flows" | while read -r flow_id flow_name; do
                echo -e "${YELLOW}Found flow: $flow_name ($flow_id)${NC}"
                
                # Get aliases
                local aliases
                aliases=$(aws bedrock-agent list-flow-aliases \
                    --region "$region" \
                    --profile "$profile" \
                    --flow-identifier "$flow_id" \
                    --query "flowAliases[].id" \
                    --output text 2>/dev/null) || true
                
                # Delete each alias
                for alias_id in $aliases; do
                    echo -e "${YELLOW}Deleting alias $alias_id...${NC}"
                    aws bedrock-agent delete-flow-alias \
                        --region "$region" \
                        --profile "$profile" \
                        --flow-identifier "$flow_id" \
                        --alias-identifier "$alias_id" 2>/dev/null || true
                done
                
                # Get versions
                local versions
                versions=$(aws bedrock-agent list-flow-versions \
                    --region "$region" \
                    --profile "$profile" \
                    --flow-identifier "$flow_id" \
                    --query "versions[].version" \
                    --output text 2>/dev/null) || true
                
                # Delete each version
                for version in $versions; do
                    echo -e "${YELLOW}Deleting version $version...${NC}"
                    aws bedrock-agent delete-flow-version \
                        --region "$region" \
                        --profile "$profile" \
                        --flow-identifier "$flow_id" \
                        --flow-version "$version" 2>/dev/null || true
                done
                
                # Delete the flow
                echo -e "${YELLOW}Deleting flow $flow_id...${NC}"
                aws bedrock-agent delete-flow \
                    --region "$region" \
                    --profile "$profile" \
                    --flow-identifier "$flow_id" 2>/dev/null || true
            done
        else
            echo -e "${YELLOW}No flows found with name prefix '${selected_flow}-'${NC}"
        fi
    fi

    # Remove DynamoDB tables
    local table_prefix=$(jq -r '.table_prefix' "$CONFIG_FILE")

    echo -e "${YELLOW}Removing DynamoDB tables...${NC}"

    # List and delete tables with project prefix
    local tables
    tables=$(aws dynamodb list-tables \
        --region "$region" \
        --profile "$profile" \
        --query "TableNames[?starts_with(@,'${table_prefix}_${selected_flow}')]" \
        --output text 2>/dev/null) || true

    for table in $tables; do
        echo -e "${YELLOW}Deleting table $table...${NC}"
        aws dynamodb delete-table \
            --region "$region" \
            --profile "$profile" \
            --table-name "$table" 2>/dev/null || true
    done

    # Remove files
    echo -e "${YELLOW}Cleaning up project files...${NC}"
    
    # Remove template files
    rm -f "${FLOW_TEMPLATES_DIR}/${selected_flow}-"*.json
    
    # Remove chunks
    rm -rf "$CHUNKS_DIR/${selected_flow}"*
    rm -f "$CHUNKS_DIR/${selected_flow}"*.*
    
    # Remove manuscripts
    rm -f "$MANUSCRIPT_DIR/${selected_flow}"*.*
    
    # Remove research
    rm -f "$RESEARCH_DIR/${selected_flow}"*.*
    
    # Remove project config file
    rm -f "$project_config"

    # Update flows list
    jq --arg name "$selected_flow" '.flows = [.flows[] | select(. != $name)]' "$FLOWS_LIST_FILE" > "${FLOWS_LIST_FILE}.tmp" && mv "${FLOWS_LIST_FILE}.tmp" "$FLOWS_LIST_FILE"

    echo -e "${GREEN}Deployment '$selected_flow' successfully removed.${NC}"
}

# AWS Configuration
configure_aws() {
    echo -e "${BLUE}=== Configure AWS Settings ===${NC}"

    while true; do
        echo -e "${BLUE}AWS configuration options:${NC}"
        echo "1) Configure AWS credentials"
        echo "2) Change AWS region"
        echo "3) Change AWS profile"
        echo "4) Change IAM role"
        echo "5) Change DynamoDB table prefix"
        echo "0) Return to main menu"

        read -p "Enter your choice: " aws_choice

        case "$aws_choice" in
            1)
                aws configure
                echo -e "${GREEN}AWS credentials updated.${NC}"
                ;;
            2)
                change_default_region
                ;;
            3)
                change_aws_profile
                ;;
            4)
                change_iam_role
                ;;
            5)
                change_table_prefix
                ;;
            0)
                echo -e "${YELLOW}Returning to main menu...${NC}"
                return
                ;;
            *)
                echo -e "${RED}Invalid choice.${NC}"
                ;;
        esac
    done
}

# Modify models and configuration
modify_configuration() {
    echo -e "${BLUE}=== Modify Configuration ===${NC}"

    while true; do
        echo -e "${BLUE}Configuration options:${NC}"
        echo "1) Change default model"
        echo "2) Modify agent-specific models"
        echo "3) Edit quality thresholds"
        echo "4) Edit chunking settings"
        echo "5) Toggle features (web search, etc.)"
        echo "0) Return to main menu"

        read -p "Enter your choice: " config_choice

        case "$config_choice" in
            1)
                change_default_model
                ;;
            2)
                modify_agent_models
                ;;
            3)
                edit_quality_thresholds
                ;;
            4)
                edit_chunking_settings
                ;;
            5)
                toggle_features
                ;;
            0)
                echo -e "${YELLOW}Returning to main menu...${NC}"
                return
                ;;
            *)
                echo -e "${RED}Invalid choice.${NC}"
                ;;
        esac
    done
}

# Change default region
change_default_region() {
    echo -e "${BLUE}=== Change Default AWS Region ===${NC}"

    local current_region=$(jq -r '.aws_region' "$CONFIG_FILE")
    echo -e "${BLUE}Current region: $current_region${NC}"

    # List available regions
    echo -e "${YELLOW}Available regions:${NC}"
    echo "us-east-1) US East (N. Virginia)"
    echo "us-east-2) US East (Ohio)"
    echo "us-west-1) US West (N. California)"
    echo "us-west-2) US West (Oregon)"
    echo "ap-south-1) Asia Pacific (Mumbai)"
    echo "ap-northeast-1) Asia Pacific (Tokyo)"
    echo "ap-northeast-2) Asia Pacific (Seoul)"
    echo "ap-southeast-1) Asia Pacific (Singapore)"
    echo "ap-southeast-2) Asia Pacific (Sydney)"
    echo "ca-central-1) Canada (Central)"
    echo "eu-central-1) EU (Frankfurt)"
    echo "eu-west-1) EU (Ireland)"
    echo "eu-west-2) EU (London)"
    echo "eu-west-3) EU (Paris)"
    echo "eu-north-1) EU (Stockholm)"
    echo "sa-east-1) South America (São Paulo)"

    read -p "Enter new AWS region: " new_region

    if [ -z "$new_region" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    # Update config file
    jq --arg region "$new_region" '.aws_region = $region' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

    echo -e "${GREEN}Default AWS region updated to $new_region.${NC}"
}

# Change default model
change_default_model() {
    echo -e "${BLUE}=== Change Default Model ===${NC}"

    local current_model=$(jq -r '.default_model' "$CONFIG_FILE")
    echo -e "${BLUE}Current default model: $current_model${NC}"

    # Get region for Bedrock
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")

    # List available Bedrock models
    echo -e "${YELLOW}Fetching available Bedrock models...${NC}"
    echo -e "${YELLOW}This may take a moment...${NC}"

    aws bedrock list-foundation-models \
        --region "$region" \
        --profile "$profile" \
        --query "modelSummaries[].modelId" \
        --output table 2>/dev/null || {
            echo -e "${RED}Failed to fetch models. Using built-in list.${NC}"
            echo -e "${YELLOW}Available models:${NC}"
            echo "anthropic.claude-3-opus-20240229-v1:0 (Anthropic Claude 3 Opus)"
            echo "anthropic.claude-3-sonnet-20240229-v1:0 (Anthropic Claude 3 Sonnet)"
            echo "anthropic.claude-3-haiku-20240307-v1:0 (Anthropic Claude 3 Haiku)"
            echo "amazon.titan-text-express-v1 (Amazon Titan Text Express)"
            echo "meta.llama3-70b-instruct-v1:0 (Meta Llama 3 70B Instruct)"
            echo "meta.llama3-8b-instruct-v1:0 (Meta Llama 3 8B Instruct)"
        }

    read -p "Enter new default model ID: " new_model

    if [ -z "$new_model" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    # Update config file
    jq --arg model "$new_model" '.default_model = $model' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

    # Ask if this should apply to all agents
    read -p "Apply this model to all agents? (y/n): " apply_all

    if [ "$apply_all" == "y" ] || [ "$apply_all" == "Y" ]; then
        local agents=$(jq -r '.models | keys[]' "$CONFIG_FILE")

        for agent in $agents; do
            jq --arg agent "$agent" --arg model "$new_model" '.models[$agent] = $model' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        done

        echo -e "${GREEN}Applied $new_model to all agents.${NC}"
    fi

    echo -e "${GREEN}Default model updated to $new_model.${NC}"
    echo -e "${YELLOW}You will need to regenerate flows for changes to take effect.${NC}"
}

# Modify agent models
modify_agent_models() {
    echo -e "${BLUE}=== Modify Agent Models ===${NC}"

    # List agents and their models
    echo -e "${BLUE}Current agent models:${NC}"
    jq -r '.models | to_entries[] | "\(.key): \(.value)"' "$CONFIG_FILE" | sort | nl

    read -p "Enter agent number to modify (or 0 for all agents): " agent_num

    if [ -z "$agent_num" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    if [ "$agent_num" -eq 0 ]; then
        # Update all agents
        change_default_model
        return
    fi

    # Get agent name from number
    local agent_name=$(jq -r '.models | keys[]' "$CONFIG_FILE" | sort | sed -n "${agent_num}p")

    if [ -z "$agent_name" ]; then
        echo -e "${RED}Invalid agent number.${NC}"
        return
    fi

    # Get current model
    local current_model=$(jq -r --arg agent "$agent_name" '.models[$agent]' "$CONFIG_FILE")
    echo -e "${BLUE}Current model for $agent_name: $current_model${NC}"

    # Get region for Bedrock
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")

    # List available Bedrock models (abbreviated output)
    echo -e "${YELLOW}Available models (sample):${NC}"
    echo "anthropic.claude-3-opus-20240229-v1:0 (Anthropic Claude 3 Opus)"
    echo "anthropic.claude-3-sonnet-20240229-v1:0 (Anthropic Claude 3 Sonnet)"
    echo "anthropic.claude-3-haiku-20240307-v1:0 (Anthropic Claude 3 Haiku)"
    echo "amazon.titan-text-express-v1 (Amazon Titan Text Express)"
    echo "meta.llama3-70b-instruct-v1:0 (Meta Llama 3 70B Instruct)"
    echo "meta.llama3-8b-instruct-v1:0 (Meta Llama 3 8B Instruct)"

    read -p "Enter new model ID for $agent_name: " new_model

    if [ -z "$new_model" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    # Update specific agent
    jq --arg agent "$agent_name" --arg model "$new_model" '.models[$agent] = $model' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

    echo -e "${GREEN}Updated $agent_name to use $new_model.${NC}"
    echo -e "${YELLOW}You will need to regenerate flows for changes to take effect.${NC}"
}

# Edit quality thresholds
edit_quality_thresholds() {
    echo -e "${BLUE}=== Edit Quality Thresholds ===${NC}"

    # List current thresholds
    echo -e "${BLUE}Current quality thresholds:${NC}"
    jq -r '.quality_thresholds | to_entries[] | "\(.key): \(.value)"' "$CONFIG_FILE" | sort | nl

    # Dump to temporary file
    jq '.quality_thresholds' "$CONFIG_FILE" > "$TEMP_DIR/quality_thresholds.json"

    # Edit in editor
    echo -e "${YELLOW}Opening editor to update quality thresholds...${NC}"
    ${EDITOR:-nano} "$TEMP_DIR/quality_thresholds.json"

    # Read updated thresholds
    local updated_thresholds=$(cat "$TEMP_DIR/quality_thresholds.json")

    # Validate JSON
    if ! jq . "$TEMP_DIR/quality_thresholds.json" > /dev/null 2>&1; then
        echo -e "${RED}Invalid JSON. Changes not saved.${NC}"
        return
    fi

    # Update config file
    jq --argjson thresholds "$updated_thresholds" '.quality_thresholds = $thresholds' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

    echo -e "${GREEN}Quality thresholds updated.${NC}"
}

# Edit chunking settings
edit_chunking_settings() {
    echo -e "${BLUE}=== Edit Chunking Settings ===${NC}"

    local current_size=$(jq -r '.chunk_size' "$CONFIG_FILE")
    local current_overlap=$(jq -r '.chunk_overlap' "$CONFIG_FILE")

    echo -e "${BLUE}Current chunking settings:${NC}"
    echo "1) Chunk size: $current_size tokens"
    echo "2) Chunk overlap: $current_overlap tokens"
    echo "0) Cancel"

    read -p "Enter setting to change: " setting_num

    case "$setting_num" in
        1)
            read -p "Enter new chunk size (tokens): " new_size
            if [[ "$new_size" =~ ^[0-9]+$ ]]; then
                jq --argjson size "$new_size" '.chunk_size = $size' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
                echo -e "${GREEN}Chunk size updated to $new_size tokens.${NC}"
            else
                echo -e "${RED}Invalid value. Must be a number.${NC}"
            fi
            ;;
        2)
            read -p "Enter new chunk overlap (tokens): " new_overlap
            if [[ "$new_overlap" =~ ^[0-9]+$ ]]; then
                jq --argjson overlap "$new_overlap" '.chunk_overlap = $overlap' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
                echo -e "${GREEN}Chunk overlap updated to $new_overlap tokens.${NC}"
            else
                echo -e "${RED}Invalid value. Must be a number.${NC}"
            fi
            ;;
        0)
            echo -e "${YELLOW}Operation cancelled.${NC}"
            ;;
        *)
            echo -e "${RED}Invalid choice.${NC}"
            ;;
    esac
}

# Toggle features
toggle_features() {
    echo -e "${BLUE}=== Toggle Features ===${NC}"

    # List current settings
    echo -e "${BLUE}Current feature settings:${NC}"
    jq -r '.processing_settings | to_entries[] | "\(.key): \(.value)"' "$CONFIG_FILE" | nl

    read -p "Enter feature number to toggle (or 0 to cancel): " feature_num

    if [ -z "$feature_num" ] || [ "$feature_num" -eq 0 ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    # Get feature name from number
    local feature_name=$(jq -r '.processing_settings | keys[]' "$CONFIG_FILE" | sed -n "${feature_num}p")

    if [ -z "$feature_name" ]; then
        echo -e "${RED}Invalid feature number.${NC}"
        return
    fi

    # Get current value
    local current_value=$(jq -r --arg feature "$feature_name" '.processing_settings[$feature]' "$CONFIG_FILE")
    
    # Toggle value
    local new_value=true
    if [ "$current_value" = "true" ]; then
        new_value=false
    fi

    # Update config
    jq --arg feature "$feature_name" --argjson value "$new_value" '.processing_settings[$feature] = $value' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

    echo -e "${GREEN}Feature '$feature_name' toggled to $new_value.${NC}"
}

# Change AWS profile
change_aws_profile() {
    echo -e "${BLUE}=== Change AWS Profile ===${NC}"

    local current_profile=$(jq -r '.aws_profile' "$CONFIG_FILE")
    echo -e "${BLUE}Current AWS profile: $current_profile${NC}"

    # List available profiles
    echo -e "${YELLOW}Available profiles:${NC}"
    aws configure list-profiles 2>/dev/null | nl || echo "No profiles found"

    read -p "Enter new AWS profile: " new_profile

    if [ -z "$new_profile" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    # Update config file
    jq --arg profile "$new_profile" '.aws_profile = $profile' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

    echo -e "${GREEN}AWS profile updated to $new_profile.${NC}"
}

# Change IAM role
change_iam_role() {
    echo -e "${BLUE}=== Change IAM Role ===${NC}"

    local current_role=$(jq -r '.role_name' "$CONFIG_FILE")
    echo -e "${BLUE}Current IAM role: $current_role${NC}"

    read -p "Enter new IAM role name: " new_role

    if [ -z "$new_role" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")

    # Check if role exists
    if ! aws iam get-role --role-name "$new_role" --region "$region" --profile "$profile" &> /dev/null; then
        echo -e "${YELLOW}Role '$new_role' does not exist.${NC}"
        read -p "Do you want to create it? (y/n): " create_role

        if [ "$create_role" == "y" ] || [ "$create_role" == "Y" ]; then
            # Update config first
            jq --arg role "$new_role" '.role_name = $role' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

            # Create the role
            initialize_iam_role
        else
            echo -e "${YELLOW}Operation cancelled.${NC}"
            return
        fi
    else
        # Update config
        jq --arg role "$new_role" '.role_name = $role' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        echo -e "${GREEN}IAM role updated to $new_role.${NC}"
    fi
}

# Change DynamoDB table prefix
change_table_prefix() {
    echo -e "${BLUE}=== Change DynamoDB Table Prefix ===${NC}"

    local current_prefix=$(jq -r '.table_prefix' "$CONFIG_FILE")
    echo -e "${BLUE}Current table prefix: $current_prefix${NC}"

    read -p "Enter new table prefix: " new_prefix

    if [ -z "$new_prefix" ]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return
    fi

    # Update config file
    jq --arg prefix "$new_prefix" '.table_prefix = $prefix' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

    echo -e "${GREEN}DynamoDB table prefix updated to $new_prefix.${NC}"
    echo -e "${YELLOW}This change will only affect new deployments.${NC}"
}

# Check if AWS CLI is properly configured
check_aws_configuration() {
    local region=$(jq -r '.aws_region' "$CONFIG_FILE")
    local profile=$(jq -r '.aws_profile' "$CONFIG_FILE")

    echo -e "${YELLOW}Checking AWS configuration...${NC}"

    # Try to use the AWS CLI
    if ! aws sts get-caller-identity --region "$region" --profile "$profile" &> /dev/null; then
        echo -e "${RED}Error: AWS CLI is not properly configured.${NC}"
        echo -e "${RED}Please configure AWS CLI with 'aws configure' command.${NC}"
        echo -e "${RED}Alternatively, select option 4) AWS Configuration from the main menu.${NC}"
        return 1
    fi

    echo -e "${GREEN}AWS configuration is valid.${NC}"
    return 0
}

# Main function
main() {
    clear
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}   NovelFlow - Bedrock Novel Editor     ${NC}"
    echo -e "${BLUE}=========================================${NC}"

    # Check requirements
    check_requirements

    # Initialize configuration and directories
    initialize_config

    while true; do
        echo -e "\n${BLUE}Main Menu:${NC}"
        echo "1) Create New Manuscript Project"
        echo "2) Process Existing Manuscript"
        echo "3) Conduct Research"
        echo "4) Remove Manuscript Project"
        echo "5) Modify Configuration (Models, Thresholds)"
        echo "6) AWS Configuration"
        echo "q) Quit"

        read -p "Enter your choice: " choice

        case "$choice" in
            1)
                create_new_deployment
                ;;
            2)
                process_existing_manuscript
                ;;
            3)
                conduct_research
                ;;
            4)
                remove_deployment
                ;;
            5)
                modify_configuration
                ;;
            6)
                configure_aws
                ;;
            q|Q)
                echo -e "${GREEN}Exiting. Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                ;;
        esac
    done
}

# Run main function 
main
