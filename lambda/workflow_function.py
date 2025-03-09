import json
import os
import boto3
import time
import uuid
from botocore.exceptions import ClientError

# Initialize clients
bedrock_client = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')
table = dynamodb.Table(os.environ['STATE_TABLE'])

# Get environment variables
EXECUTIVE_DIRECTOR_ID = os.environ.get('EXECUTIVE_DIRECTOR_ID')
EXECUTIVE_DIRECTOR_ALIAS_ID = os.environ.get('EXECUTIVE_DIRECTOR_ALIAS_ID')
STATE_TABLE = os.environ['STATE_TABLE']

def lambda_handler(event, context):
    """
    Main workflow handler for the manuscript editing system
    """
    print("Received event:", json.dumps(event))
    
    try:
        # Extract action from the event
        action = event.get('action')
        
        if action == 'edit_manuscript':
            return process_manuscript(event)
        elif action == 'get_status':
            return get_manuscript_status(event)
        else:
            return {
                'statusCode': 400,
                'body': f'Invalid action: {action}'
            }
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error processing request: {str(e)}'
        }

def process_manuscript(event):
    """
    Process a new manuscript for editing
    """
    # Extract manuscript details
    manuscript_id = event.get('manuscript_id')
    if not manuscript_id:
        manuscript_id = f"manuscript-{uuid.uuid4()}"
    
    title = event.get('title', 'Untitled Manuscript')
    author = event.get('author', 'Unknown Author')
    genre = event.get('genre', 'General')
    content = event.get('content', '')
    
    if not content:
        return {
            'statusCode': 400,
            'body': 'Manuscript content cannot be empty'
        }
    
    # Store the manuscript in DynamoDB
    timestamp = int(time.time())
    item = {
        'ManuscriptId': manuscript_id,
        'Title': title,
        'Author': author,
        'Genre': genre,
        'Content': content,
        'Status': 'SUBMITTED',
        'CreatedAt': timestamp,
        'UpdatedAt': timestamp
    }
    
    table.put_item(Item=item)
    
    # Start the editing process by invoking the Executive Director agent
    if EXECUTIVE_DIRECTOR_ID and EXECUTIVE_DIRECTOR_ALIAS_ID:
        try:
            # Prepare the prompt for the agent
            prompt = f"""
            I need you to edit a manuscript with the following details:
            Title: {title}
            Author: {author}
            Genre: {genre}
            
            Please review the manuscript and collaborate with your specialized editor agents to provide comprehensive feedback.
            
            Here's the manuscript content:
            {content}
            """
            
            # Invoke the agent asynchronously
            response = bedrock_client.invoke_agent(
                agentId=EXECUTIVE_DIRECTOR_ID,
                agentAliasId=EXECUTIVE_DIRECTOR_ALIAS_ID,
                sessionId=manuscript_id,
                inputText=prompt
            )
            
            # Update status to PROCESSING
            table.update_item(
                Key={'ManuscriptId': manuscript_id},
                UpdateExpression="SET Status = :status, UpdatedAt = :time",
                ExpressionAttributeValues={
                    ':status': 'PROCESSING',
                    ':time': timestamp
                }
            )
            
            return {
                'statusCode': 200,
                'manuscript_id': manuscript_id,
                'status': 'PROCESSING',
                'message': 'Manuscript submitted successfully and editing has started'
            }
        
        except ClientError as e:
            print(f"Error invoking agent: {str(e)}")
            
            # Update status to ERROR
            table.update_item(
                Key={'ManuscriptId': manuscript_id},
                UpdateExpression="SET Status = :status, ErrorMessage = :error, UpdatedAt = :time",
                ExpressionAttributeValues={
                    ':status': 'ERROR',
                    ':error': str(e),
                    ':time': timestamp
                }
            )
            
            return {
                'statusCode': 500,
                'manuscript_id': manuscript_id,
                'status': 'ERROR',
                'message': f'Error invoking editing agent: {str(e)}'
            }
    else:
        return {
            'statusCode': 500,
            'manuscript_id': manuscript_id,
            'status': 'ERROR',
            'message': 'Executive Director agent configuration is missing'
        }

def get_manuscript_status(event):
    """
    Get the status of a manuscript in the editing process
    """
    manuscript_id = event.get('manuscript_id')
    
    if not manuscript_id:
        return {
            'statusCode': 400,
            'body': 'Missing manuscript_id parameter'
        }
        
    try:
        # Get the manuscript from DynamoDB
        response = table.get_item(
            Key={'ManuscriptId': manuscript_id}
        )
        
        if 'Item' not in response:
            return {
                'statusCode': 404,
                'message': f'Manuscript with ID {manuscript_id} not found'
            }
            
        manuscript = response['Item']
        
        return {
            'statusCode': 200,
            'manuscript_id': manuscript_id,
            'title': manuscript.get('Title'),
            'author': manuscript.get('Author'),
            'status': manuscript.get('Status'),
            'created_at': manuscript.get('CreatedAt'),
            'updated_at': manuscript.get('UpdatedAt'),
            'feedback': manuscript.get('Feedback', {}),
            'error_message': manuscript.get('ErrorMessage', '')
        }
        
    except Exception as e:
        print(f"Error retrieving manuscript status: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error retrieving manuscript status: {str(e)}'
        }
