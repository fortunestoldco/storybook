import json
import os
import boto3
import time
from urllib.parse import urlparse

# Initialize clients
bedrock_client = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['STATE_TABLE'])

def lambda_handler(event, context):
    """
    Research function that queries knowledge base for relevant information
    """
    print("Received event:", json.dumps(event))
    
    try:
        # Extract parameters
        manuscript_id = event.get('manuscript_id')
        query = event.get('query')
        kb_id = event.get('knowledge_base_id')
        
        if not manuscript_id or not query or not kb_id:
            return {
                'statusCode': 400,
                'body': 'Missing required parameters: manuscript_id, query, or knowledge_base_id'
            }
            
        # Query the knowledge base
        response = bedrock_client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 5
                }
            }
        )
        
        # Process and format the results
        research_results = []
        for result in response.get('retrievalResults', []):
            content = result.get('content', {}).get('text', '')
            metadata = {}
            for meta in result.get('metadata', {}).get('textMetadata', {}).get('customAttributes', []):
                metadata[meta['key']] = meta['value']
                
            research_results.append({
                'content': content,
                'score': result.get('score', 0),
                'metadata': metadata
            })
        
        # Save the results to DynamoDB
        update_response = table.update_item(
            Key={'ManuscriptId': manuscript_id},
            UpdateExpression="SET ResearchResults = list_append(if_not_exists(ResearchResults, :empty_list), :results)",
            ExpressionAttributeValues={
                ':results': [{'query': query, 'results': research_results, 'timestamp': int(time.time())}],
                ':empty_list': []
            },
            ReturnValues="UPDATED_NEW"
        )
        
        return {
            'statusCode': 200,
            'body': 'Research completed successfully',
            'results': research_results,
            'manuscript_id': manuscript_id
        }
        
    except Exception as e:
        print(f"Error processing research request: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error processing research request: {str(e)}'
        }
