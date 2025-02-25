"""Storage service for file operations in the Storybook application."""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Union, BinaryIO
import boto3
from botocore.exceptions import ClientError

from storybook.config import (
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, 
    S3_BUCKET_NAME
)

class StorageService:
    """Service for cloud storage operations."""
    
    def __init__(self, bucket_name: str = None):
        """Initialize the storage service."""
        self.bucket_name = bucket_name or S3_BUCKET_NAME
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    
    def upload_file(
        self, 
        file_path: str, 
        object_name: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload a file to S3 storage."""
        # If object_name not specified, use file_path
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        try:
            self.s3_client.upload_file(
                file_path, 
                self.bucket_name, 
                object_name,
                ExtraArgs=extra_args
            )
            
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{object_name}"
            
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "bucket": self.bucket_name,
                "object_name": object_name,
                "url": url
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"Error uploading file: {str(e)}"
            }
    
    def upload_fileobj(
        self, 
        file_obj: BinaryIO, 
        object_name: str,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload a file object to S3 storage."""
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        try:
            self.s3_client.upload_fileobj(
                file_obj, 
                self.bucket_name, 
                object_name,
                ExtraArgs=extra_args
            )
            
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{object_name}"
            
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "bucket": self.bucket_name,
                "object_name": object_name,
                "url": url
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"Error uploading file: {str(e)}"
            }
    
    def download_file(self, object_name: str, file_path: str) -> Dict[str, Any]:
        """Download a file from S3 storage."""
        try:
            self.s3_client.download_file(
                self.bucket_name, 
                object_name, 
                file_path
            )
            
            return {
                "status": "success",
                "message": "File downloaded successfully",
                "bucket": self.bucket_name,
                "object_name": object_name,
                "file_path": file_path
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"Error downloading file: {str(e)}"
            }
    
    def get_object(self, object_name: str) -> Dict[str, Any]:
        """Get an object from S3 storage."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            
            # Read the content
            content = response['Body'].read()
            
            return {
                "status": "success",
                "message": "Object retrieved successfully",
                "bucket": self.bucket_name,
                "object_name": object_name,
                "content": content,
                "metadata": response.get('Metadata', {}),
                "content_type": response.get('ContentType', '')
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"Error retrieving object: {str(e)}"
            }
    
    def list_objects(self, prefix: str = "") -> Dict[str, Any]:
        """List objects in the S3 bucket with optional prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        "key": obj['Key'],
                        "last_modified": obj['LastModified'].isoformat(),
                        "size": obj['Size'],
                        "url": f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                    })
            
            return {
                "status": "success",
                "message": f"Listed {len(objects)} objects",
                "bucket": self.bucket_name,
                "prefix": prefix,
                "objects": objects
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"Error listing objects: {str(e)}"
            }
    
    def delete_object(self, object_name: str) -> Dict[str, Any]:
        """Delete an object from S3 storage."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            
            return {
                "status": "success",
                "message": "Object deleted successfully",
                "bucket": self.bucket_name,
                "object_name": object_name
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"Error deleting object: {str(e)}"
            }
    
    def generate_presigned_url(
        self, 
        object_name: str, 
        expiration: int = 3600,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a presigned URL for S3 operations."""
        try:
            params = {
                'Bucket': self.bucket_name,
                'Key': object_name
            }
            
            if content_type:
                params['ContentType'] = content_type
            
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params=params,
                ExpiresIn=expiration
            )
            
            return {
                "status": "success",
                "message": "Presigned URL generated successfully",
                "bucket": self.bucket_name,
                "object_name": object_name,
                "url": url,
                "expires_in": expiration
            }
        except ClientError as e:
            return {
                "status": "error",
                "message": f"Error generating presigned URL: {str(e)}"
            }
