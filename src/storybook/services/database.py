"""Database connection and operations for the Storybook application."""

import os
from typing import Dict, List, Any, Optional, Union
from pymongo import MongoClient
from bson.objectid import ObjectId

from storybook.config import MONGODB_CONNECTION_STRING, DB_NAME

class DatabaseConnector:
    """Connector for MongoDB database operations."""
    
    def __init__(self, connection_string: str = None, db_name: str = None):
        """Initialize the database connector."""
        self.connection_string = connection_string or MONGODB_CONNECTION_STRING
        self.db_name = db_name or DB_NAME
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Connect to the MongoDB database."""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            raise
    
    def insert_one(self, collection: str, document: Dict[str, Any]) -> Any:
        """Insert a single document into a collection."""
        if not self.db:
            self._connect()
        
        # Convert string IDs to ObjectId where appropriate
        if "_id" in document and isinstance(document["_id"], str):
            try:
                document["_id"] = ObjectId(document["_id"])
            except:
                pass  # Keep as string if not a valid ObjectId
        
        return self.db[collection].insert_one(document)
    
    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> Any:
        """Insert multiple documents into a collection."""
        if not self.db:
            self._connect()
        return self.db[collection].insert_many(documents)
    
    def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a collection."""
        if not self.db:
            self._connect()
        
        # Convert string ID to ObjectId if needed
        if "_id" in query and isinstance(query["_id"], str):
            try:
                query["_id"] = ObjectId(query["_id"])
            except:
                pass  # Keep as string if not a valid ObjectId
        
        result = self.db[collection].find_one(query)
        
        # Convert ObjectId to string for serialization
        if result and "_id" in result and isinstance(result["_id"], ObjectId):
            result["_id"] = str(result["_id"])
        
        return result
    
    def find(self, collection: str, query: Dict[str, Any], limit: int = 0, sort: List[tuple] = None) -> List[Dict[str, Any]]:
        """Find documents in a collection."""
        if not self.db:
            self._connect()
        
        # Convert string ID to ObjectId if needed
        if "_id" in query and isinstance(query["_id"], str):
            try:
                query["_id"] = ObjectId(query["_id"])
            except:
                pass
        
        cursor = self.db[collection].find(query)
        
        if limit > 0:
            cursor = cursor.limit(limit)
        
        if sort:
            cursor = cursor.sort(sort)
        
        # Convert ObjectId to string for serialization
        results = []
        for doc in cursor:
            if "_id" in doc and isinstance(doc["_id"], ObjectId):
                doc["_id"] = str(doc["_id"])
            results.append(doc)
        
        return results
    
    def update_one(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> Any:
        """Update a single document in a collection."""
        if not self.db:
            self._connect()
        
        # Convert string ID to ObjectId if needed
        if "_id" in query and isinstance(query["_id"], str):
            try:
                query["_id"] = ObjectId(query["_id"])
            except:
                pass
        
        return self.db[collection].update_one(query, update)
    
    def update_many(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> Any:
        """Update multiple documents in a collection."""
        if not self.db:
            self._connect()
        
        # Convert string ID to ObjectId if needed
        if "_id" in query and isinstance(query["_id"], str):
            try:
                query["_id"] = ObjectId(query["_id"])
            except:
                pass
        
        return self.db[collection].update_many(query, update)
    
    def delete_one(self, collection: str, query: Dict[str, Any]) -> Any:
        """Delete a single document from a collection."""
        if not self.db:
            self._connect()
        
        # Convert string ID to ObjectId if needed
        if "_id" in query and isinstance(query["_id"], str):
            try:
                query["_id"] = ObjectId(query["_id"])
            except:
                pass
        
        return self.db[collection].delete_one(query)
    
    def delete_many(self, collection: str, query: Dict[str, Any]) -> Any:
        """Delete multiple documents from a collection."""
        if not self.db:
            self._connect()
        
        # Convert string ID to ObjectId if needed
        if "_id" in query and isinstance(query["_id"], str):
            try:
                query["_id"] = ObjectId(query["_id"])
            except:
                pass
        
        return self.db[collection].delete_many(query)
    
    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
