from typing import Dict, List, Optional, Any
import logging
import os
from bson.objectid import ObjectId

from langchain_core.documents import Document
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection

from storybook.config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)

class MongoDBStore:
    """Interface to MongoDB for document storage and retrieval."""
    
    def __init__(self):
        """Initialize the MongoDB connection."""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB_NAME]
        
        # Initialize the collections if they don't exist
        self._ensure_collections_exist()
        
    def _ensure_collections_exist(self):
        """Ensure required collections exist and have appropriate indexes."""
        # List of collections we need
        collections = [
            "manuscripts", 
            "characters", 
            "worlds", 
            "subplots",
            "research",
            "analysis"
        ]
        
        # Create collections if they don't exist
        existing_collections = self.db.list_collection_names()
        for collection_name in collections:
            if collection_name not in existing_collections:
                self.db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
        
        # Add indexes where needed
        self.db["manuscripts"].create_index("title")
        self.db["characters"].create_index([("manuscript_id", 1), ("character_name", 1)])
        self.db["worlds"].create_index([("manuscript_id", 1), ("name", 1)])
        self.db["subplots"].create_index("manuscript_id")
        self.db["research"].create_index("manuscript_id")
        self.db["analysis"].create_index([("manuscript_id", 1), ("analysis_type", 1)])
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a MongoDB collection by name."""
        return self.db[collection_name]
    
    def store_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Store a document in the specified collection."""
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def get_document(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID from the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one({"_id": ObjectId(document_id)})
            
            # Convert ObjectId to string
            if document and "_id" in document:
                document["_id"] = str(document["_id"])
                
            return document
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None
    
    def update_document(self, collection_name: str, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document in the specified collection."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
    
    def query_documents(self, collection_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query documents based on criteria."""
        collection = self.get_collection(collection_name)
        documents = list(collection.find(query))
        
        # Convert ObjectId to string
        for document in documents:
            if "_id" in document:
                document["_id"] = str(document["_id"])
        
        return documents
    
    def store_documents_with_embeddings(self, collection_name: str, documents: List[Document]) -> List[str]:
        """Store documents with their embeddings in the specified collection."""
        # For simplicity in this initial version, we'll just store the documents
        # without embeddings. A real implementation would generate embeddings here.
        collection = self.get_collection(collection_name)
        
        document_ids = []
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            document = {
                "content": content,
                "metadata": metadata,
                # We'd add embedding vector here
            }
            
            result = collection.insert_one(document)
            document_ids.append(str(result.inserted_id))
        
        return document_ids
    
    def similarity_search(self, collection_name: str, query: str, k: int = 5) -> List[Document]:
        """Search for documents similar to the query."""
        # For simplicity, just return some documents from the collection
        # A real implementation would use vector similarity search
        collection = self.get_collection(collection_name)
        
        # Simple text search (crude approximation of similarity search)
        documents = collection.find(
            {"$text": {"$search": query}}
        ).limit(k)
        
        # If no text index, fall back to a simple scan (inefficient)
        if not list(documents):
            documents = collection.find().limit(k)
        
        results = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if "_id" in doc:
                metadata["id"] = str(doc["_id"])
            
            results.append(Document(page_content=content, metadata=metadata))
        
        return results
