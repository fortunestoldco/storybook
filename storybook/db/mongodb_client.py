from typing import Dict, List, Optional, Any
import logging
import os
from bson.objectid import ObjectId

from langchain_core.documents import Document
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.vectorstores import MongoDBVectorStore
from langchain_openai import OpenAIEmbeddings

from storybook.config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)


class MongoDBStore:
    """Interface to MongoDB for document storage and retrieval."""

    def __init__(self):
        """Initialize the MongoDB connection."""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB_NAME]
        
        # Initialize vector store for semantic search capabilities
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = MongoDBVectorStore(
            collection=self.db["vectors"],
            index_name="vector_index",
            embedding=self.embeddings,
            text_key="content",
        )

        # Initialize the collections if they don't exist
        self._ensure_collections_exist()
        
    def _ensure_collections_exist(self):
        """Create collections if they don't exist."""
        collections = [
            "manuscripts", 
            "characters", 
            "worlds", 
            "subplots", 
            "research", 
            "analysis",
            "vectors"
        ]
        
        for collection in collections:
            if collection not in self.db.list_collection_names():
                self.db.create_collection(collection)
                logger.info(f"Created collection: {collection}")
                
    def get_collection(self, collection_name: str) -> Collection:
        """Get a collection by name."""
        return self.db[collection_name]
    
    def store_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Store a document in the specified collection."""
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def get_document(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        collection = self.get_collection(collection_name)
        try:
            document = collection.find_one({"_id": ObjectId(document_id)})
            if document:
                document["_id"] = str(document["_id"])  # Convert ObjectId to string
            return document
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None
    
    def update_document(self, collection_name: str, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document by ID."""
        collection = self.get_collection(collection_name)
        try:
            result = collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
    
    def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document by ID."""
        collection = self.get_collection(collection_name)
        try:
            result = collection.delete_one({"_id": ObjectId(document_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def query_documents(self, collection_name: str, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Query documents based on criteria."""
        collection = self.get_collection(collection_name)
        try:
            documents = collection.find(query).limit(limit)
            result = []
            for doc in documents:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                result.append(doc)
            return result
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return []
    
    def store_documents_with_embeddings(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """Store documents with their embeddings in the specified collection."""
        # Store in vector store for semantic search
        if documents:
            try:
                # Add documents to vector store
                ids = self.vector_store.add_documents(documents)
                
                # Also store in regular collection for non-vector queries
                collection = self.get_collection(collection_name)
                regular_ids = []
                
                for doc in documents:
                    content = doc.page_content
                    metadata = doc.metadata.copy()
                    metadata["vector_id"] = ids[len(regular_ids)]  # Link to vector ID
                    
                    document = {
                        "content": content,
                        "metadata": metadata,
                    }
                    
                    result = collection.insert_one(document)
                    regular_ids.append(str(result.inserted_id))
                
                return ids
            except Exception as e:
                logger.error(f"Error storing documents with embeddings: {e}")
                return []
        return []

    def similarity_search(
        self, collection_name: str, query: str, k: int = 5
    ) -> List[Document]:
        """Search for documents similar to the query."""
        try:
            # Use vector search
            results = self.vector_store.similarity_search(query, k=k)
            
            # If no results or error, fall back to basic search
            if not results:
                collection = self.get_collection(collection_name)
                
                # First try text search if index exists
                try:
                    documents = list(collection.find({"$text": {"$search": query}}).limit(k))
                except:
                    # No text index, do a basic scan with regex
                    documents = list(collection.find({
                        "$or": [
                            {"content": {"$regex": re.escape(query), "$options": "i"}},
                            {"metadata.title": {"$regex": re.escape(query), "$options": "i"}}
                        ]
                    }).limit(k))
                
                results = []
                for doc in documents:
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    if "_id" in doc:
                        metadata["id"] = str(doc["_id"])
                    
                    results.append(Document(page_content=content, metadata=metadata))
            
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            # Fallback to empty results
            return []
