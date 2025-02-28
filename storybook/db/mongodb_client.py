from __future__ import annotations

# Standard library imports
from typing import Any, Dict, List, Optional
import logging
import os
import threading

# Third-party imports
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid, OperationFailure, ServerSelectionTimeoutError
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Local imports
from storybook.config import (
    MONGODB_URI,
    MONGODB_DB_NAME,
    OPENAI_API_KEY,
    COLLECTION_MANUSCRIPTS,
    COLLECTION_CHARACTERS,
    COLLECTION_WORLDS,
    COLLECTION_SUBPLOTS,
    COLLECTION_RESEARCH,
    COLLECTION_ANALYSIS
)

logger = logging.getLogger(__name__)

# Singleton instance and lock
_client_instance = None
_db_instance = None
_embeddings_instance = None
_client_lock = threading.Lock()


def get_mongodb_client() -> MongoClient:
    """Get or create a MongoDB client singleton."""
    global _client_instance
    
    with _client_lock:
        if _client_instance is None:
            # Get MongoDB URI from environment, fallback to default
            mongodb_uri = os.getenv('MONGODB_URI', MONGODB_URI)
            logger.info(f"Connecting to MongoDB at: {mongodb_uri}")
            
            try:
                _client_instance = MongoClient(
                    mongodb_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    maxPoolSize=50,  # Optimize max pool size
                    socketTimeoutMS=5000  # Optimize socket timeout
                )
                # Test connection
                _client_instance.server_info()
                logger.info("Successfully connected to MongoDB")
            except ServerSelectionTimeoutError as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                logger.error(f"Attempted connection to: {mongodb_uri}")
                raise
                
    return _client_instance


def get_mongodb_database():
    """Get the MongoDB database singleton."""
    global _db_instance
    
    with _client_lock:
        if _db_instance is None:
            client = get_mongodb_client()
            _db_instance = client[MONGODB_DB_NAME]
            
    return _db_instance


def get_embeddings():
    """Get the embeddings singleton."""
    global _embeddings_instance
    
    with _client_lock:
        if _embeddings_instance is None:
            _embeddings_instance = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY
            )
            
    return _embeddings_instance


class MongoDBStore:
    """MongoDB storage implementation using singletons for connection management."""

    def __init__(self):
        """Initialize MongoDB connection and embeddings using singletons."""
        self.client = get_mongodb_client()
        self.db = get_mongodb_database()
        self.embeddings = get_embeddings()
        
        # Initialize collections
        self._ensure_collections_exist()

    def _ensure_collections_exist(self):
        """Ensure required collections exist with proper indexes."""
        collections = {
            COLLECTION_MANUSCRIPTS: [
                ("content", pymongo.TEXT),
                ("embedding", pymongo.ASCENDING)
            ],
            COLLECTION_CHARACTERS: [
                ("name", pymongo.TEXT),
                ("embedding", pymongo.ASCENDING)
            ],
            COLLECTION_WORLDS: [
                ("name", pymongo.TEXT),
                ("embedding", pymongo.ASCENDING)
            ],
            COLLECTION_SUBPLOTS: [
                ("title", pymongo.TEXT),
                ("embedding", pymongo.ASCENDING)
            ],
            COLLECTION_RESEARCH: [
                ("content", pymongo.TEXT),
                ("embedding", pymongo.ASCENDING)
            ],
            COLLECTION_ANALYSIS: [
                ("content", pymongo.TEXT),
                ("embedding", pymongo.ASCENDING)
            ]
        }

        with _client_lock:
            for collection_name, indexes in collections.items():
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)

                collection = self.db[collection_name]
                for field, index_type in indexes:
                    index_name = f"{field}_idx"
                    if index_name not in collection.index_information():
                        collection.create_index(
                            [(field, index_type)],
                            name=index_name,
                            sparse=True
                        )

    async def store_document(
        self,
        collection: str,
        document: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> str:
        """Store a document with optional embedding."""
        if embedding:
            document["embedding"] = embedding
        result = self.db[collection].insert_one(document)
        return str(result.inserted_id)
        
    def get_document(self, collection: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return self.db[collection].find_one({"_id": ObjectId(document_id)})
        
    def update_document(self, collection: str, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document."""
        result = self.db[collection].update_one(
            {"_id": ObjectId(document_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
        
    def delete_document(self, collection: str, document_id: str) -> bool:
        """Delete a document."""
        result = self.db[collection].delete_one({"_id": ObjectId(document_id)})
        return result.deleted_count > 0
        
    def query_documents(self, collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query documents with a filter dictionary."""
        return list(self.db[collection].find(query))
        
    def similarity_search(self, collection: str, query: str, k: int = 5) -> List[Document]:
        """Search for documents similar to the query."""
        # Generate embedding for the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Find documents with similar embeddings
        cursor = self.db[collection].aggregate([
            {
                "$vectorSearch": {
                    "index": f"{collection}_vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": k * 3,  # Retrieve more candidates for better results
                    "limit": k
                }
            }
        ])
        
        # Convert to Document objects
        results = []
        for doc in cursor:
            # Skip documents without content
            if "content" not in doc:
                continue
                
            # Create Document object
            document = Document(
                page_content=doc["content"],
                metadata={k: v for k, v in doc.items() if k not in ["_id", "content", "embedding"]}
            )
            results.append(document)
            
        return results
        
    def store_documents_with_embeddings(self, collection: str, documents: List[Document]) -> List[str]:
        """Store multiple documents with embeddings."""
        # Generate embeddings in batch
        texts = [doc.page_content for doc in documents]
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Store documents with embeddings
        doc_ids = []
        for doc, embedding in zip(documents, embeddings_list):
            document_data = {
                "content": doc.page_content,
                "embedding": embedding,
                **doc.metadata
            }
            doc_id = self.store_document(collection, document_data)
            doc_ids.append(doc_id)
            
        return doc_ids


# Store singleton to prevent repeated instantiation
_store_instance = None
_store_lock = threading.Lock()

def get_mongodb_store() -> MongoDBStore:
    """Get or create a MongoDBStore singleton."""
    global _store_instance
    
    with _store_lock:
        if _store_instance is None:
            _store_instance = MongoDBStore()
            
    return _store_instance
