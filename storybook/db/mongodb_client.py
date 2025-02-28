from __future__ import annotations

# Standard library imports
from typing import Any, Dict, List, Optional
import logging
import os

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

class MongoDBStore:
    """MongoDB storage implementation."""
    
    def __init__(self):
        """Initialize MongoDB connection and embeddings."""
        try:
            # Get MongoDB URI from environment, fallback to default
            mongodb_uri = os.getenv('MONGODB_URI', MONGODB_URI)
            logger.info(f"Connecting to MongoDB at: {mongodb_uri}")
            
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            self.db = self.client[MONGODB_DB_NAME]
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY
            )
            # Test connection
            self.client.server_info()
            logger.info("Successfully connected to MongoDB")
            # Initialize collections
            self._ensure_collections_exist()
        except ServerSelectionTimeoutError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.error(f"Attempted connection to: {mongodb_uri}")
            raise

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
