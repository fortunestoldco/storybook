from typing import Dict, List, Optional, Any
import logging
import os
from bson.objectid import ObjectId

from langchain_core.documents import Document
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection

# Consider adding these imports for langchain-mongodb integration
# from langchain_mongodb import MongoDBAtlasVectorSearch
# from langchain_mongodb.vectorstores import MongoDBVectorStore

from storybook.config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)


class MongoDBStore:
    """Interface to MongoDB for document storage and retrieval."""

    def __init__(self):
        """Initialize the MongoDB connection."""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB_NAME]
        
        # If using langchain-mongodb, you could initialize it like:
        # self.vector_store = MongoDBVectorStore(
        #     client=self.client,
        #     db_name=MONGODB_DB_NAME,
        #     collection_name="vectors"
        # )

        # Initialize the collections if they don't exist
        self._ensure_collections_exist()

    # Rest of the class implementation
    # ...
    
    def store_documents_with_embeddings(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """Store documents with their embeddings in the specified collection."""
        # This method could be updated to use langchain-mongodb:
        
        # For example:
        # if collection_name == "vectors":
        #     ids = self.vector_store.add_documents(documents)
        #     return ids
        
        # Current implementation (fallback)
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

    def similarity_search(
        self, collection_name: str, query: str, k: int = 5
    ) -> List[Document]:
        """Search for documents similar to the query."""
        # This method could be updated to use langchain-mongodb:
        
        # For example:
        # if collection_name == "vectors":
        #     return self.vector_store.similarity_search(query, k=k)
        
        # Current implementation (fallback)
        collection = self.get_collection(collection_name)

        # Simple text search (crude approximation of similarity search)
        documents = collection.find({"$text": {"$search": query}}).limit(k)

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