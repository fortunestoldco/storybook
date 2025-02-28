from __future__ import annotations

# Standard library imports
from typing import Any, Dict, List, Optional
import logging

# Third-party imports
from pymongo import MongoClient
from bson.objectid import ObjectId  # Changed import location
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid, OperationFailure

from langchain_core.documents import Document
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

# Local imports
from storybook.config import (
    MONGODB_URI,    MONGODB_URI,
    MONGODB_DB_NAME,
    COLLECTION_MANUSCRIPTS,    COLLECTION_MANUSCRIPTS,
    COLLECTION_CHARACTERS,    COLLECTION_CHARACTERS,
    COLLECTION_WORLDS,    COLLECTION_WORLDS,
    COLLECTION_SUBPLOTS,LOTS,
    COLLECTION_RESEARCH,
    COLLECTION_ANALYSIS,    COLLECTION_ANALYSIS,
)

logger = logging.getLogger(__name__)



class MongoDBStore:
    """Interface to MongoDB for document storage and retrieval."""    """Interface to MongoDB for document storage and retrieval."""

    def __init__(self):
        """Initialize the MongoDB connection."""on."""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB_NAME]DB_NAME]

        # Initialize embeddings for vector search Initialize embeddings for vector search
        self.embeddings = OpenAIEmbeddings()        self.embeddings = OpenAIEmbeddings()

        # Initialize vector search
        self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(        self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=MONGODB_URI,I,
            namespace=f"{MONGODB_DB_NAME}.vectors",
            embedding=self.embeddings,elf.embeddings,
            index_name="vector_index",ctor_index",
        )

        # Initialize the collections if they don't existe collections if they don't exist
        self._ensure_collections_exist()llections_exist()

    def _ensure_collections_exist(self):tions_exist(self):
        """Ensure all required collections exist."""""Ensure all required collections exist."""
        collections = [        collections = [
            "manuscripts",
            "characters",
            "worlds",
            "subplots",
            "research",            "research",
            "analysis",
            "vectors",
        ]

        for collection in collections:ions:
            # Check if collection exists, if not create ittion exists, if not create it
            if collection not in self.db.list_collection_names():n self.db.list_collection_names():
                self.db.create_collection(collection)collection(collection)

                # Create indices for faster lookupsCreate indices for faster lookups
                if collection == "manuscripts":
                    self.db[collection].create_index("title")                    self.db[collection].create_index("title")
                elif collection in [
                    "characters",
                    "worlds",
                    "subplots",   "subplots",
                    "research",                    "research",
                    "analysis",
                ]:                ]:
                    self.db[collection].create_index("manuscript_id")id")

                # Create text index for basic text search basic text search
                self.db[collection].create_index(                self.db[collection].create_index(
                    [("content", pymongo.TEXT)], sparse=True
                )

                logger.info(f"Created collection: {collection}"): {collection}")

    def get_collection(self, collection_name: str) -> Collection:    def get_collection(self, collection_name: str) -> Collection:
        """Get a MongoDB collection."""oDB collection."""
        return self.db[collection_name]

    def store_document(self, collection_name: str, document: Dict[str, Any]) -> str:[str, Any]) -> str:
        """Store a document in the specified collection."""""
        collection = self.get_collection(collection_name)ection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)t.inserted_id)

    def get_document(
        self, collection_name: str, document_id: strn_name: str, document_id: str
    ) -> Optional[Dict[str, Any]]:y]]:
        """Get a document by ID from the specified collection."""
        collection = self.get_collection(collection_name)lf.get_collection(collection_name)
        try:        try:
            document = collection.find_one({"_id": ObjectId(document_id)})ollection.find_one({"_id": ObjectId(document_id)})
            if document:
                document["_id"] = str(document["_id"])  document["_id"] = str(document["_id"])
                return document
            return None
        except Exception as e:pt Exception as e:
            logger.error(f"Error retrieving document from {collection_name}: {e}") document from {collection_name}: {e}")
            return None

    def update_document(
        self, collection_name: str, document_id: str, updates: Dict[str, Any] str, document_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update a document in the specified collection."""ment in the specified collection."""
        collection = self.get_collection(collection_name)        collection = self.get_collection(collection_name)
        try:
            result = collection.update_one(
                {"_id": ObjectId(document_id)}, {"$set": updates}updates}
            ))
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document in {collection_name}: {e}")or updating document in {collection_name}: {e}")
            return False

    def delete_document(self, collection_name: str, document_id: str) -> bool:    def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document from the specified collection."""ment from the specified collection."""
        collection = self.get_collection(collection_name)
        try:
            result = collection.delete_one({"_id": ObjectId(document_id)})document_id)})
            return result.deleted_count > 0
        except Exception as e:pt Exception as e:
            logger.error(f"Error deleting document from {collection_name}: {e}")rom {collection_name}: {e}")
            return False

    def query_documents(
        self, collection_name: str, query: Dict[str, Any]e: str, query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query documents from the specified collection."""
        collection = self.get_collection(collection_name)self.get_collection(collection_name)
        try:        try:
            documents = list(collection.find(query))find(query))
            for document in documents:
                if "_id" in document:"_id" in document:
                    document["_id"] = str(document["_id"])
            return documents
        except Exception as e:
            logger.error(f"Error querying documents from {collection_name}: {e}")er.error(f"Error querying documents from {collection_name}: {e}")
            return []

    def store_documents_with_embeddings(mbeddings(
        self, collection_name: str, documents: List[Document], documents: List[Document]
    ) -> List[str]:
        """Store documents with their embeddings in the specified collection."""s with their embeddings in the specified collection."""
        # If storing to vectors collection, use the vector store        # If storing to vectors collection, use the vector store
        if collection_name == "vectors":
            try:
                # Add documents to vector store                # Add documents to vector store
                doc_ids = self.vector_store.add_documents(documents)e.add_documents(documents)
                return doc_ids
            except Exception as e:as e:
                logger.error(f"Error storing documents with embeddings: {e}")toring documents with embeddings: {e}")
                return []

        # For other collections, store documents normally but also add to vectorsections, store documents normally but also add to vectors
        document_ids = []

        # Store in the original collectionre in the original collection
        collection = self.get_collection(collection_name)        collection = self.get_collection(collection_name)
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata

            document = {
                "content": content,
                "metadata": metadata,
            }            }

            result = collection.insert_one(document)result = collection.insert_one(document)
            doc_id = str(result.inserted_id)
            document_ids.append(doc_id)d(doc_id)

            # Update metadata with the original document ID            # Update metadata with the original document ID
            metadata["original_id"] = doc_idnal_id"] = doc_id
            metadata["collection"] = collection_name            metadata["collection"] = collection_name

        # Also store in vector store for search capability
        try:
            self.vector_store.add_documents(documents)
        except Exception as e:pt Exception as e:
            logger.error(f"Error adding to vector store: {e}")ore: {e}")

        return document_ids

    def similarity_search(    def similarity_search(
        self, collection_name: str, query: str, k: int = 5
    ) -> List[Document]:
        """Search for documents similar to the query."""arch for documents similar to the query."""
        try:
            # Use vector store for similarity search for similarity search
            filter_dict = {}
            if collection_name != "vectors":            if collection_name != "vectors":
                filter_dict = {"metadata.collection": collection_name}ame}

            results = self.vector_store.similarity_search(lts = self.vector_store.similarity_search(
                query, k=k, filter=filter_dict
            )
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")ror(f"Error in similarity search: {e}")

            # Fallback to basic text search if vector search failsic text search if vector search fails
            collection = self.get_collection(collection_name)ction = self.get_collection(collection_name)
            try:            try:
                # Try text search if available if available
                documents = list(
                    collection.find(
                        {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}: "textScore"}}
                    )
                    .sort([("score", {"$meta": "textScore"})]){"$meta": "textScore"})])
                    .limit(k)
                )

                if not documents:
                    # Basic keyword matching if text search index is not availableic keyword matching if text search index is not available
                    regex_pattern = "|".join(                    regex_pattern = "|".join(
                        [word for word in query.split() if len(word) > 2]d for word in query.split() if len(word) > 2]
                    )
                    if regex_pattern:
                        documents = list(
                            collection.find(on.find(
                                {"content": {"$regex": regex_pattern, "$options": "i"}}egex_pattern, "$options": "i"}}
                            ).limit(k)                            ).limit(k)
                        )

                results = []
                for doc in documents:
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})ata = doc.get("metadata", {})
                    if "_id" in doc:                    if "_id" in doc:









                return []                logger.error(f"Error in fallback search: {nested_e}")            except Exception as nested_e:                return results                    results.append(Document(page_content=content, metadata=metadata))                        metadata["id"] = str(doc["_id"])                        metadata["id"] = str(doc["_id"])

                    results.append(Document(page_content=content, metadata=metadata))

                return results
            except Exception as nested_e:
                logger.error(f"Error in fallback search: {nested_e}")
                return []
