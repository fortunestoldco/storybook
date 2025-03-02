import os
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config import MONGODB_CONFIG


class MongoDBManager:
    """Manager for MongoDB operations."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the MongoDB manager.

        Args:
            connection_string: MongoDB connection string. If None, uses the one from config.
        """
        self.connection_string = connection_string or os.getenv("MONGODB_URI", MONGODB_CONFIG["connection_string"])
        self.database_name = os.getenv("MONGODB_DB", MONGODB_CONFIG["database_name"])
        self.client = MongoClient(self.connection_string)
        
        # Check if database exists, if not create it
        self._initialize_database()
        
        self.db = self.client[self.database_name]

    def _initialize_database(self):
        """Initialize the database and collections if they don't exist."""
        # Check if database exists in list of database names
        existing_dbs = self.client.list_database_names()
        
        if self.database_name not in existing_dbs:
            print(f"Creating new database: {self.database_name}")
            # MongoDB actually creates the database when you first create a collection
            db = self.client[self.database_name]
            
            # Create all required collections
            collections_config = MONGODB_CONFIG["collections"]
            for collection_name in collections_config.values():
                db.create_collection(collection_name)
                print(f"Created collection: {collection_name}")
            
            # Create vector search index on documents collection
            self._create_vector_search_index()
        else:
            # Check if all required collections exist
            db = self.client[self.database_name]
            existing_collections = db.list_collection_names()
            collections_config = MONGODB_CONFIG["collections"]
            
            for collection_name in collections_config.values():
                if collection_name not in existing_collections:
                    print(f"Creating missing collection: {collection_name}")
                    db.create_collection(collection_name)
            
            # Check if vector search index exists, create if not
            try:
                indexes = list(db[MONGODB_CONFIG["collections"]["documents"]].list_indexes())
                vector_index_exists = any("vector" in idx.get("name", "") for idx in indexes)
                
                if not vector_index_exists:
                    self._create_vector_search_index()
            except Exception as e:
                print(f"Error checking vector index: {str(e)}")
                # Try to create vector index anyway
                self._create_vector_search_index()
    
    def _create_vector_search_index(self):
        """Create a vector search index on the documents collection."""
        try:
            db = self.client[self.database_name]
            documents_collection = db[MONGODB_CONFIG["collections"]["documents"]]
            
            # Create a basic index to support vector search
            documents_collection.create_index([("embedding", ASCENDING)])
            
            print(f"Created vector search index on documents collection")
            
            # Setup vector store for future use
            embeddings = OpenAIEmbeddings()
            
            # Initialize with langchain-mongodb
            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=self.connection_string,
                namespace=f"{self.database_name}.{MONGODB_CONFIG['collections']['documents']}",
                embedding=embeddings,
                index_name="vector_index"
            )
            
            print(f"Initialized vector search capabilities")
            
        except OperationFailure as e:
            print(f"Error creating vector search index: {str(e)}")
            print("Will attempt to continue without vector search capabilities")

    def get_collection(self, collection_name: str) -> Collection:
        """Get a collection by name.

        Args:
            collection_name: Name of the collection to get.

        Returns:
            The requested collection.
        """
        return self.db[collection_name]

    def save_state(self, project_id: str, state: Dict) -> None:
        """Save the project state to MongoDB.

        Args:
            project_id: ID of the project.
            state: The state to save.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["project_state"])
        collection.update_one(
            {"project_id": project_id},
            {"$set": state},
            upsert=True
        )

    def load_state(self, project_id: str) -> Optional[Dict]:
        """Load the project state from MongoDB.

        Args:
            project_id: ID of the project.

        Returns:
            The loaded state, or None if not found.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["project_state"])
        return collection.find_one({"project_id": project_id})

    def save_document(self, document: Dict) -> None:
        """Save a document to MongoDB.

        Args:
            document: The document to save.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["documents"])
        if "_id" in document:
            collection.replace_one({"_id": document["_id"]}, document, upsert=True)
        else:
            collection.insert_one(document)

    def load_document(self, document_id: str) -> Optional[Dict]:
        """Load a document from MongoDB.

        Args:
            document_id: ID of the document to load.

        Returns:
            The loaded document, or None if not found.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["documents"])
        return collection.find_one({"_id": document_id})

    def save_research(self, research: Dict) -> None:
        """Save research data to MongoDB.

        Args:
            research: The research data to save.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["research"])
        if "_id" in research:
            collection.replace_one({"_id": research["_id"]}, research, upsert=True)
        else:
            collection.insert_one(research)

    def load_research(self, query: Dict) -> List[Dict]:
        """Load research data from MongoDB.

        Args:
            query: Query to filter research data.

        Returns:
            List of matching research documents.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["research"])
        return list(collection.find(query))

    def save_feedback(self, feedback: Dict) -> None:
        """Save human feedback to MongoDB.

        Args:
            feedback: The feedback to save.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["feedback"])
        collection.insert_one(feedback)

    def load_feedback(self, project_id: str) -> List[Dict]:
        """Load human feedback for a project from MongoDB.

        Args:
            project_id: ID of the project.

        Returns:
            List of feedback documents.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["feedback"])
        return list(collection.find({"project_id": project_id}))

    def save_metrics(self, metrics: Dict) -> None:
        """Save quality metrics to MongoDB.

        Args:
            metrics: The metrics to save.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["metrics"])
        if "_id" in metrics:
            collection.replace_one({"_id": metrics["_id"]}, metrics, upsert=True)
        else:
            collection.insert_one(metrics)

    def load_metrics(self, project_id: str) -> List[Dict]:
        """Load quality metrics for a project from MongoDB.

        Args:
            project_id: ID of the project.

        Returns:
            List of metrics documents.
        """
        collection = self.get_collection(MONGODB_CONFIG["collections"]["metrics"])
        return list(collection.find({"project_id": project_id}))
    
    def vector_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Perform a vector search on documents.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            
        Returns:
            List of matching documents.
        """
        try:
            from langchain_core.documents import Document
            
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            
            # Initialize vector store
            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=self.connection_string,
                namespace=f"{self.database_name}.{MONGODB_CONFIG['collections']['documents']}",
                embedding=embeddings,
                index_name="vector_index"
            )
            
            # Perform the search
            results = vector_store.similarity_search(query, k=limit)
            
            # Convert documents to dictionaries
            return [doc.metadata for doc in results]
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            # Fallback to regular search if vector search fails
            collection = self.get_collection(MONGODB_CONFIG["collections"]["documents"])
            return list(collection.find({"$text": {"$search": query}}).limit(limit))