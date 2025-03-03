# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config import MONGODB_CONFIG


class MongoDBManager:
    """
    Manager for MongoDB operations.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the MongoDB manager.
        
        Args:
            connection_string: MongoDB connection string. If None, uses the one from config.
        """
        self.connection_string = connection_string or MONGODB_CONFIG["connection_string"]
        self.client = MongoClient(self.connection_string)
        self.db = self.client[MONGODB_CONFIG["database_name"]]
    
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

