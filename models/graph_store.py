from typing import Dict, Any, List, Optional
from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from db_config import MONGODB_URI, DB_NAME

class GraphStoreManager:
    """Manager for MongoDB-based knowledge graph operations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize with MongoDB connection string."""
        self.connection_string = connection_string or MONGODB_URI
        self.database_name = DB_NAME
        
    def create_graph_store(self, entity_extraction_model: BaseChatModel,
                          collection_name: str = "knowledge_graph",
                          entity_prompt: Optional[ChatPromptTemplate] = None,
                          query_prompt: Optional[ChatPromptTemplate] = None,
                          max_depth: int = 2,
                          allowed_entity_types: Optional[List[str]] = None,
                          allowed_relationship_types: Optional[List[str]] = None,
                          entity_examples: Optional[str] = None,
                          entity_name_examples: Optional[str] = None,
                          validate: bool = False,
                          validation_action: str = "warn") -> MongoDBGraphStore:
        """Create a MongoDB-based knowledge graph store."""
        return MongoDBGraphStore(
            connection_string=self.connection_string,
            database_name=self.database_name,
            collection_name=collection_name,
            entity_extraction_model=entity_extraction_model,
            entity_prompt=entity_prompt,
            query_prompt=query_prompt,
            max_depth=max_depth,
            allowed_entity_types=allowed_entity_types,
            allowed_relationship_types=allowed_relationship_types,
            entity_examples=entity_examples,
            entity_name_examples=entity_name_examples,
            validate=validate,
            validation_action=validation_action
        )