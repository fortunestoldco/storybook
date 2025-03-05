from typing import Dict, Any, List, Optional
from langchain_mongodb.cache import MongoDBCache, MongoDBAtlasSemanticCache
from langchain_core.embeddings import Embeddings
from db_config import MONGODB_URI, DB_NAME

class CacheManager:
    """Manages different types of MongoDB caches."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize with MongoDB connection string."""
        self.connection_string = connection_string or MONGODB_URI
        self.database_name = DB_NAME
        
    def get_model_cache(self, collection_name: str = "model_cache") -> MongoDBCache:
        """Get a standard MongoDB cache for LLM responses."""
        return MongoDBCache(
            connection_string=self.connection_string,
            database_name=self.database_name,
            collection_name=collection_name
        )
    
    def get_semantic_cache(self, embeddings: Embeddings, 
                          collection_name: str = "semantic_cache",
                          similarity_threshold: float = 0.9) -> MongoDBAtlasSemanticCache:
        """Get a semantic cache for LLM responses based on embeddings."""
        return MongoDBAtlasSemanticCache(
            connection_string=self.connection_string,
            database_name=self.database_name,
            collection_name=collection_name,
            embedding=embeddings,
            similarity_threshold=similarity_threshold
        )