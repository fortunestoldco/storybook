from typing import Dict, Any, List, Optional
from langchain_mongodb.indexes import MongoDBRecordManager
from pymongo.collection import Collection
from storybook.db_config import get_collection, MONGODB_URI, DB_NAME

class RecordManagerFactory:
    """Factory for creating MongoDB record managers."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize with MongoDB connection string."""
        self.connection_string = connection_string or MONGODB_URI
        self.database_name = DB_NAME
        
    def create_record_manager(self, collection_name: str) -> MongoDBRecordManager:
        """Create a MongoDB record manager for a collection."""
        collection = get_collection(collection_name)
        return MongoDBRecordManager(collection=collection)