from typing import Dict, Any, List, Optional, Sequence
from langchain_mongodb.loaders import MongoDBLoader
from langchain_core.documents import Document
from pymongo.collection import Collection
from storybook.db_config import get_collection

class MongoDBDocumentLoader:
    """Loader for documents from MongoDB collections."""
    
    def load_from_collection(self, collection_name: str, 
                            filter_criteria: Optional[Dict] = None,
                            field_names: Optional[Sequence[str]] = None,
                            metadata_names: Optional[Sequence[str]] = None,
                            include_db_collection_in_metadata: bool = True) -> List[Document]:
        """Load documents from a MongoDB collection."""
        collection = get_collection(collection_name)
        
        loader = MongoDBLoader(
            collection=collection,
            filter_criteria=filter_criteria,
            field_names=field_names,
            metadata_names=metadata_names,
            include_db_collection_in_metadata=include_db_collection_in_metadata
        )
        
        return loader.load()