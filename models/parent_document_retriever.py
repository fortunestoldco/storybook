from typing import Dict, Any, List, Optional
from langchain_mongodb.retrievers.parent_document import MongoDBAtlasParentDocumentRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from db_config import MONGODB_URI, DB_NAME

class ParentDocumentRetrieverManager:
    """Manages parent document retriever operations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize with MongoDB connection string."""
        self.connection_string = connection_string or MONGODB_URI
        self.database_name = DB_NAME
        
    def create_retriever(self, embedding_model: Embeddings, 
                         child_splitter: TextSplitter,
                         collection_name: str = "document_with_chunks",
                         id_key: str = "doc_id") -> MongoDBAtlasParentDocumentRetriever:
        """Create a parent document retriever."""
        return MongoDBAtlasParentDocumentRetriever.from_connection_string(
            connection_string=self.connection_string,
            embedding_model=embedding_model,
            child_splitter=child_splitter,
            database_name=self.database_name,
            collection_name=collection_name,
            id_key=id_key
        )
    
    def add_documents(self, retriever: MongoDBAtlasParentDocumentRetriever, 
                     documents: List[Document]) -> None:
        """Add documents to the retriever."""
        retriever.add_documents(documents)
    
    def retrieve(self, retriever: MongoDBAtlasParentDocumentRetriever, 
                query: str, **kwargs) -> List[Document]:
        """Retrieve documents based on a query."""
        return retriever.invoke(query, **kwargs)