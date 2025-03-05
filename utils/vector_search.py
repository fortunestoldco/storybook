from typing import Dict, Any, List, Optional
import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearchSettings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class SearchResult:
    """Represents a search result from vector search."""
    def __init__(self, title: str, url: str, content: str, score: float = 0.0, raw_content: Optional[str] = None):
        self.title = title
        self.url = url
        self.content = content
        self.score = score
        self.raw_content = raw_content

class VectorSearch:
    """Manages vector search operations across different collections."""
    
    def __init__(self, embeddings: Embeddings, connection_string: Optional[str] = None, database_name: Optional[str] = None):
        """Initialize with embedding model."""
        self.embeddings = embeddings
        self.connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017")
        self.database_name = database_name or os.getenv("MONGODB_DATABASE_NAME", "storybook_system")
        
    def get_vectorstore(self, collection_name: str) -> MongoDBAtlasVectorSearch:
        """Get a vector store for a specific collection."""
        client = MongoClient(self.connection_string)
        collection = client[self.database_name][collection_name]
        
        settings = MongoDBAtlasVectorSearchSettings(
            index_name=f"{collection_name}_index",
            text_key="text",
            embedding_key="embedding",
            metadata_key="metadata"
        )
        
        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embeddings,
            index_name=f"{collection_name}_index",
            settings=settings
        )
    
    def create_vector_search_index(self, collection_name: str, dimensions: int = 1536):
        """Create vector search index for a collection."""
        vectorstore = self.get_vectorstore(collection_name)
        vectorstore.create_vector_search_index(dimensions=dimensions)
        return vectorstore
    
    def add_texts(self, collection_name: str, texts: List[str], 
                 metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to vector store and return IDs."""
        vectorstore = self.get_vectorstore(collection_name)
        return vectorstore.add_texts(texts=texts, metadatas=metadatas)
    
    def add_documents(self, collection_name: str, documents: List[Document]) -> List[str]:
        """Add documents to vector store and return IDs."""
        vectorstore = self.get_vectorstore(collection_name)
        return vectorstore.add_documents(documents=documents)
    
    def similarity_search(self, collection_name: str, query: str, k: int = 4, 
                         pre_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search in vector store."""
        vectorstore = self.get_vectorstore(collection_name)
        return vectorstore.similarity_search(query=query, k=k, pre_filter=pre_filter)
    
    def similarity_search_with_score(self, collection_name: str, query: str, k: int = 4,
                                    pre_filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """Perform similarity search with score in vector store."""
        vectorstore = self.get_vectorstore(collection_name)
        return vectorstore.similarity_search_with_score(query=query, k=k, pre_filter=pre_filter)
    
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete documents from vector store by IDs."""
        vectorstore = self.get_vectorstore(collection_name)
        return vectorstore.delete(ids=ids)
    
    def as_retriever(self, collection_name: str, search_type: str = "similarity", 
                    search_kwargs: Optional[Dict[str, Any]] = None):
        """Get vector store as a retriever."""
        vectorstore = self.get_vectorstore(collection_name)
        return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
