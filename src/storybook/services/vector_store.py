"""Vector store connection and operations for the Storybook application."""

import os
from typing import Dict, List, Any, Optional, Union
import pinecone
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from storybook.config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT,
    VECTOR_INDEX_NAME, VECTOR_DIMENSION
)

class VectorStoreConnector:
    """Connector for vector database operations."""
    
    def __init__(self, index_name: str = None):
        """Initialize the vector store connector."""
        self.index_name = index_name or VECTOR_INDEX_NAME
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self._initialize_pinecone()
        self.vectorstore = None
        self._connect()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client."""
        try:
            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=VECTOR_DIMENSION,
                    metric="cosine"
                )
        except Exception as e:
            print(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def _connect(self):
        """Connect to the vector store."""
        try:
            self.vectorstore = Pinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
        except Exception as e:
            print(f"Error connecting to vector store: {str(e)}")
            raise
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to the vector store."""
        if not self.vectorstore:
            self._connect()
        
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not self.vectorstore:
            self._connect()
        
        return self.vectorstore.add_documents(documents=documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents in the vector store."""
        if not self.vectorstore:
            self._connect()
        
        return self.vectorstore.similarity_search(query=query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents in the vector store with relevance scores."""
        if not self.vectorstore:
            self._connect()
        
        return self.vectorstore.similarity_search_with_score(query=query, k=k)
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents from the vector store by ID."""
        if not self.vectorstore:
            self._connect()
        
        index = pinecone.Index(self.index_name)
        index.delete(ids=ids)
