from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Result from a vector search operation."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score
        }

class VectorSearch:
    """Interface for vector search operations."""
    
    async def search(self, query: str, limit: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform semantic search.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents (each with "content" and "metadata")
            
        Returns:
            List of document IDs
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        raise NotImplementedError("Subclasses must implement this method")