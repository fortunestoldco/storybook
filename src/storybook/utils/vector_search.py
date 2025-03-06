from typing import List, Dict, Any
import numpy as np
from langchain_core.embeddings import Embeddings

class VectorSearch:
    """Vector search implementation for similarity matching."""
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.documents: List[str] = []
        self.document_embeddings: List[np.ndarray] = []

    async def add_documents(self, documents: List[str]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
            
        embeddings = await self.embeddings.aembed_documents(documents)
        self.documents.extend(documents)
        self.document_embeddings.extend(embeddings)

    async def similarity_search(
        self, 
        query: str, 
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        query_embedding = await self.embeddings.aembed_query(query)
        
        if not self.document_embeddings:
            return []

        # Convert to numpy arrays for efficient computation
        query_array = np.array(query_embedding)
        doc_array = np.array(self.document_embeddings)

        # Calculate similarities using dot product
        similarities = np.dot(doc_array, query_array)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "content": self.documents[idx],
                "similarity": float(similarities[idx])
            })
            
        return results

    def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.documents.clear()
        self.document_embeddings.clear()