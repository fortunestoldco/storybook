from typing import Dict, List, Optional, Any
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from storybook.db.mongodb_client import MongoDBStore
from storybook.config import (
    COLLECTION_MANUSCRIPTS,
    COLLECTION_CHARACTERS,
    COLLECTION_WORLDS,
    COLLECTION_SUBPLOTS,
    COLLECTION_RESEARCH,
    COLLECTION_ANALYSIS,
)

logger = logging.getLogger(__name__)

class DocumentStore:
    """Manages document storage and retrieval."""
    
    def __init__(self):
        """Initialize the document store."""
        self.db = MongoDBStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )

    def store_manuscript(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a manuscript in the database."""
        document = {
            "title": title,
            "content": content,
            "metadata": metadata or {}
        }
        return self.db.store_document(COLLECTION_MANUSCRIPTS, document)

    def get_manuscript(self, manuscript_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a manuscript from the database."""
        return self.db.get_document(COLLECTION_MANUSCRIPTS, manuscript_id)

    def update_manuscript(self, manuscript_id: str, updates: Dict[str, Any]) -> bool:
        """Update a manuscript in the database."""
        return self.db.update_document(COLLECTION_MANUSCRIPTS, manuscript_id, updates)

    def store_character_details(self, manuscript_id: str, character_name: str, details: Dict[str, Any]) -> str:
        """Store character details in the database."""
        document = {
            "manuscript_id": manuscript_id,
            "character_name": character_name,
            **details
        }
        return self.db.store_document(COLLECTION_CHARACTERS, document)

    def store_world_details(self, manuscript_id: str, name: str, details: Dict[str, Any]) -> str:
        """Store world-building details in the database."""
        document = {
            "manuscript_id": manuscript_id,
            "name": name,
            **details
        }
        return self.db.store_document(COLLECTION_WORLDS, document)

    def store_subplot(self, manuscript_id: str, title: str, details: Dict[str, Any]) -> str:
        """Store subplot details in the database."""
        document = {
            "manuscript_id": manuscript_id,
            "title": title,
            **details
        }
        return self.db.store_document(COLLECTION_SUBPLOTS, document)

    def store_research_document(self, manuscript_id: str, research_type: str, data: Dict[str, Any]) -> str:
        """Store research document with associated manuscript."""
        document = {
            "manuscript_id": manuscript_id,
            "research_type": research_type,
            "data": data,
            "timestamp": self._get_timestamp()
        }
        return self.db.store_document(COLLECTION_RESEARCH, document)

    def store_analysis_document(self, manuscript_id: str, analysis_type: str, data: Dict[str, Any]) -> str:
        """Store analysis document with associated manuscript."""
        document = {
            "manuscript_id": manuscript_id,
            "analysis_type": analysis_type,
            "data": data,
            "timestamp": self._get_timestamp()
        }
        return self.db.store_document(COLLECTION_ANALYSIS, document)

    def search_research(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant research documents."""
        return self.db.similarity_search(COLLECTION_RESEARCH, query, k=k)
        
    def store_research_from_web(self, urls: List[str], tags: Optional[List[str]] = None) -> List[str]:
        """Crawl websites and store research data."""
        all_documents = []
        metadata = {"source_type": "web", "tags": tags or []}
        
        # Use WebBaseLoader
        try:
            loader = WebBaseLoader(urls)
            documents = loader.load()
            # Add metadata
            for doc in documents:
                doc.metadata.update(metadata)
            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"WebBaseLoader failed: {e}")
            return []
        
        # Split documents
        split_documents = self.text_splitter.split_documents(all_documents)
        
        # Store in MongoDB with embeddings
        return self.db.store_documents_with_embeddings(
            COLLECTION_RESEARCH, 
            split_documents
        )

    def store_manuscript_chunks(self, manuscript_id: str, title: str, content: str) -> List[str]:
        """Split and store manuscript chunks with embeddings."""
        doc = Document(
            page_content=content,
            metadata={"manuscript_id": manuscript_id, "title": title}
        )
        chunks = self.text_splitter.split_documents([doc])
        return self.db.store_documents_with_embeddings(COLLECTION_MANUSCRIPTS, chunks)

    def get_manuscript_relevant_parts(self, manuscript_id: str, query: str, k: int = 5) -> List[Document]:
        """Get relevant parts of a manuscript based on a query."""
        # Get all chunks for this manuscript
        results = self.db.similarity_search(COLLECTION_MANUSCRIPTS, query, k=k)
        return [doc for doc in results if doc.metadata.get("manuscript_id") == manuscript_id]

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
