from __future__ import annotations

# Standard library imports
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

# Third-party imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.web_base import WebBaseLoader

# Local imports
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
            chunk_size=1500, chunk_overlap=150
        )
        # Remove direct embedding initialization
        # self.embeddings = OpenAIEmbeddings()
        # Use the vector store from MongoDBStore instead

    async def store_document(
        self,
        collection: str,
        document: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> str:
        """Store a document with optional embedding."""
        if embedding:
            document["embedding"] = embedding
        result = self.db.db[collection].insert_one(document)
        return str(result.inserted_id)

    def store_manuscript(
        self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a manuscript in the database."""
        document = {"title": title, "content": content, "metadata": metadata or {}}
        return self.db.store_document(COLLECTION_MANUSCRIPTS, document)

    def get_manuscript(self, manuscript_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a manuscript from the database."""
        return self.db.get_document(COLLECTION_MANUSCRIPTS, manuscript_id)

    def update_manuscript(self, manuscript_id: str, updates: Dict[str, Any]) -> bool:
        """Update a manuscript in the database."""
        return self.db.update_document(COLLECTION_MANUSCRIPTS, manuscript_id, updates)

    def store_character_details(
        self, manuscript_id: str, character_name: str, details: Dict[str, Any]
    ) -> str:
        """Store character details in the database."""
        document = {
            "manuscript_id": manuscript_id,
            "character_name": character_name,
            **details,
        }
        return self.db.store_document(COLLECTION_CHARACTERS, document)

    def store_world_details(
        self, manuscript_id: str, name: str, details: Dict[str, Any]
    ) -> str:
        """Store world-building details in the database."""
        document = {"manuscript_id": manuscript_id, "name": name, **details}
        return self.db.store_document(COLLECTION_WORLDS, document)

    def store_subplot(
        self, manuscript_id: str, title: str, details: Dict[str, Any]
    ) -> str:
        """Store subplot details in the database."""
        document = {"manuscript_id": manuscript_id, "title": title, **details}
        return self.db.store_document(COLLECTION_SUBPLOTS, document)

    def store_research_document(
        self, manuscript_id: str, research_type: str, data: Dict[str, Any]
    ) -> str:
        """Store research document with associated manuscript."""
        document = {
            "manuscript_id": manuscript_id,
            "research_type": research_type,
            "data": data,
            "timestamp": self._get_timestamp(),
        }
        return self.db.store_document(COLLECTION_RESEARCH, document)

    def store_analysis_document(
        self, manuscript_id: str, analysis_type: str, data: Dict[str, Any]
    ) -> str:
        """Store analysis document with associated manuscript."""
        document = {
            "manuscript_id": manuscript_id,
            "analysis_type": analysis_type,
            "data": data,
            "timestamp": self._get_timestamp(),
        }
        return self.db.store_document(COLLECTION_ANALYSIS, document)

    def search_research(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant research documents."""
        return self.db.similarity_search(COLLECTION_RESEARCH, query, k=k)

    def store_research_from_web(
        self, urls: List[str], tags: Optional[List[str]] = None
    ) -> List[str]:
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
            COLLECTION_RESEARCH, split_documents
        )

    def store_manuscript_chunks(
        self, manuscript_id: str, title: str, content: str
    ) -> List[str]:
        """Split and store manuscript chunks with embeddings."""
        doc = Document(
            page_content=content,
            metadata={"manuscript_id": manuscript_id, "title": title},
        )
        chunks = self.text_splitter.split_documents([doc])
        return self.db.store_documents_with_embeddings(COLLECTION_MANUSCRIPTS, chunks)

    def get_manuscript_relevant_parts(
        self, manuscript_id: str, query: str, k: int = 5
    ) -> List[Document]:
        """Get relevant parts of a manuscript based on a query."""
        # Get all chunks for this manuscript
        filter_dict = {"metadata.manuscript_id": manuscript_id}
        results = self.db.similarity_search(COLLECTION_MANUSCRIPTS, query, k=k)

        # If we don't find enough results with the filter, try a broader search
        if len(results) < k:
            # Get any documents for this manuscript
            manuscript_docs = self.db.query_documents(
                COLLECTION_MANUSCRIPTS, {"manuscript_id": manuscript_id}
            )

            if manuscript_docs:
                # Create basic text chunks if no vectorized chunks exist
                content = manuscript_docs[0].get("content", "")
                if content:
                    chunks = self._simple_chunk_text(content, 1000, 100)
                    results = [
                        Document(
                            page_content=chunk,
                            metadata={
                                "manuscript_id": manuscript_id,
                                "title": manuscript_docs[0].get("title", ""),
                            },
                        )
                        for chunk in chunks
                    ]

                    # Try to match query terms in chunks
                    query_terms = set(query.lower().split())
                    scored_chunks = []

                    for doc in results:
                        score = sum(
                            1
                            for term in query_terms
                            if term in doc.page_content.lower()
                        )
                        scored_chunks.append((doc, score))

                    # Sort by score and take top k
                    scored_chunks.sort(key=lambda x: x[1], reverse=True)
                    results = [doc for doc, _ in scored_chunks[:k]]

        return [
            doc for doc in results if doc.metadata.get("manuscript_id") == manuscript_id
        ]

    def _simple_chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple text chunking as fallback."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to end at a sentence boundary
            if end < len(text):
                sentence_end = text.rfind(". ", start, end) + 1
                if sentence_end > start:
                    end = sentence_end

            chunks.append(text[start:end])
            start = end - overlap

            # Make sure we make progress
            if start >= end:
                start = end

        return chunks

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
