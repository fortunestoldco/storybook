# test_*.py files need pytest and MagicMock imports
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from storybook.db.document_store import DocumentStore


class TestDocumentStore:

    @pytest.fixture
    def mock_mongodb_store(self):
        with patch("storybook.db.document_store.MongoDBStore") as mock_store:
            # Create mock store instance
            mock_store_instance = MagicMock()
            mock_store.return_value = mock_store_instance

            # Setup mock store behavior
            mock_store_instance.store_document.return_value = "test_id"
            mock_store_instance.get_document.return_value = {
                "title": "Test Title",
                "content": "Test Content",
            }
            mock_store_instance.update_document.return_value = True
            mock_store_instance.store_documents_with_embeddings.return_value = [
                "doc1",
                "doc2",
            ]

            yield mock_store_instance

    @pytest.fixture
    def mock_text_splitter(self):
        with patch(
            "storybook.db.document_store.RecursiveCharacterTextSplitter"
        ) as mock_splitter:
            # Create mock splitter instance
            mock_splitter_instance = MagicMock()
            mock_splitter.return_value = mock_splitter_instance

            # Setup mock splitter behavior
            mock_splitter_instance.split_documents.return_value = [
                Document(page_content="Chunk 1", metadata={}),
                Document(page_content="Chunk 2", metadata={}),
            ]

            yield mock_splitter_instance

    @pytest.fixture
    def mock_web_loader(self):
        with patch("storybook.db.document_store.WebBaseLoader") as mock_loader:
            # Create mock loader instance
            mock_loader_instance = MagicMock()
            mock_loader.return_value = mock_loader_instance

            # Setup mock loader behavior
            mock_loader_instance.load.return_value = [
                Document(page_content="Web content 1", metadata={}),
                Document(page_content="Web content 2", metadata={}),
            ]

            yield mock_loader_instance

    @pytest.fixture
    def document_store(self):
        return DocumentStore()

    def test_init(self, mock_mongodb_store, mock_text_splitter):
        # Test initialization
        store = DocumentStore()

        # Verify store components
        assert store.db == mock_mongodb_store
        assert store.text_splitter == mock_text_splitter

    def test_store_manuscript(self, document_store):
        manuscript = {
            "title": "Test Novel",
            "content": "Test content",
            "metadata": {"author": "Test Author"},
        }

        result = document_store.store_manuscript(manuscript)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_manuscript(self, document_store):
        manuscript_id = "test_123"

        result = document_store.get_manuscript(manuscript_id)

        assert isinstance(result, dict)
        assert "title" in result
        assert "content" in result

    def test_store_manuscript(self, mock_mongodb_store):
        # Setup
        store = DocumentStore()

        # Test storing manuscript
        result = store.store_manuscript(
            "Test Title", "Test Content", {"author": "Test Author"}
        )

        # Verify
        mock_mongodb_store.store_document.assert_called_once_with(
            "manuscripts",
            {
                "title": "Test Title",
                "content": "Test Content",
                "metadata": {"author": "Test Author"},
            },
        )
        assert result == "test_id"

    def test_get_manuscript(self, mock_mongodb_store):
        # Setup
        store = DocumentStore()

        # Test getting manuscript
        result = store.get_manuscript("test_id")

        # Verify
        mock_mongodb_store.get_document.assert_called_once_with(
            "manuscripts", "test_id"
        )
        assert result["title"] == "Test Title"
        assert result["content"] == "Test Content"

    def test_update_manuscript(self, mock_mongodb_store):
        # Setup
        store = DocumentStore()

        # Test updating manuscript
        result = store.update_manuscript("test_id", {"title": "Updated Title"})

        # Verify
        mock_mongodb_store.update_document.assert_called_once_with(
            "manuscripts", "test_id", {"title": "Updated Title"}
        )
        assert result == True

    def test_store_research_from_web(
        self, mock_mongodb_store, mock_web_loader, mock_text_splitter
    ):
        # Setup
        store = DocumentStore()

        # Test storing research from web
        result = store.store_research_from_web(["https://test.com"], ["tag1", "tag2"])

        # Verify WebBaseLoader was called
        mock_web_loader.load.assert_called_once()

        # Verify text splitter was called
        mock_text_splitter.split_documents.assert_called_once()

        # Verify documents were stored with embeddings
        mock_mongodb_store.store_documents_with_embeddings.assert_called_once()

        # Verify result
        assert result == ["doc1", "doc2"]

    def test_store_manuscript_chunks(self, mock_mongodb_store, mock_text_splitter):
        # Setup
        store = DocumentStore()

        # Test storing manuscript chunks
        result = store.store_manuscript_chunks("test_id", "Test Title", "Test Content")

        # Verify text splitter was called with correct Document
        mock_text_splitter.split_documents.assert_called_once()
        doc_arg = mock_text_splitter.split_documents.call_args[0][0][0]
        assert doc_arg.page_content == "Test Content"
        assert doc_arg.metadata["manuscript_id"] == "test_id"
        assert doc_arg.metadata["title"] == "Test Title"

        # Verify chunks were stored with embeddings
        mock_mongodb_store.store_documents_with_embeddings.assert_called_once_with(
            "manuscripts",
            [
                Document(page_content="Chunk 1", metadata={}),
                Document(page_content="Chunk 2", metadata={}),
            ],
        )

        # Verify result
        assert result == ["doc1", "doc2"]

    def test_get_manuscript_relevant_parts(self, mock_mongodb_store):
        # Setup
        store = DocumentStore()

        # Mock similarity search results
        doc1 = Document(
            page_content="Relevant part 1", metadata={"manuscript_id": "test_id"}
        )
        doc2 = Document(
            page_content="Relevant part 2", metadata={"manuscript_id": "test_id"}
        )
        mock_mongodb_store.similarity_search.return_value = [doc1, doc2]

        # Test getting relevant parts
        result = store.get_manuscript_relevant_parts("test_id", "test query")

        # Verify similarity search was called
        mock_mongodb_store.similarity_search.assert_called_once_with(
            "manuscripts", "test query", k=5
        )

        # Verify result
        assert len(result) == 2
        assert result[0].page_content == "Relevant part 1"
        assert result[1].page_content == "Relevant part 2"

        # Test with no results from similarity search
        mock_mongodb_store.similarity_search.return_value = []
        mock_mongodb_store.query_documents.return_value = [
            {"content": "Full content", "title": "Test"}
        ]

        # Mock _simple_chunk_text
        with patch.object(store, "_simple_chunk_text") as mock_chunk:
            mock_chunk.return_value = ["Chunk 1", "Chunk 2"]

            result = store.get_manuscript_relevant_parts("test_id", "test query")

            # Verify fallback to query_documents
            mock_mongodb_store.query_documents.assert_called_once()

            # Verify _simple_chunk_text was called
            mock_chunk.assert_called_once_with("Full content", 1000, 100)

            # Verify result
            assert len(result) == 2
            assert result[0].page_content == "Chunk 1"
            assert result[1].page_content == "Chunk 2"

    def test_simple_chunk_text(self, mock_mongodb_store):
        # Setup
        store = DocumentStore()

        # Test with text shorter than chunk size
        result = store._simple_chunk_text("Short text", 100, 10)
        assert len(result) == 1
        assert result[0] == "Short text"

        # Test with text longer than chunk size
        text = "This is a longer text. It contains multiple sentences. " * 10
        result = store._simple_chunk_text(text, 100, 20)

        # Verify chunks were created
        assert len(result) > 1

        # Verify each chunk is around the expected size
        for chunk in result:
            assert len(chunk) <= 120  # chunk_size + some leeway

        # Verify no content is lost (accounting for overlap)
        combined = "".join(result)
        # The combined text will be longer due to overlap
        assert text in combined
