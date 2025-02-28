import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from storybook.db.mongodb_client import MongoDBStore

class TestMongoDBStore:
    @pytest.fixture
    def mock_client(self):
        with patch('storybook.db.mongodb_client.MongoClient') as mock_client:
            # Mock the MongoDB client
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            
            # Mock the database
            mock_db = MagicMock()
            mock_client_instance.__getitem__.return_value = mock_db
            
            # Mock the collections
            mock_collection = MagicMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_db.list_collection_names.return_value = []
            
            yield mock_client_instance
    
    @pytest.fixture
    def mock_vector_store(self):
        with patch('langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch') as mock_store:re:
            mock_store_instance = MagicMock()agicMock()
            mock_store.from_connection_string.return_value = mock_store_instance.return_value = mock_store_instance
            yield mock_store_instance
    
    @pytest.fixture
    def mock_embeddings(self):def mock_embeddings(self):
        with patch('storybook.db.mongodb_client.OpenAIEmbeddings') as mock_embeddings:'storybook.db.mongodb_client.OpenAIEmbeddings') as mock_embeddings:
            # Mock the embeddingsngs
            mock_embeddings_instance = MagicMock()
            mock_embeddings.return_value = mock_embeddings_instancen_value = mock_embeddings_instance
            
            yield mock_embeddings_instance
    
    def test_init(self, mock_client, mock_vector_store, mock_embeddings):vector_store, mock_embeddings):
        # Test the initialization of MongoDBStore    # Test the initialization of MongoDBStore
        store = MongoDBStore()
        
        # Verify that MongoDB client was initializedclient was initialized
        assert store.client == mock_clientassert store.client == mock_client
        
        # Verify that vector store was initializedtialized
        assert store.vector_store == mock_vector_storeassert store.vector_store == mock_vector_store
        
        # Verify that embeddings were initialized
        assert store.embeddings == mock_embeddingsassert store.embeddings == mock_embeddings
        
        # Verify that collections were checked
        store.client.__getitem__.return_value.list_collection_names.assert_called_once()store.client.__getitem__.return_value.list_collection_names.assert_called_once()
    
    def test_store_document(self, mock_client, mock_vector_store, mock_embeddings):
        # Setup    # Setup
        store = MongoDBStore()
        mock_collection = store.client.__getitem__.return_value.__getitem__.return_valuellection = store.client.__getitem__.return_value.__getitem__.return_value
        mock_collection.insert_one.return_value.inserted_id = "test_id"_one.return_value.inserted_id = "test_id"
        
        # Execute
        result = store.store_document("test_collection", {"test": "data"})result = store.store_document("test_collection", {"test": "data"})
        
        # Verify
        mock_collection.insert_one.assert_called_once_with({"test": "data"})mock_collection.insert_one.assert_called_once_with({"test": "data"})
        assert result == "test_id"esult == "test_id"
    
    def test_get_document(self, mock_client, mock_vector_store, mock_embeddings):ck_client, mock_vector_store, mock_embeddings):
        # Setup    # Setup
        store = MongoDBStore()
        mock_collection = store.client.__getitem__.return_value.__getitem__.return_valuellection = store.client.__getitem__.return_value.__getitem__.return_value
        mock_document = {"_id": MagicMock(spec=object), "test": "data"}: MagicMock(spec=object), "test": "data"}
        mock_collection.find_one.return_value = mock_document
        
        # Execute
        result = store.get_document("test_collection", "test_id")result = store.get_document("test_collection", "test_id")
        
        # Verify
        assert result["test"] == "data"assert result["test"] == "data"
        assert "_id" in result_id" in result
    
    def test_store_documents_with_embeddings(self, mock_client, mock_vector_store, mock_embeddings):ith_embeddings(self, mock_client, mock_vector_store, mock_embeddings):
        # Setup    # Setup
        store = MongoDBStore()
        mock_vector_store.add_documents.return_value = ["doc1", "doc2"]ctor_store.add_documents.return_value = ["doc1", "doc2"]
        
        # Create test documents
        docs = [docs = [
            Document(page_content="Test content 1", metadata={"source": "test1"}),nt="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"})ment(page_content="Test content 2", metadata={"source": "test2"})
        ]
        
        # Test with vectors collection Test with vectors collection
        result = store.store_documents_with_embeddings("vectors", docs)result = store.store_documents_with_embeddings("vectors", docs)
        
        # Verify vector store was used
        mock_vector_store.add_documents.assert_called_once_with(docs)mock_vector_store.add_documents.assert_called_once_with(docs)
        assert result == ["doc1", "doc2"]2"]
        
        # Reset mock
        mock_vector_store.add_documents.reset_mock()mock_vector_store.add_documents.reset_mock()
        
        # Test with other collection
        mock_collection = store.client.__getitem__.return_value.__getitem__.return_valuemock_collection = store.client.__getitem__.return_value.__getitem__.return_value
        mock_collection.insert_one.side_effect = [ide_effect = [
            MagicMock(inserted_id="id1"),
            MagicMock(inserted_id="id2")
        ]
        
        result = store.store_documents_with_embeddings("test_collection", docs)esult = store.store_documents_with_embeddings("test_collection", docs)
        
        # Verify documents were stored in collection
        assert mock_collection.insert_one.call_count == 2assert mock_collection.insert_one.call_count == 2
        
        # Verify documents were also added to vector storee
        mock_vector_store.add_documents.assert_called_once()mock_vector_store.add_documents.assert_called_once()
        
        # Verify result
        assert result == ["id1", "id2"]assert result == ["id1", "id2"]
    
    def test_similarity_search(self, mock_client, mock_vector_store, mock_embeddings):ck_client, mock_vector_store, mock_embeddings):
        # Setup    # Setup
        store = MongoDBStore()
        mock_docs = [Document(page_content="Test result", metadata={})]cs = [Document(page_content="Test result", metadata={})]
        mock_vector_store.similarity_search.return_value = mock_docslarity_search.return_value = mock_docs
        
        # Execute
        result = store.similarity_search("test_collection", "test query")result = store.similarity_search("test_collection", "test query")
        
        # Verify
        mock_vector_store.similarity_search.assert_called_once_with(mock_vector_store.similarity_search.assert_called_once_with(
            "test query", k=5, filter={"metadata.collection": "test_collection"}t query", k=5, filter={"metadata.collection": "test_collection"}
        )
        assert result == mock_docs
        
        # Test similarity search failure with fallbackailure with fallback
        mock_vector_store.similarity_search.side_effect = Exception("Test error")mock_vector_store.similarity_search.side_effect = Exception("Test error")
        mock_collection = store.client.__getitem__.return_value.__getitem__.return_valueurn_value.__getitem__.return_value
        mock_collection.find.return_value = [
            {"content": "fallback result", "metadata": {}, "_id": "test_id"}
        ]
        
        # Execute Execute
        result = store.similarity_search("test_collection", "test query")result = store.similarity_search("test_collection", "test query")
        
        # Verify fallback was used
        assert len(result) == 1assert len(result) == 1
        assert result[0].page_content == "fallback result"ent == "fallback result"
