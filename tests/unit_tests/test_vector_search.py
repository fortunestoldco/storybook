import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
from storybook.utils.vector_search import VectorSearch

@pytest.fixture
def mock_embeddings():
    embeddings = Mock()
    embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    embeddings.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    return embeddings

@pytest.fixture
async def vector_search(mock_embeddings):
    vs = VectorSearch(mock_embeddings)
    await vs.add_documents(["doc1", "doc2"])
    return vs

@pytest.mark.asyncio
async def test_add_documents(mock_embeddings):
    # Arrange
    vs = VectorSearch(mock_embeddings)
    
    # Act
    await vs.add_documents(["test doc"])
    
    # Assert
    assert len(vs.documents) == 1
    assert len(vs.document_embeddings) == 1
    mock_embeddings.aembed_documents.assert_called_once()

@pytest.mark.asyncio
async def test_similarity_search(vector_search):
    # Act
    results = await vector_search.similarity_search("test query", k=2)
    
    # Assert
    assert len(results) == 2
    assert "content" in results[0]
    assert "similarity" in results[0]
    assert isinstance(results[0]["similarity"], float)

def test_clear(vector_search):
    # Act
    vector_search.clear()
    
    # Assert
    assert len(vector_search.documents) == 0
    assert len(vector_search.document_embeddings) == 0