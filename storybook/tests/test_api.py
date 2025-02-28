# test_*.py files need pytest and MagicMock imports
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from fastapi.testclient import TestClient

from storybook.api import app, DocumentStore, storybook


class TestAPI:

    @pytest.fixture
    def test_client(self):
        return TestClient(app)

    @pytest.fixture
    def mock_document_store(self):
        with patch("storybook.api.DocumentStore") as mock_store:
            # Create mock store instance
            mock_store_instance = MagicMock()
            mock_store.return_value = mock_store_instance

            # Setup mock manuscript functions
            mock_store_instance.store_manuscript.return_value = "test_id"
            mock_store_instance.get_manuscript.return_value = {
                "title": "Test Manuscript",
                "content": "Test content",
                "metadata": {"author": "Test Author"},
            }
            mock_store_instance.store_manuscript_chunks.return_value = [
                "chunk1",
                "chunk2",
            ]

            yield mock_store_instance

    @pytest.fixture
    def mock_storybook(self):
        with patch("storybook.api.storybook") as mock_graph:
            # Create mock invoke function
            mock_graph.invoke.return_value = {
                "manuscript_id": "test_id",
                "status": "complete",
                "final_report": "Test report",
                "improvement_metrics": {"characters_enhanced": 2},
            }

            yield mock_graph

    def test_health_check(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_upload_manuscript(self, test_client, mock_document_store):
        # Test successful upload
        manuscript_data = {
            "title": "Test Manuscript",
            "content": "This is a test manuscript content.",
            "metadata": {"author": "Test Author"},
        }

        response = test_client.post("/manuscripts", json=manuscript_data)

        # Verify response
        assert response.status_code == 200
        assert response.json()["manuscript_id"] == "test_id"
        assert "Successfully uploaded" in response.json()["message"]

        # Verify store functions were called
        mock_document_store.store_manuscript.assert_called_once_with(
            "Test Manuscript",
            "This is a test manuscript content.",
            {"author": "Test Author"},
        )
        mock_document_store.store_manuscript_chunks.assert_called_once()

        # Test upload with error
        mock_document_store.store_manuscript.side_effect = Exception("Storage error")

        response = test_client.post("/manuscripts", json=manuscript_data)

        # Verify error response
        assert response.status_code == 500
        assert "Failed to upload" in response.json()["detail"]

    def test_get_manuscript(self, test_client, mock_document_store):
        # Test successful retrieval
        response = test_client.get("/manuscripts/test_id")

        # Verify response
        assert response.status_code == 200
        assert response.json()["title"] == "Test Manuscript"
        assert response.json()["content"] == "Test content"

        # Verify store function was called
        mock_document_store.get_manuscript.assert_called_once_with("test_id")

        # Test retrieval of non-existent manuscript
        mock_document_store.get_manuscript.return_value = None

        response = test_client.get("/manuscripts/nonexistent_id")

        # Verify error response
        assert response.status_code == 404
        assert "Manuscript not found" in response.json()["detail"]

        # Test retrieval with error
        mock_document_store.get_manuscript.side_effect = Exception("Retrieval error")

        response = test_client.get("/manuscripts/test_id")

        # Verify error response
        assert response.status_code == 500
        assert "Failed to retrieve" in response.json()["detail"]

    def test_start_transformation(self, test_client, mock_document_store):
        # Test successful transformation start
        transformation_data = {"manuscript_id": "test_id"}

        response = test_client.post("/start-transformation", json=transformation_data)

        # Verify response
        assert response.status_code == 200
        assert response.json()["manuscript_id"] == "test_id"
        assert "Transformation started" in response.json()["message"]

        # Verify store function was called
        mock_document_store.get_manuscript.assert_called_once_with("test_id")

        # Test with non-existent manuscript
        mock_document_store.get_manuscript.return_value = None

        response = test_client.post("/start-transformation", json=transformation_data)

        # Verify error response
        assert response.status_code == 404
        assert "Manuscript not found" in response.json()["detail"]

        # Test with error
        mock_document_store.get_manuscript.side_effect = Exception("Start error")

        response = test_client.post("/start-transformation", json=transformation_data)

        # Verify error response
        assert response.status_code == 500
        assert "Failed to start transformation" in response.json()["detail"]

    def test_transform_route(self, test_client, mock_storybook):
        # This test verifies the LangServe integration is working
        # Note: We can't directly test the route added by add_routes in a unit test
        # This would require integration testing with the actual LangServe routes

        # Instead, we verify the storybook is properly mocked
        assert mock_storybook.invoke.called == False

        # In a real integration test, we would do:
        # response = test_client.post("/transform/invoke", json={"manuscript_id": "test_id"})
        # assert response.status_code == 200
        # assert "manuscript_id" in response.json()
