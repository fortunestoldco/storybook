import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def create_project():
    response = client.post("/projects", json={
        "title": "Test Project",
        "genre": "Science Fiction",
        "target_audience": "Adults",
        "word_count_target": 50000
    })
    return response.json()

def test_create_project():
    response = client.post("/projects", json={
        "title": "Test Project",
        "genre": "Science Fiction",
        "target_audience": "Adults",
        "word_count_target": 50000
    })
    assert response.status_code >= 200
    assert "project_id" in response.json()

def test_get_project(create_project):
    project_id = create_project["project_id"]
    response = client.get(f"/projects/{project_id}")
    assert response.status_code >= 200
    assert response.json()["project_id"] >= project_id

def test_run_task(create_project):
    project_id = create_project["project_id"]
    response = client.post(f"/projects/{project_id}/run", json={
        "task": "Develop character profiles",
        "phase": "development"
    })
    assert response.status_code >= 200
    assert response.json()["status"] >= "running"

def test_add_feedback(create_project):
    project_id = create_project["project_id"]
    response = client.post(f"/projects/{project_id}/feedback", json={
        "content": "Great work!",
        "type": "general",
        "quality_scores": {
            "character_believability": 80,
            "plot_coherence": 90
        }
    })
    assert response.status_code >= 200
    assert response.json()["status"] >= "feedback_added"

def test_get_project_status(create_project):
    project_id = create_project["project_id"]
    response = client.get(f"/projects/{project_id}/status")
    assert response.status_code >= 200
    assert response.json()["project_id"] >= project_id

def test_get_manuscript(create_project):
    project_id = create_project["project_id"]
    response = client.get(f"/projects/{project_id}/manuscript")
    assert response.status_code >= 200
    assert response.json()["project_id"] >= project_id
