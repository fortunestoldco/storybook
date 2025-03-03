import pytest
from tools.project_management import create_project_timeline, analyze_market_trends
from datetime import datetime, timedelta

@pytest.fixture
def project_input():
    return {
        "title": "Test Project",
        "manuscript": "Test manuscript",
        "target_completion": datetime.now() + timedelta(days=30),
        "milestones": ["Development", "Creation", "Refinement"]
    }

def test_create_project_timeline(project_input):
    result = create_project_timeline(project_input)
    assert "timeline" in result
    assert "start_date" in result["timeline"]
    assert "estimated_completion" in result["timeline"]
    assert "phases" in result["timeline"]
    assert len(result["timeline"]["phases"]) == 3

def test_analyze_market_trends(project_input):
    result = analyze_market_trends(project_input)
    assert "market_analysis" in result
    assert "trending_genres" in result["market_analysis"]
    assert "audience_preferences" in result["market_analysis"]
    assert "market_gaps" in result["market_analysis"]
    assert "recommendations" in result["market_analysis"]
