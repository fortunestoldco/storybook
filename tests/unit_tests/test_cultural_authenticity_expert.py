import pytest
from unittest.mock import Mock
from typing import Dict, Any

from storybook.agents.development.cultural_authenticity_expert import CulturalAuthenticityExpert
from storybook.state import NovelSystemState

@pytest.fixture
def mock_state():
    return NovelSystemState(
        phase="development",
        project=Mock(
            content={
                "cultural_elements": {},
                "representation": {}
            }
        ),
        current_input={
            "task": {
                "type": "authenticity_check",
                "culture": "japanese"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_authenticity_check():
    """Test cultural authenticity checking."""
    expert = CulturalAuthenticityExpert()
    result = await expert.process(mock_state, {})
    
    assert "messages" in result
    assert "cultural_updates" in result
    assert "authenticity" in result["cultural_updates"]

@pytest.mark.asyncio
async def test_representation_analysis():
    """Test representation analysis."""
    mock_state.current_input["task"]["type"] = "representation"
    expert = CulturalAuthenticityExpert()
    result = await expert.process(mock_state, {})
    
    assert "messages" in result
    assert "cultural_updates" in result
    assert "representation" in result["cultural_updates"]