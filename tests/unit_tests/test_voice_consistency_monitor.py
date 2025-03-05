import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from storybook.agents.creation.voice_consistency_monitor import VoiceConsistencyMonitor
from storybook.state import NovelSystemState, Project

@pytest.fixture
def mock_project():
    """Create a mock project for testing."""
    return Project(
        title="Test Novel",
        content={
            "style_guide": {"tone": "formal"},
            "tone_preferences": {"mood": "dark"},
            "narrative_voice": {"pov": "third_person"}
        },
        quality_assessment={}
    )

@pytest.fixture
def mock_state(mock_project):
    """Create a mock state for testing."""
    return NovelSystemState(
        phase="creation",
        project=mock_project,
        current_input={
            "task": {
                "type": "style_check",
                "section_id": "chapter_1"
            }
        },
        phase_history={}
    )

@pytest.mark.asyncio
async def test_style_consistency_check(mock_state):
    """Test style consistency checking."""
    monitor = VoiceConsistencyMonitor()
    result = await monitor.process(mock_state, {})
    
    assert "messages" in result
    assert "voice_updates" in result
    assert "style_check" in result["voice_updates"]

@pytest.mark.asyncio
async def test_tone_analysis(mock_state):
    """Test tone analysis."""
    mock_state.current_input["task"]["type"] = "tone_check"
    monitor = VoiceConsistencyMonitor()
    result = await monitor.process(mock_state, {})
    
    assert "messages" in result
    assert "voice_updates" in result
    assert "tone_analysis" in result["voice_updates"]

@pytest.mark.asyncio
async def test_default_voice_check(mock_state):
    """Test default narrative voice check."""
    mock_state.current_input["task"]["type"] = "unknown"
    monitor = VoiceConsistencyMonitor()
    result = await monitor.process(mock_state, {})
    
    assert "messages" in result
    assert "voice_updates" in result
    assert "voice_check" in result["voice_updates"]