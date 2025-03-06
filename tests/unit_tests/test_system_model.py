import pytest
from datetime import datetime
from storybook.models.system import NovelSystemState

@pytest.fixture
def system_state():
    return NovelSystemState(
        project_id="test_project",
        phase="development"
    )

def test_system_state_initialization(system_state):
    assert system_state.project_id == "test_project"
    assert system_state.phase == "development"
    assert system_state.status == "active"
    assert system_state.current_agent is None
    assert isinstance(system_state.created_at, datetime)

def test_update_state(system_state):
    result = {
        "metadata_updates": {"word_count": 1000}
    }
    
    system_state.update_state(
        agent="test_agent",
        action="test_action",
        result=result
    )
    
    assert system_state.current_agent == "test_agent"
    assert len(system_state.history) == 1
    assert system_state.metadata["word_count"] == 1000

def test_get_agent_history(system_state):
    # Add some history entries
    system_state.update_state("agent1", "action1", {})
    system_state.update_state("agent2", "action2", {})
    system_state.update_state("agent1", "action3", {})
    
    history = system_state.get_agent_history("agent1")
    assert len(history) == 2
    assert all(entry["agent"] == "agent1" for entry in history)

def test_to_dict(system_state):
    state_dict = system_state.to_dict()
    assert state_dict["project_id"] == "test_project"
    assert state_dict["phase"] == "development"
    assert "created_at" in state_dict
    assert "updated_at" in state_dict