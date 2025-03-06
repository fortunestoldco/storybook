import pytest
from storybook.graph import create_phase_graph
from storybook.configuration import Configuration

@pytest.fixture
def config():
    return Configuration()

@pytest.mark.integration
async def test_initialization_phase(config):
    # Arrange
    graph = create_phase_graph("initialization", "test_project", config)
    
    # Act
    result = await graph.ainvoke({
        "input": "Initialize new project",
        "task": "project_setup"
    })
    
    # Assert
    assert isinstance(result, dict)
    assert "output" in result

@pytest.mark.integration
async def test_development_phase(config):
    # Arrange
    graph = create_phase_graph("development", "test_project", config)
    
    # Act
    result = await graph.ainvoke({
        "input": "Develop story outline",
        "task": "outline_creation"
    })
    
    # Assert
    assert isinstance(result, dict)
    assert "output" in result