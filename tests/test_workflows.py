import unittest
from unittest.mock import patch, MagicMock
from workflows import (
    create_initialization_graph,
    create_development_graph,
    create_creation_graph,
    create_refinement_graph,
    create_finalization_graph,
    create_storybook_workflow,
    ModelProvider,
    StoryState
)
from langchain.schema.runnable import RunnableConfig
from agents import AgentFactory
from mongodb import MongoDBManager
from test_helpers import create_test_story_input, create_test_config
import pytest
from typing import Dict, Any

@pytest.fixture
def test_input() -> Dict[str, Any]:
    return create_test_story_input()

@pytest.fixture
def config(agent_factory):
    return create_test_config()

@pytest.fixture
def agent_factory(mongodb_manager):
    return AgentFactory(mongodb_manager)

@pytest.fixture
def mongodb_manager():
    return MongoDBManager()

def test_create_initialization_graph(config, agent_factory):
    """Test initialization graph creation."""
    graph = create_initialization_graph(config)
    
    assert graph is not None
    assert "executive_director" in graph.nodes
    assert "human_feedback_manager" in graph.nodes
    assert "quality_assessment_director" in graph.nodes
    assert "project_timeline_manager" in graph.nodes
    assert "market_alignment_director" in graph.nodes

def test_create_development_graph(config, agent_factory):
    """Test development graph creation."""
    graph = create_development_graph(config)
    
    assert graph is not None
    assert "executive_director" in graph.nodes
    assert "creative_director" in graph.nodes
    assert "structure_architect" in graph.nodes
    assert all(node in graph.nodes for node in [
        "plot_development_specialist",
        "world_building_expert",
        "character_psychology_specialist",
        "character_voice_designer",
        "character_relationship_mapper",
        "domain_knowledge_specialist",
        "cultural_authenticity_expert",
        "market_alignment_director"
    ])

def test_workflow_execution(test_input, config):
    """Test complete workflow execution."""
    workflow = create_storybook_workflow(config)
    result = workflow.invoke(test_input)
    
    # Verify workflow completion
    assert result["title"] == test_input["title"]
    assert result["model_provider"] == test_input["model_provider"]
    assert "feedback" in result
    
    # Verify phase completion
    for phase in ["initialization", "development", "creation", "refinement", "finalization"]:
        assert f"{phase}_complete" in result
    
    # Verify quality metrics
    assert "quality_score" in result
    assert 0.0 <= result["quality_score"] <= 1.0

def test_workflow_graph_structure(config):
    """Test the structure of all workflow graphs."""
    graphs = {
        "initialization": create_initialization_graph(config),
        "development": create_development_graph(config),
        "creation": create_creation_graph(config),
        "refinement": create_refinement_graph(config),
        "finalization": create_finalization_graph(config)
    }
    
    # Test common nodes across all graphs
    for name, graph in graphs.items():
        assert "executive_director" in graph.nodes
        assert graph is not None

    # Test phase-specific nodes
    assert "creative_director" in graphs["development"].nodes
    assert "content_development_director" in graphs["creation"].nodes
    assert "editorial_director" in graphs["refinement"].nodes
    assert "market_alignment_director" in graphs["finalization"].nodes

if __name__ == '__main__':
    pytest.main(['-v', __file__])
