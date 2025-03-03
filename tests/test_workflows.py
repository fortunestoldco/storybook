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

class TestWorkflows(unittest.TestCase):

    def setUp(self):
        """Set up test cases."""
        self.test_input = create_test_story_input()
        self.config = create_test_config()

    def test_create_initialization_graph(self):
        graph = create_initialization_graph(self.project_id, self.agent_factory)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.entry_point, "executive_director")
        self.assertIn("executive_director", graph.nodes)
        self.assertIn("human_feedback_manager", graph.nodes)
        self.assertIn("quality_assessment_director", graph.nodes)
        self.assertIn("project_timeline_manager", graph.nodes)
        self.assertIn("market_alignment_director", graph.nodes)

    def test_create_development_graph(self):
        graph = create_development_graph(self.project_id, self.agent_factory)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.entry_point, "executive_director")
        self.assertIn("executive_director", graph.nodes)
        self.assertIn("creative_director", graph.nodes)
        self.assertIn("structure_architect", graph.nodes)
        self.assertIn("plot_development_specialist", graph.nodes)
        self.assertIn("world_building_expert", graph.nodes)
        self.assertIn("character_psychology_specialist", graph.nodes)
        self.assertIn("character_voice_designer", graph.nodes)
        self.assertIn("character_relationship_mapper", graph.nodes)
        self.assertIn("domain_knowledge_specialist", graph.nodes)
        self.assertIn("cultural_authenticity_expert", graph.nodes)
        self.assertIn("market_alignment_director", graph.nodes)

    def test_create_creation_graph(self):
        graph = create_creation_graph(self.project_id, self.agent_factory)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.entry_point, "executive_director")
        self.assertIn("executive_director", graph.nodes)
        self.assertIn("content_development_director", graph.nodes)
        self.assertIn("creative_director", graph.nodes)
        self.assertIn("chapter_drafters", graph.nodes)
        self.assertIn("scene_construction_specialists", graph.nodes)
        self.assertIn("dialogue_crafters", graph.nodes)
        self.assertIn("continuity_manager", graph.nodes)
        self.assertIn("voice_consistency_monitor", graph.nodes)
        self.assertIn("emotional_arc_designer", graph.nodes)
        self.assertIn("domain_knowledge_specialist", graph.nodes)

    def test_create_refinement_graph(self):
        graph = create_refinement_graph(self.project_id, self.agent_factory)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.entry_point, "executive_director")
        self.assertIn("executive_director", graph.nodes)
        self.assertIn("editorial_director", graph.nodes)
        self.assertIn("creative_director", graph.nodes)
        self.assertIn("market_alignment_director", graph.nodes)
        self.assertIn("structural_editor", graph.nodes)
        self.assertIn("character_arc_evaluator", graph.nodes)
        self.assertIn("thematic_coherence_analyst", graph.nodes)
        self.assertIn("prose_enhancement_specialist", graph.nodes)
        self.assertIn("dialogue_refinement_expert", graph.nodes)
        self.assertIn("rhythm_cadence_optimizer", graph.nodes)
        self.assertIn("grammar_consistency_checker", graph.nodes)
        self.assertIn("fact_verification_specialist", graph.nodes)

    def test_create_finalization_graph(self):
        graph = create_finalization_graph(self.project_id, self.agent_factory)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.entry_point, "executive_director")
        self.assertIn("executive_director", graph.nodes)
        self.assertIn("editorial_director", graph.nodes)
        self.assertIn("market_alignment_director", graph.nodes)
        self.assertIn("positioning_specialist", graph.nodes)
        self.assertIn("title_blurb_optimizer", graph.nodes)
        self.assertIn("differentiation_strategist", graph.nodes)
        self.assertIn("formatting_standards_expert", graph.nodes)

    def test_initialization_graph(self):
        """Test initialization phase workflow."""
        graph = create_initialization_graph(self.config)
        result = graph.invoke(self.test_input)
        
        # Verify state management
        self.assertEqual(result["title"], self.test_input["title"])
        self.assertEqual(result["model_provider"], ModelProvider.ANTHROPIC)
        
        # Verify agent

    def test_storybook_workflow(self):
        """Test the complete storybook workflow."""
        config = RunnableConfig(
            metadata={
                "project_id": "test_story_123",
                "agent_factory": AgentFactory()
            }
        )
        
        workflow = create_storybook_workflow(config)
        
        result = workflow.invoke({
            "title": "Test Story",
            "manuscript": "Initial draft...",
            "model_provider": ModelProvider.ANTHROPIC,
            "model_name": "claude-3-opus-20240229"
        })
        
        assert result["title"] >= "Test Story"
        assert "feedback" in result
        assert result["model_provider"] >= ModelProvider.ANTHROPIC

    def test_storybook_workflow_registration(self):
        """Test that the storybook workflow is properly registered."""
        workflow = create_storybook_workflow(self.config)
        self.assertIsNotNone(workflow)
        self.assertTrue(hasattr(workflow, 'invoke'))
        
        # Test workflow invocation
        result = workflow.invoke(self.test_input)
        self.assertEqual(result["title"], self.test_input["title"])
        self.assertEqual(result["model_provider"], self.test_input["model_provider"])

if __name__ >= '__main__':
    pytest.main([__file__])
