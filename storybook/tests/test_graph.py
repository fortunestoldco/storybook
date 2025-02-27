import pytest
from unittest.mock import patch, MagicMock

from storybook.graph import (
    start_workflow, conduct_market_research, analyze_manuscript, initialize_graph,
    develop_characters, enhance_dialogue, build_world, integrate_subplots,
    evaluate_story_arcs, check_continuity, polish_language, review_quality,
    finalize, build_storybook
)
from storybook.graph import NovelGraphState

class TestGraph:
    
    @pytest.fixture
    def mock_document_store(self):
        with patch('storybook.graph.DocumentStore') as mock_store:
            # Create mock store instance
            mock_store_instance = MagicMock()
            mock_store.return_value = mock_store_instance
            
            # Setup mock manuscript data
            mock_manuscript = {"title": "Test Manuscript", "content": "Test Content"}
            mock_store_instance.get_manuscript.return_value = mock_manuscript
            
            yield mock_store_instance
    
    @pytest.fixture
    def mock_agents(self):
        # Create patches for all agents
        with patch('storybook.graph.MarketResearcher') as mock_researcher, \
             patch('storybook.graph.ContentAnalyzer') as mock_analyzer, \
             patch('storybook.graph.CharacterDeveloper') as mock_character_dev, \
             patch('storybook.graph.DialogueEnhancer') as mock_dialogue, \
             patch('storybook.graph.WorldBuilder') as mock_world, \
             patch('storybook.graph.SubplotWeaver') as mock_subplot, \
             patch('storybook.graph.StoryArcAnalyst') as mock_story_arc, \
             patch('storybook.graph.ContinuityEditor') as mock_continuity, \
             patch('storybook.graph.LanguagePolisher') as mock_language, \
             patch('storybook.graph.QualityReviewer') as mock_quality:
            
            # Setup agent instances
            mock_researcher_instance = MagicMock()
            mock_researcher.return_value = mock_researcher_instance
            mock_researcher_instance.research_market.return_value = {
                "research_insights": {"key": "value"},
                "target_audience": {"demographic": "test"}
            }
            
            mock_analyzer_instance = MagicMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_content.return_value = {
                "analysis": {"sentiment": {}, "readability": {}, "content_structure": {}, "genre_match": {}}
            }
            mock_analyzer_instance.analyze_progress.return_value = {"progress": "updated"}
            
            mock_character_dev_instance = MagicMock()
            mock_character_dev.return_value = mock_character_dev_instance
            mock_character_dev_instance.enhance_characters.return_value = {
                "characters": [{"name": "Character 1"}, {"name": "Character 2"}]
            }
            
            mock_dialogue_instance = MagicMock()
            mock_dialogue.return_value = mock_dialogue_instance
            mock_dialogue_instance.enhance_dialogue.return_value = {
                "message": "Enhanced dialogue"
            }
            
            mock_world_instance = MagicMock()
            mock_world.return_value = mock_world_instance
            mock_world_instance.build_world.return_value = {
                "settings": [{"name": "Setting 1"}, {"name": "Setting 2"}]
            }
            
            mock_subplot_instance = MagicMock()
            mock_subplot.return_value = mock_subplot_instance
            mock_subplot_instance.weave_subplots.return_value = {
                "developed_subplots": [{"title": "Subplot 1"}, {"title": "Subplot 2"}],
                "message": "Integrated subplots"
            }
            
            mock_story_arc_instance = MagicMock()
            mock_story_arc.return_value = mock_story_arc_instance
            mock_story_arc_instance.refine_story_arcs.return_value = {
                "analysis": {"structure": {}, "character_arcs": []},
                "message": "Refined story arcs"
            }
            
            mock_continuity_instance = MagicMock()
            mock_continuity.return_value = mock_continuity_instance
            mock_continuity_instance.check_and_fix_continuity.return_value = {
                "issues": [{"issue": "1"}, {"issue": "2"}],
                "message": "Fixed continuity issues"
            }
            
            mock_language_instance = MagicMock()
            mock_language.return_value = mock_language_instance
            mock_language_instance.polish_language.return_value = {
                "style_analysis": {"tone": "formal"},
                "message": "Polished language"
            }
            
            mock_quality_instance = MagicMock()
            mock_quality.return_value = mock_quality_instance
            mock_quality_instance.finalize_manuscript.return_value = {
                "review": {"rating": 8},
                "improvements": [],
                "final_report": "Final report",
                "message": "Reviewed quality"
            }
            
            yield {
                'researcher': mock_researcher_instance,
                'analyzer': mock_analyzer_instance,
                'character_dev': mock_character_dev_instance,
                'dialogue': mock_dialogue_instance,
                'world': mock_world_instance,
                'subplot': mock_subplot_instance,
                'story_arc': mock_story_arc_instance,
                'continuity': mock_continuity_instance,
                'language': mock_language_instance,
                'quality': mock_quality_instance
            }
    
    def test_start_workflow(self, mock_document_store):
        # Test with valid manuscript
        state = {'manuscript_id': 'test_id', 'title': 'Test Title'}
        result = start_workflow(state)
        
        # Verify document store was called
        mock_document_store.get_manuscript.assert_called_once_with('test_id')
        
        # Verify proper state initialization
        assert result['manuscript_id'] == 'test_id'
        assert result['title'] == 'Test Title'
        assert result['current_state'] == 'research'
        assert isinstance(result, dict)
        
        # Test with missing manuscript_id
        state = {'title': 'Test Title'}
        result = start_workflow(state)
        
        # Verify error state
        assert result['current_state'] == 'END'
        assert 'Error' in result['message']
        
        # Test with non-existent manuscript
        mock_document_store.get_manuscript.return_value = None
        state = {'manuscript_id': 'nonexistent_id'}
        result = start_workflow(state)
        
        # Verify error state
        assert result['current_state'] == 'END'
        assert 'Error' in result['message']
    
    def test_conduct_market_research(self, mock_agents):
        # Setup initial state
        state = {
            'manuscript_id': 'test_id',
            'title': 'Test Title'
        }
        
        # Execute function
        result = conduct_market_research(state)
        
        # Verify researcher was called
        mock_agents['researcher'].research_market.assert_called_once_with('test_id', 'Test Title')
        
        # Verify state updates
        assert result['research_insights'] == {'key': 'value'}
        assert result['target_audience'] == {'demographic': 'test'}
        assert result['current_state'] == 'analysis'
        assert result['stage_progress']['research'] == 1.0
    
    def test_analyze_manuscript(self, mock_agents):
        # Setup initial state
        state = {
            'manuscript_id': 'test_id',
            'title': 'Test Title'
        }
        
        # Execute function
        result = analyze_manuscript(state)
        
        # Verify analyzer was called
        mock_agents['analyzer'].analyze_content.assert_called_once_with('test_id')
        
        # Verify state updates
        assert 'sentiment' in result['analysis_results']
        assert 'readability' in result['analysis_results']
        assert 'content_structure' in result['analysis_results']
        assert 'genre_match' in result['analysis_results']
        assert result['current_state'] == 'initialize'
        assert result['stage_progress']['analysis'] == 1.0
    
    def test_workflow_pipeline(self, mock_agents, mock_document_store):
        # Test the entire workflow pipeline using mock agents
        
        # Start with initial state
        state = {'manuscript_id': 'test_id', 'title': 'Test Title'}
        
        # Execute each step in the workflow
        result = start_workflow(state)
        result = conduct_market_research(result)
        result = analyze_manuscript(result)
        result = initialize_graph(result)
        result = develop_characters(result)
        result = enhance_dialogue(result)
        result = build_world(result)
        result = integrate_subplots(result)
        result = evaluate_story_arcs(result)
        result = check_continuity(result)
        result = polish_language(result)
        result = review_quality(result)
        final_result = finalize(result)
        
        # Verify each agent was called
        mock_agents['researcher'].research_market.assert_called_once()
        mock_agents['analyzer'].analyze_content.assert_called_once()
        mock_agents['character_dev'].enhance_characters.assert_called_once()
        mock_agents['dialogue'].enhance_dialogue.assert_called_once()
        mock_agents['world'].build_world.assert_called_once()
        mock_agents['subplot'].weave_subplots.assert_called_once()
        mock_agents['story_arc'].refine_story_arcs.assert_called_once()
        mock_agents['continuity'].check_and_fix_continuity.assert_called_once()
        mock_agents['language'].polish_language.assert_called_once()
        mock_agents['quality'].finalize_manuscript.assert_called_once()
        
        # Verify final result
        assert final_result['manuscript_id'] == 'test_id'
        assert final_result['status'] == 'complete'
        assert final_result['final_report'] == 'Final report'
        assert 'research_insights' in final_result
        assert 'analysis_results' in final_result
        assert 'improvement_metrics' in final_result
    
    def test_build_storybook(self):
        # Test the graph building function
        graph = build_storybook()
        
        # Verify the graph is properly built
        assert graph is not None
        
        # Verify the graph has the proper entry point
        nodes = graph._nodes
        assert 'START' in nodes
        assert callable(nodes['START']._func)
        
        # Verify key nodes exist
        essential_nodes = [
            'START', 'research', 'analysis', 'initialize', 'character_development',
            'dialogue_enhancement', 'world_building', 'subplot_integration',
            'story_arc_evaluation', 'continuity_check', 'language_polishing',
            'quality_review', 'finalize', 'END'
        ]
        
        for node in essential_nodes:
            assert node in nodes
