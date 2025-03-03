import pytest
from tools.quality_assessment import analyze_story_structure, evaluate_character_arcs, assess_narrative_coherence
from tools.analysis import StoryAnalysisInput

@pytest.fixture
def story_analysis_input():
    return StoryAnalysisInput(
        title="Test Story",
        manuscript="This is a test manuscript.",
        criteria=["structure", "characters", "coherence"]
    )

def test_analyze_story_structure(story_analysis_input):
    result = analyze_story_structure(story_analysis_input)
    assert "structure_score" in result
    assert "findings" in result
    assert "suggestions" in result
    assert result["structure_score"] >= 0.0 and result["structure_score"] <= 1.0

def test_evaluate_character_arcs(story_analysis_input):
    result = evaluate_character_arcs(story_analysis_input)
    assert "character_score" in result
    assert "findings" in result
    assert "suggestions" in result
    assert result["character_score"] >= 0.0 and result["character_score"] <= 1.0

def test_assess_narrative_coherence(story_analysis_input):
    result = assess_narrative_coherence(story_analysis_input)
    assert "coherence_score" in result
    assert "findings" in result
    assert "suggestions" in result
    assert result["coherence_score"] >= 0.0 and result["coherence_score"] <= 1.0
