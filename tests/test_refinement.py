import pytest
from tools.refinement import edit_content, analyze_story_coherence, verify_story_elements
from pydantic import ValidationError

def test_edit_content():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "revision_depth": "detailed"
    }
    result = edit_content(input_data)
    assert "edits" in result
    assert "coherence" in result
    assert result["edits"]["section"] == "detailed"
    assert result["edits"]["word_count"] == 450

def test_analyze_story_coherence():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "revision_depth": "detailed"
    }
    result = analyze_story_coherence(input_data)
    assert "coherence_analysis" in result
    assert "feedback" in result
    assert result["coherence_analysis"]["overall_score"] == 0.87
    assert result["feedback"][0] == "Consider revising pacing in middle chapters"

def test_verify_story_elements():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "focus_areas": ["technical", "consistency"]
    }
    result = verify_story_elements(input_data)
    assert "verification" in result
    assert "feedback" in result
    assert result["verification"]["elements_checked"][0] == "Character consistency"
    assert result["verification"]["issues_found"][0] == "Inconsistent character traits in chapter 3"

def test_edit_content_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "revision_depth": "detailed"
    }
    with pytest.raises(ValidationError):
        edit_content(input_data)

def test_analyze_story_coherence_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "revision_depth": "detailed"
    }
    with pytest.raises(ValidationError):
        analyze_story_coherence(input_data)

def test_verify_story_elements_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "focus_areas": ["technical", "consistency"]
    }
    with pytest.raises(ValidationError):
        verify_story_elements(input_data)
