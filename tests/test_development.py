import pytest
from tools.development import develop_plot_structure, develop_characters, develop_world_building
from pydantic import ValidationError

def test_develop_plot_structure():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "phase": "plot_development",
        "elements": ["exposition", "rising_action", "climax", "falling_action", "resolution"]
    }
    result = develop_plot_structure(input_data)
    assert "plot_elements" in result
    assert "completion_score" in result
    assert result["plot_elements"]["exposition"] == "Initial story setup and character introduction"
    assert result["completion_score"] == 0.85

def test_develop_characters():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "phase": "character_development",
        "elements": ["protagonist", "antagonist"]
    }
    result = develop_characters(input_data)
    assert "characters" in result
    assert "completion_score" in result
    assert result["characters"]["protagonist"]["arc"] == "Growth and transformation"
    assert result["completion_score"] == 0.90

def test_develop_world_building():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "phase": "world_building",
        "elements": ["setting", "rules", "culture", "history"]
    }
    result = develop_world_building(input_data)
    assert "world_elements" in result
    assert "completion_score" in result
    assert result["world_elements"]["setting"] == "Primary story location and time period"
    assert result["completion_score"] == 0.80

def test_develop_plot_structure_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "phase": "plot_development",
        "elements": ["exposition", "rising_action", "climax", "falling_action", "resolution"]
    }
    with pytest.raises(ValidationError):
        develop_plot_structure(input_data)

def test_develop_characters_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "phase": "character_development",
        "elements": ["protagonist", "antagonist"]
    }
    with pytest.raises(ValidationError):
        develop_characters(input_data)

def test_develop_world_building_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "phase": "world_building",
        "elements": ["setting", "rules", "culture", "history"]
    }
    with pytest.raises(ValidationError):
        develop_world_building(input_data)
