import pytest
from tools.finalization import assess_market_readiness, perform_quality_check, optimize_marketability
from pydantic import ValidationError

def test_assess_market_readiness():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "market_requirements": ["Target audience", "Market fit"],
        "quality_targets": ["Readiness score", "Unique selling points"]
    }
    result = assess_market_readiness(input_data)
    assert "market_assessment" in result
    assert "recommendations" in result
    assert result["market_assessment"]["readiness_score"] == 0.88
    assert result["market_assessment"]["target_audience"] == "Clearly defined"

def test_perform_quality_check():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "market_requirements": ["Technical quality", "Narrative cohesion"],
        "quality_targets": ["Final score", "Certification"]
    }
    result = perform_quality_check(input_data)
    assert "quality_check" in result
    assert "final_score" in result
    assert "certification" in result
    assert result["quality_check"]["technical_quality"] == "Excellent"
    assert result["final_score"] == 0.92

def test_optimize_marketability():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "market_requirements": ["Title strength", "Hook effectiveness"],
        "quality_targets": ["Genre alignment", "Market elements"]
    }
    result = optimize_marketability(input_data)
    assert "optimization" in result
    assert "suggestions" in result
    assert result["optimization"]["title_strength"] == "High impact"
    assert result["optimization"]["hook_effectiveness"] == "Compelling"

def test_assess_market_readiness_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "market_requirements": ["Target audience", "Market fit"],
        "quality_targets": ["Readiness score", "Unique selling points"]
    }
    with pytest.raises(ValidationError):
        assess_market_readiness(input_data)

def test_perform_quality_check_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "market_requirements": ["Technical quality", "Narrative cohesion"],
        "quality_targets": ["Final score", "Certification"]
    }
    with pytest.raises(ValidationError):
        perform_quality_check(input_data)

def test_optimize_marketability_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "market_requirements": ["Title strength", "Hook effectiveness"],
        "quality_targets": ["Genre alignment", "Market elements"]
    }
    with pytest.raises(ValidationError):
        optimize_marketability(input_data)
