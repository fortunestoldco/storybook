import pytest
from tools.creation import generate_content, review_content, manage_continuity
from pydantic import ValidationError

def test_generate_content():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "section": "Introduction",
        "requirements": ["Character introduction", "Setting description"]
    }
    result = generate_content(input_data)
    assert "content" in result
    assert "metadata" in result
    assert result["content"]["section"] == "Introduction"
    assert result["content"]["word_count"] == 500

def test_review_content():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "section": "Introduction",
        "requirements": ["Character introduction", "Setting description"]
    }
    result = review_content(input_data)
    assert "review" in result
    assert result["review"]["quality_score"] == 0.85
    assert result["review"]["consistency"] == "High"

def test_manage_continuity():
    input_data = {
        "title": "Test Story",
        "manuscript": "Test manuscript",
        "section": "Introduction",
        "requirements": ["Character introduction", "Setting description"]
    }
    result = manage_continuity(input_data)
    assert "continuity" in result
    assert result["continuity"]["timeline_check"] == "Consistent"
    assert result["continuity"]["character_consistency"] == "Maintained"

def test_generate_content_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "section": "Introduction",
        "requirements": ["Character introduction", "Setting description"]
    }
    with pytest.raises(ValidationError):
        generate_content(input_data)

def test_review_content_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "section": "Introduction",
        "requirements": ["Character introduction", "Setting description"]
    }
    with pytest.raises(ValidationError):
        review_content(input_data)

def test_manage_continuity_invalid_input():
    input_data = {
        "title": "Test Story",
        "manuscript": 123,  # Invalid type
        "section": "Introduction",
        "requirements": ["Character introduction", "Setting description"]
    }
    with pytest.raises(ValidationError):
        manage_continuity(input_data)
