import pytest
from unittest.mock import patch, Mock
from storybook.utils.chat_model import load_chat_model

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

def test_load_openai_model(mock_env_vars):
    model = load_chat_model("gpt-4")
    assert model.model_name == "gpt-4"

def test_load_anthropic_model(mock_env_vars):
    model = load_chat_model("claude-3-opus", {"provider": "anthropic"})
    assert model.model_name == "claude-3-opus"

@patch("langchain_aws.ChatBedrockConverse")
def test_load_bedrock_model(mock_bedrock, mock_env_vars):
    config = {"provider": "bedrock"}
    model = load_chat_model("anthropic.claude-3", config)
    mock_bedrock.assert_called_once()

def test_missing_api_key():
    with pytest.raises(ValueError):
        load_chat_model("gpt-4", {"api_key": None})