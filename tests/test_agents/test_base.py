import pytest
from storybook.agents.base import BaseAgent
from storybook.config import LLMProvider

def test_base_agent_llm_config():
    """Test BaseAgent LLM configuration handling."""
    
    # Test default initialization
    agent = BaseAgent()
    assert agent.llm is not None

    # Test OpenAI configuration
    openai_config = {
        "provider": LLMProvider.OPENAI,
        "config": {
            "model_name": "gpt-4",
            "temperature": 0.7
        }
    }
    agent = BaseAgent(llm_config=openai_config)
    assert agent.llm is not None

    # Test Anthropic configuration
    anthropic_config = {
        "provider": LLMProvider.ANTHROPIC,
        "config": {
            "model_name": "claude-3-sonnet",
            "temperature": 0.7
        }
    }
    agent = BaseAgent(llm_config=anthropic_config)
    assert agent.llm is not None

    # Test invalid configuration
    with pytest.raises(ValueError):
        BaseAgent(llm_config={"provider": "invalid"})