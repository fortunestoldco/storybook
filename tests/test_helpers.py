from typing import Dict, Any, Optional
from workflows import ModelProvider
from langchain.schema.runnable import RunnableConfig

def create_test_story_input(
    title: str = "Test Story",
    manuscript: str = "Test manuscript",
    model_provider: ModelProvider = ModelProvider.ANTHROPIC,
    model_name: str = "claude-3-opus-20240229"
) -> Dict[str, Any]:
    """Creates a test story input with default values."""
    return {
        "title": title,
        "manuscript": manuscript,
        "model_provider": model_provider,
        "model_name": model_name
    }

def create_test_config(
    project_id: str = "test_story_123",
    agent_factory: Optional[Any] = None
) -> RunnableConfig:
    """Creates a test configuration for workflow graphs."""
    return RunnableConfig(
        metadata={
            "project_id": project_id,
            "agent_factory": agent_factory,
            "graph_id": "storybook"
        }
    )