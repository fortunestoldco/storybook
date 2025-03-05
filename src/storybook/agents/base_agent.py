from typing import Dict, Any, List
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.base import NovelWritingTool

class BaseAgent:
    """Base class for all novel writing agents."""
    
    def __init__(
        self, 
        name: str,
        tools: List[NovelWritingTool]
    ):
        self.name = name
        self.tools = tools
    
    def get_system_prompt(self, state: NovelSystemState) -> SystemMessage:
        """Get the system prompt for this agent."""
        return SystemMessage(content=f"You are the {self.name} agent.")
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process the current state and return updates."""
        raise NotImplementedError("Agents must implement process")