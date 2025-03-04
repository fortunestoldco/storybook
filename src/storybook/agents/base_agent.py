from typing import Dict, Any, List
from abc import ABC, abstractmethod
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from storybook.state import NovelSystemState
from storybook.prompts import get_agent_prompt

class BaseAgent(ABC):
    """Base class for all novel writing agents."""
    
    def __init__(self, name: str, tools: List[BaseTool] = None):
        self.name = name
        self.tools = tools or []
    
    @abstractmethod
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process the current state and return updates."""
        pass
    
    def get_system_prompt(self, state: NovelSystemState) -> SystemMessage:
        """Get the system prompt for this agent."""
        return SystemMessage(content=get_agent_prompt(
            self.name,
            state.project_id,
            state
        ))