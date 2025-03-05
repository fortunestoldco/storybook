from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.base import NovelWritingTool

class BaseAgent(ABC):
    """Base class for all novel writing agents."""
    
    def __init__(
        self,
        name: str,
        tools: List[NovelWritingTool],
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.tools = tools
        self._system_prompt = system_prompt
        self._validate_tools()
    
    def _validate_tools(self) -> None:
        """Validate that all tools are properly initialized."""
        if not all(isinstance(tool, NovelWritingTool) for tool in self.tools):
            raise ValueError("All tools must inherit from NovelWritingTool")
    
    def get_system_prompt(self, state: NovelSystemState) -> SystemMessage:
        """Get the system prompt for this agent."""
        base_prompt = self._system_prompt or f"You are the {self.name} agent."
        return SystemMessage(content=base_prompt)
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process the current state and return updates."""
        try:
            result = await self._process(state, config)
            return self._format_response(result)
        except Exception as e:
            return await self._handle_error(e)
    
    @abstractmethod
    async def _process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Internal processing logic to be implemented by agents."""
        raise NotImplementedError()
    
    def _format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the agent's response."""
        if "messages" not in result:
            result["messages"] = [AIMessage(content="Task completed")]
        return result
    
    async def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle processing errors."""
        return {
            "messages": [AIMessage(content=f"Error: {str(error)}")],
            "error": str(error),
            "status": "failed"
        }