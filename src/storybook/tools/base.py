from typing import Dict, Any
from langchain_core.tools import BaseTool
from pydantic import Field

class NovelWritingTool(BaseTool):
    """Base class for all novel writing tools."""
    
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    
    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async implementation of the tool."""
        raise NotImplementedError("Tool must implement async run")
    
    def _run(self, *args, **kwargs) -> Dict[str, Any]:
        """Sync implementation not supported."""
        raise NotImplementedError("Only async operations supported")