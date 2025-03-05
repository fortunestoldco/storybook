from typing import Dict, Any
from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

class NovelWritingTool(ABC, Runnable):
    """Base class for all novel writing tools."""
    
    name: str
    description: str
    
    @abstractmethod
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        """Run the tool asynchronously."""
        raise NotImplementedError()
        
    def validate_input(self, **kwargs) -> bool:
        """Validate tool inputs."""
        return True
        
    async def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle tool execution errors."""
        return {
            "error": str(error),
            "status": "failed",
            "recommendations": []
        }