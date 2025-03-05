from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig

class NovelWritingTool(Runnable):
    """Base class for all novel writing tools."""
    
    name: str
    description: str
    
    async def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Synchronously invoke the tool."""
        try:
            self.validate_input(**input)
            return await self._arun(**input)
        except Exception as e:
            return await self.handle_error(e)
    
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        """Run the tool asynchronously."""
        raise NotImplementedError("Tools must implement _arun")
    
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