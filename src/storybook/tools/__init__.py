"""Tool registry and initialization."""

from typing import Dict, List
from langchain_core.tools import BaseTool
from ..configuration import Configuration
from ..db_config import initialize_config

class ToolRegistry:
    """Registry for managing tools across different agents."""
    
    def __init__(self, config: Configuration = None):
        self._tools: Dict[str, List[BaseTool]] = {}
        if config:
            initialize_config(config)
    
    def register_tool(self, tool: BaseTool, category: str):
        """Register a tool under a category."""
        if category not in self._tools:
            self._tools[category] = {}
        self._tools[category][tool.name] = tool
    
    def get_tools(self, category: str = None) -> Dict[str, BaseTool]:
        """Get all tools for a category or all tools if no category specified."""
        if category:
            return self._tools.get(category, {})
        return {name: tool for cat in self._tools.values() 
                for name, tool in cat.items()}

tool_registry = ToolRegistry()
