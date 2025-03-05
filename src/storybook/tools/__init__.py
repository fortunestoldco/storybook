"""Tool registry and initialization."""

from typing import Dict, Any, Callable
from langchain_core.tools import BaseTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, BaseTool]] = {}
    
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

# Import all tool modules to register their tools
from . import management_tools
from . import creative_tools
from . import development_tools
from . import creation_tools
from . import refinement_tools
from . import finalization_tools

# Import additional tool modules
from . import plot_tools
from . import character_tools
from . import research_tools
from . import editorial_tools
from . import timeline_tools
from . import market_tools
from . import formatting_tools

__all__ = ["tool_registry"]
