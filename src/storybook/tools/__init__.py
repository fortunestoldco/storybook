"""Tool registry and initialization."""

from typing import Dict, List
from langchain_core.tools import BaseTool

class ToolRegistry:
    """Registry for managing tools across different agents."""
    
    def __init__(self):
        self._tools: Dict[str, List[BaseTool]] = {}
    
    def register_tool(self, tool: BaseTool, agent_type: str):
        """Register a tool for a specific agent type."""
        if agent_type not in self._tools:
            self._tools[agent_type] = []
        self._tools[agent_type].append(tool)
    
    def get_tools_for_agent(self, agent_type: str) -> List[BaseTool]:
        """Get all tools registered for an agent type."""
        return self._tools.get(agent_type, [])

# Create singleton tool registry
tool_registry = ToolRegistry()

# Import tool modules
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
