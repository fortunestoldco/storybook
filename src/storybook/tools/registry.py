"""Tool registry for the storybook system."""

from typing import Dict, List, Type
from langchain_core.tools import BaseTool


class ToolRegistry:
    """Registry for tools used by storybook agents."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools_by_agent: Dict[str, List[BaseTool]] = {}
        
    def register_tool(self, tool: BaseTool, agent_name: str):
        """Register a tool for use by a specific agent.
        
        Args:
            tool: The tool to register.
            agent_name: Name of the agent that can use this tool.
        """
        if agent_name not in self.tools_by_agent:
            self.tools_by_agent[agent_name] = []
            
        self.tools_by_agent[agent_name].append(tool)
    
    def get_tools_for_agent(self, agent_name: str) -> List[BaseTool]:
        """Get all tools available to a specific agent.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            List of tools available to the agent.
        """
        return self.tools_by_agent.get(agent_name, [])
