"""Tool registry for the storybook system."""

from typing import Dict, List, Type
from langchain_core.tools import BaseTool

class ProjectTimelineTool(BaseTool):
    name = "Project Timeline Tracker"
    description = "View and adjust project deadlines"

class TeamCommunicationTool(BaseTool):
    name = "Team Communication Hub"
    description = "Send directives to other agents"

class ProgressDashboardTool(BaseTool):
    name = "Progress Dashboard"
    description = "Monitor completion status"

class ResourceAllocationTool(BaseTool):
    name = "Resource Allocation Tool"
    description = "Assign resources to different aspects"

class CreativeVisionTool(BaseTool):
    name = "Creative Vision Board"
    description = "Document and share artistic direction"

class StyleGuideTool(BaseTool):
    name = "Style Guide Creator"
    description = "Establish creative standards"

class InspirationTool(BaseTool):
    name = "Inspiration Repository"
    description = "Store creative references"

class ConceptEvaluationTool(BaseTool):
    name = "Concept Evaluation Matrix"
    description = "Assess creative ideas"

# ... Add similar tool classes for other agents ...

class ToolRegistry:
    """Registry for tools used by storybook agents."""
    
    def __init__(self):
        self.tools_by_agent: Dict[str, List[BaseTool]] = {}
        self._initialize_tools()
        
    def _initialize_tools(self):
        """Initialize default tools for each agent."""
        # Executive Director tools
        exec_tools = [
            ProjectTimelineTool(),
            TeamCommunicationTool(),
            ProgressDashboardTool(),
            ResourceAllocationTool()
        ]
        self.register_tools(exec_tools, "executive_director")
        
        # Creative Director tools
        creative_tools = [
            CreativeVisionTool(),
            StyleGuideTool(),
            InspirationTool(),
            ConceptEvaluationTool()
        ]
        self.register_tools(creative_tools, "creative_director")
        
        # Add tools for other agents similarly...
    
    def register_tools(self, tools: List[BaseTool], agent_name: str):
        """Register multiple tools for an agent."""
        for tool in tools:
            self.register_tool(tool, agent_name)
            
    def register_tool(self, tool: BaseTool, agent_name: str):
        if agent_name not in self.tools_by_agent:
            self.tools_by_agent[agent_name] = []
        self.tools_by_agent[agent_name].append(tool)
    
    def get_tools_for_agent(self, agent_name: str) -> List[BaseTool]:
        return self.tools_by_agent.get(agent_name, [])
