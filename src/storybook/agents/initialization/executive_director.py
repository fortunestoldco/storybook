from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.quality import QualityMetricsTool
from storybook.tools.delegation import TaskDelegationTool
from storybook.tools.project import ProjectManagementTool
from storybook.agents.base_agent import BaseAgent

class ExecutiveDirector(BaseAgent):
    """Director responsible for overall project management."""
    
    def __init__(self):
        super().__init__(
            name="executive_director",
            tools=[
                QualityMetricsTool(),
                TaskDelegationTool(),
                ProjectManagementTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process executive management tasks."""
        task = state.current_input.get("task", "")
        
        if "delegate" in task.lower():
            delegation = await self.tools[1].arun(
                task=task,
                agents=state.project.content.get("agents", {})
            )
            return {
                "messages": [AIMessage(content="Task delegated")],
                "delegation": delegation
            }
        
        if "quality" in task.lower():
            quality = await self.tools[0].arun(
                content=state.project.content,
                metrics=state.project.content.get("quality_metrics", {})
            )
            return {
                "messages": [AIMessage(content="Quality assessed")],
                "quality": quality
            }
        
        # Default to project management
        management = await self.tools[2].arun(
            content=state.project.content,
            phase=state.phase
        )
        return {
            "messages": [AIMessage(content="Project status updated")],
            "project_updates": management
        }