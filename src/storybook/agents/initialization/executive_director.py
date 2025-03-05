from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.quality import QualityMetricsTool
from storybook.tools.project import TaskDelegationTool
from storybook.agents.base_agent import BaseAgent

class ExecutiveDirector(BaseAgent):
    """Director responsible for high-level project management."""
    
    def __init__(self):
        super().__init__(
            name="executive_director",
            tools=[
                TaskDelegationTool(),
                QualityMetricsTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process executive management tasks."""
        task = state.current_input.get("task", {})
        
        if "quality" in task.get("type", "").lower():
            quality = await self.tools[1].invoke({
                "content": state.project.content,
                "metric_types": task.get("metric_types", [])
            })
            return {
                "messages": [AIMessage(content="Quality metrics assessment completed")],
                "management_updates": {"quality": quality}
            }
        
        delegation = await self.tools[0].invoke({
            "content": state.project.content,
            "task_type": task.get("type"),
            "priority": task.get("priority", 1),
            "requirements": task.get("requirements", {})
        })
        return {
            "messages": [AIMessage(content="Task delegation completed")],
            "management_updates": {"delegation": delegation}
        }