from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.content import (
    ContentPlanningTool,
    ContentQualityTool,
    ContentProgressTool
)
from storybook.agents.base_agent import BaseAgent

class ContentDevelopmentDirector(BaseAgent):
    """Director responsible for content development and planning."""
    
    def __init__(self):
        super().__init__(
            name="content_development_director",
            tools=[
                ContentPlanningTool(),
                ContentQualityTool(),
                ContentProgressTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process content development tasks."""
        task = state.current_input.get("task", "")
        
        if "plan" in task.lower():
            plan = await self.tools[0].arun(
                content=state.project.content,
                outline=state.project.content.get("outline", {}),
                milestones=state.project.content.get("milestones", {})
            )
            return {
                "messages": [AIMessage(content="Content plan updated")],
                "content_updates": {"plan": plan}
            }
            
        if "quality" in task.lower():
            quality = await self.tools[1].arun(
                content=state.project.content,
                criteria=state.project.content.get("quality_criteria", {})
            )
            return {
                "messages": [AIMessage(content="Content quality assessed")],
                "content_updates": {"quality": quality}
            }
            
        # Default to progress tracking
        progress = await self.tools[2].arun(
            content=state.project.content,
            milestones=state.project.content.get("milestones", {})
        )
        return {
            "messages": [AIMessage(content="Content progress tracked")],
            "content_updates": {"progress": progress}
        }