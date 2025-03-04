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
    """Director responsible for overseeing content development process."""
    
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
        """Manage content development process."""
        task = state.current_input.get("task", "")
        
        if "quality" in task.lower():
            assessment = await self.tools[1].arun(
                content=state.project.content,
                quality_metrics=state.project.quality_assessment,
                phase=state.phase
            )
            return {
                "messages": [AIMessage(content="Content quality assessment complete")],
                "content_updates": {"quality_assessment": assessment}
            }
        
        if "progress" in task.lower():
            progress = await self.tools[2].arun(
                content=state.project.content,
                milestones=state.project.content.get("milestones", {}),
                timeline=state.project.content.get("timeline", {})
            )
            return {
                "messages": [AIMessage(content="Content progress tracked")],
                "content_updates": {"progress": progress}
            }
        
        # Default to content planning
        plan = await self.tools[0].arun(
            content=state.project.content,
            outline=state.project.content.get("outline", {}),
            style_preferences=state.project.style_preferences
        )
        
        return {
            "messages": [AIMessage(content="Content development plan updated")],
            "content_updates": {"plan": plan}
        }