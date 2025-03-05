from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.content import (
    ContentPlanningTool,
    ContentDevelopmentTool,
    ContentQualityTool
)
from storybook.agents.base_agent import BaseAgent

class ContentDevelopmentDirector(BaseAgent):
    """Director responsible for content development and quality."""
    
    def __init__(self):
        super().__init__(
            name="content_development_director",
            tools=[
                ContentPlanningTool(),
                ContentDevelopmentTool(),
                ContentQualityTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process content development tasks."""
        task = state.current_input.get("task", {})
        
        if "quality" in task.get("type", "").lower():
            quality = await self.tools[2].invoke({
                "content": state.project.content,
                "section_id": task.get("section_id")
            })
            return {
                "messages": [AIMessage(content="Quality assessment completed")],
                "content_updates": {"quality": quality}
            }
            
        if "development" in task.get("type", "").lower():
            development = await self.tools[1].invoke({
                "content": state.project.content,
                "section_id": task.get("section_id")
            })
            return {
                "messages": [AIMessage(content="Content development completed")],
                "content_updates": {"development": development}
            }
        
        plan = await self.tools[0].invoke({
            "content": state.project.content,
            "phase": task.get("phase", "outline")
        })
        return {
            "messages": [AIMessage(content="Content planning completed")],
            "content_updates": {"plan": plan}
        }