from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.creativity import (
    CreativeVisionTool,
    StoryElementsTool,
    ThematicAnalysisTool
)
from storybook.agents.base_agent import BaseAgent

class CreativeDirector(BaseAgent):
    """Director responsible for creative vision and story elements."""
    
    def __init__(self):
        super().__init__(
            name="creative_director",
            tools=[
                CreativeVisionTool(),
                StoryElementsTool(),
                ThematicAnalysisTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process creative direction tasks."""
        task = state.current_input.get("task", {})
        
        if "theme" in task.get("type", "").lower():
            analysis = await self.tools[2].invoke({
                "content": state.project.content,
                "themes": task.get("themes", [])
            })
            return {
                "messages": [AIMessage(content="Thematic analysis completed")],
                "creative_updates": {"themes": analysis}
            }
            
        if "elements" in task.get("type", "").lower():
            elements = await self.tools[1].invoke({
                "content": state.project.content,
                "elements": task.get("elements", {})
            })
            return {
                "messages": [AIMessage(content="Story elements updated")],
                "creative_updates": {"elements": elements}
            }
        
        vision = await self.tools[0].invoke({
            "content": state.project.content,
            "style_preferences": task.get("style_preferences", {})
        })
        return {
            "messages": [AIMessage(content="Creative vision updated")],
            "creative_updates": {"vision": vision}
        }