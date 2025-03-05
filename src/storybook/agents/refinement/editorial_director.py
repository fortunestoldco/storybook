from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.editorial import (
    EditorialPlanningTool,
    EditorialRevisionTool,
    StyleGuideTool
)
from storybook.agents.base_agent import BaseAgent

class EditorialDirector(BaseAgent):
    """Director responsible for editorial oversight and planning."""
    
    def __init__(self):
        super().__init__(
            name="editorial_director",
            tools=[
                EditorialPlanningTool(),
                EditorialRevisionTool(),
                StyleGuideTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        task = state.current_input.get("task", {})
        
        if "style" in task.get("type", "").lower():
            style = await self.tools[2].invoke({
                "content": state.project.content,
                "style_guide": task.get("style_guide", {})
            })
            return {
                "messages": [AIMessage(content="Style guide application completed")],
                "editorial_updates": {"style": style}
            }
        
        if "revision" in task.get("type", "").lower():
            revision = await self.tools[1].invoke({
                "content": state.project.content,
                "scope": task.get("scope", "global")
            })
            return {
                "messages": [AIMessage(content="Editorial revision completed")],
                "editorial_updates": {"revision": revision}
            }
        
        planning = await self.tools[0].invoke({
            "content": state.project.content,
            "scope": task.get("scope", "global")
        })
        return {
            "messages": [AIMessage(content="Editorial planning completed")],
            "editorial_updates": {"planning": planning}
        }