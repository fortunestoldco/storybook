from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.editorial import (
    EditorialPlanningTool,
    EditorialRevisionTool
)
from storybook.agents.base_agent import BaseAgent

class EditorialDirector(BaseAgent):
    """Director responsible for editorial process."""
    
    def __init__(self):
        super().__init__(
            name="editorial_director",
            tools=[
                EditorialPlanningTool(),
                EditorialRevisionTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process editorial tasks."""
        task = state.current_input.get("task", {})
        
        if "revision" in task.get("type", "").lower():
            revision = await self.tools[1].invoke({
                "content": state.project.content,
                "revision_type": task.get("revision_type", "comprehensive")
            })
            return {
                "messages": [AIMessage(content="Editorial revision completed")],
                "editorial_updates": {"revision": revision}
            }
        
        plan = await self.tools[0].invoke({
            "content": state.project.content,
            "editorial_phase": task.get("phase", "initial")
        })
        return {
            "messages": [AIMessage(content="Editorial planning completed")],
            "editorial_updates": {"plan": plan}
        }