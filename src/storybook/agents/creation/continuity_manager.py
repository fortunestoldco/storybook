from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.continuity import (
    TimelineTool,
    PlotConsistencyTool,
    CharacterTrackingTool
)
from storybook.agents.base_agent import BaseAgent

class ContinuityManager(BaseAgent):
    """Manager responsible for maintaining narrative continuity."""
    
    def __init__(self):
        super().__init__(
            name="continuity_manager",
            tools=[
                TimelineTool(),
                PlotConsistencyTool(),
                CharacterTrackingTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Manage and maintain narrative continuity."""
        task = state.current_input.get("task", "")
        
        if "timeline" in task.lower():
            timeline = await self.tools[0].arun(
                content=state.project.content,
                events=task.get("events", []),
                chronology=state.project.content.get("timeline", {})
            )
            return {
                "messages": [AIMessage(content="Timeline updated and verified")],
                "continuity_updates": {"timeline": timeline}
            }
        
        if "character_tracking" in task.lower():