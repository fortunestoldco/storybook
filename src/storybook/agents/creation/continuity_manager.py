from typing import Dict, Any, List
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
            character_tracking = await self.tools[2].arun(
                content=state.project.content,
                characters=task.get("characters", [])
            )
            return {
                "messages": [AIMessage(content="Character tracking updated and verified")],
                "continuity_updates": {"character_tracking": character_tracking}
            }
            
        if "plot_consistency" in task.lower():
            consistency = await self.tools[1].arun(
                content=state.project.content,
                plot_threads=task.get("plot_threads", [])
            )
            return {
                "messages": [AIMessage(content="Plot consistency verified")],
                "continuity_updates": {"plot_consistency": consistency}
            }
            
        return {
            "messages": [AIMessage(content="No specific continuity task identified")],
            "continuity_updates": {}
        }