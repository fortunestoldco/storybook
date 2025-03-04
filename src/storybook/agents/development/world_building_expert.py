from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.worldbuilding import (
    WorldDesignTool,
    ConsistencyCheckerTool,
    LocationManagerTool
)
from storybook.agents.base_agent import BaseAgent

class WorldBuildingExpert(BaseAgent):
    """Expert responsible for creating and maintaining the story world."""
    
    def __init__(self):
        super().__init__(
            name="world_building_expert",
            tools=[
                WorldDesignTool(),
                ConsistencyCheckerTool(),
                LocationManagerTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Develop and maintain world-building elements."""
        task = state.current_input.get("task", "")
        
        if "consistency" in task.lower():
            check_result = await self.tools[1].arun(
                content=state.project.content,
                world_elements=state.project.content.get("world", {})
            )
            return {
                "messages": [AIMessage(content="World consistency verified")],
                "world_updates": {"consistency": check_result}
            }
        
        if "location" in task.lower():
            location_data = await self.tools[2].arun(
                content=state.project.content,
                location_name=task.get("location_name", "")
            )
            return {
                "messages": [AIMessage(content="Location details updated")],
                "world_updates": {"locations": location_data}
            }
        
        # Default to world design
        world_elements = await self.tools[0].arun(
            content=state.project.content,
            genre=state.project.genre,
            style_preferences=state.project.style_preferences
        )
        
        return {
            "messages": [AIMessage(content="World elements updated")],
            "world_updates": {"elements": world_elements}
        }