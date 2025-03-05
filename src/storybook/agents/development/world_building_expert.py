from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.worldbuilding import (
    WorldDesignTool,
    SystemDesignTool,
    ConsistencyCheckerTool
)
from storybook.agents.base_agent import BaseAgent

class WorldBuildingExpert(BaseAgent):
    """Expert responsible for world building and consistency."""
    
    def __init__(self):
        super().__init__(
            name="world_building_expert",
            tools=[
                WorldDesignTool(),
                SystemDesignTool(),
                ConsistencyCheckerTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process world building tasks."""
        task = state.current_input.get("task", {})
        
        if "consistency" in task.get("type", "").lower():
            check = await self.tools[2].invoke({
                "content": state.project.content,
                "check_type": task.get("check_type", "all")
            })
            return {
                "messages": [AIMessage(content="Consistency check completed")],
                "world_updates": {"consistency": check}
            }
            
        if "system" in task.get("type", "").lower():
            system = await self.tools[1].invoke({
                "content": state.project.content,
                "system_type": task.get("system_type", "magic")
            })
            return {
                "messages": [AIMessage(content="System design completed")],
                "world_updates": {"system": system}
            }
        
        world = await self.tools[0].invoke({
            "content": state.project.content,
            "world_type": task.get("world_type", "fantasy")
        })
        return {
            "messages": [AIMessage(content="World design updated")],
            "world_updates": {"world": world}
        }