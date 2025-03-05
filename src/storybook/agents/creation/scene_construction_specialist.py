from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.scene import (
    SceneStructureTool,
    SceneFlowTool,
    SceneRevisionTool
)
from storybook.agents.base_agent import BaseAgent

class SceneConstructionSpecialist(BaseAgent):
    """Specialist responsible for scene construction and refinement."""
    
    def __init__(self):
        super().__init__(
            name="scene_construction_specialist",
            tools=[
                SceneStructureTool(),
                SceneFlowTool(),
                SceneRevisionTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process scene construction tasks."""
        task = state.current_input.get("task", {})
        
        if "revision" in task.get("type", "").lower():
            revision = await self.tools[2].invoke({
                "content": state.project.content,
                "scene_id": task.get("scene_id")
            })
            return {
                "messages": [AIMessage(content="Scene revision completed")],
                "scene_updates": {"revision": revision}
            }
            
        if "flow" in task.get("type", "").lower():
            flow = await self.tools[1].invoke({
                "content": state.project.content,
                "scene_id": task.get("scene_id")
            })
            return {
                "messages": [AIMessage(content="Scene flow analysis completed")],
                "scene_updates": {"flow": flow}
            }
        
        structure = await self.tools[0].invoke({
            "content": state.project.content,
            "scene_id": task.get("scene_id")
        })
        return {
            "messages": [AIMessage(content="Scene structure analysis completed")],
            "scene_updates": {"structure": structure}
        }