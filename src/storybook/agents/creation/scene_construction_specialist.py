from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.scene import (
    SceneConstructionTool,
    SceneFlowAnalysisTool,
    SceneTransitionTool
)
from storybook.agents.base_agent import BaseAgent

class SceneConstructionSpecialist(BaseAgent):
    """Specialist responsible for scene construction and organization."""
    
    def __init__(self):
        super().__init__(
            name="scene_construction_specialist",
            tools=[
                SceneConstructionTool(),
                SceneFlowAnalysisTool(),
                SceneTransitionTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process scene construction tasks."""
        task = state.current_input.get("task", {})
        scene_id = task.get("scene_id")
        
        if "flow" in task.get("type", "").lower():
            flow = await self.tools[1].arun(
                content=state.project.content,
                scene_id=scene_id,
                context=state.project.content.get("scene_context", {})
            )
            return {
                "messages": [AIMessage(content="Scene flow analyzed")],
                "scene_updates": {"flow": flow}
            }
            
        if "transition" in task.get("type", "").lower():
            transition = await self.tools[2].arun(
                content=state.project.content,
                scene_id=scene_id,
                next_scene=task.get("next_scene_id")
            )
            return {
                "messages": [AIMessage(content="Scene transition crafted")],
                "scene_updates": {"transition": transition}
            }
        
        scene = await self.tools[0].arun(
            content=state.project.content,
            scene_id=scene_id,
            parameters=task.get("parameters", {})
        )
        return {
            "messages": [AIMessage(content="Scene constructed")],
            "scene_updates": {"scene": scene}
        }