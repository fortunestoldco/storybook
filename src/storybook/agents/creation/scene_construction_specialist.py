from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.scene import (
    SceneStructureTool,
    ScenePacingTool,
    SceneTransitionTool
)
from storybook.agents.base_agent import BaseAgent

class SceneConstructionSpecialist(BaseAgent):
    """Specialist responsible for crafting individual scenes."""
    
    def __init__(self):
        super().__init__(
            name="scene_construction_specialist",
            tools=[
                SceneStructureTool(),
                ScenePacingTool(),
                SceneTransitionTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Construct and refine scenes."""
        task = state.current_input.get("task", "")
        scene_id = task.get("scene_id", "")
        chapter_id = task.get("chapter_id", "")

        if "pacing" in task.lower():
            pacing = await self.tools[1].arun(
                scene_id=scene_id,
                content=state.project.content.get("scenes", {}).get(scene_id, {}),
                chapter_context=state.project.content.get("chapters", {}).get(chapter_id, {})
            )
            return {
                "messages": [AIMessage(content=f"Scene pacing optimized: {scene_id}")],
                "scene_updates": {scene_id: {"pacing": pacing}}
            }

        if "transition" in task.lower():
            transition = await self.tools[2].arun(
                scene_id=scene_id,
                prev_scene=task.get("prev_scene", ""),
                next_scene=task.get("next_scene", ""),
                chapter_content=state.project.content.get("chapters", {}).get(chapter_id, {})
            )
            return {
                "messages": [AIMessage(content=f"Scene transitions updated: {scene_id}")],
                "scene_updates": {scene_id: {"transitions": transition}}
            }

        # Default to scene structure
        structure = await self.tools[0].arun(
            scene_id=scene_id,
            chapter_id=chapter_id,
            plot_threads=state.project.content.get("plot_threads", []),
            characters=state.project.content.get("characters", {})
        )

        return {
            "messages": [AIMessage(content=f"Scene structure created: {scene_id}")],
            "scene_updates": {scene_id: {"structure": structure}}
        }