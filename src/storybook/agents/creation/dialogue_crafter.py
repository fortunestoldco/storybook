from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.dialogue import (
    DialogueGenerationTool,
    CharacterVoiceTool,
    SubtextTool
)
from storybook.agents.base_agent import BaseAgent

class DialogueCrafter(BaseAgent):
    """Specialist responsible for crafting character dialogue."""
    
    def __init__(self):
        super().__init__(
            name="dialogue_crafter",
            tools=[
                DialogueGenerationTool(),
                CharacterVoiceTool(),
                SubtextTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Craft and refine dialogue."""
        task = state.current_input.get("task", "")
        scene_id = task.get("scene_id", "")
        characters = task.get("characters", [])

        if "voice" in task.lower():
            voice_adjustments = await self.tools[1].arun(
                characters=characters,
                dialogue=task.get("dialogue", ""),
                character_profiles=state.project.content.get("characters", {})
            )
            return {
                "messages": [AIMessage(content="Character voices adjusted in dialogue")],
                "dialogue_updates": {
                    scene_id: {"voice_adjustments": voice_adjustments}
                }
            }

        if "subtext" in task.lower():
            subtext = await self.tools[2].arun(
                dialogue=task.get("dialogue", ""),
                context=task.get("context", {}),
                character_relationships=state.project.content.get("relationship_graph", {})
            )
            return {
                "messages": [AIMessage(content="Dialogue subtext enhanced")],
                "dialogue_updates": {
                    scene_id: {"subtext": subtext}
                }
            }

        # Default to dialogue generation
        dialogue = await self.tools[0].arun(
            scene_id=scene_id,
            characters=characters,
            scene_context=state.project.content.get("scenes", {}).get(scene_id, {}),
            character_profiles=state.project.content.get("characters", {})
        )

        return {
            "messages": [AIMessage(content=f"Dialogue generated for scene: {scene_id}")],
            "dialogue_updates": {
                scene_id: {"dialogue": dialogue}
            }
        }