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
    """Agent responsible for crafting dialogue."""
    
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
        """Process dialogue crafting tasks."""
        task = state.current_input.get("task", {})
        scene_id = task.get("scene_id")
        characters = task.get("characters", [])
        
        if "voice" in task.get("type", "").lower():
            voices = await self.tools[1].arun(
                content=state.project.content,
                characters=characters
            )
            return {
                "messages": [AIMessage(content="Character voices defined")],
                "dialogue_updates": {"voices": voices}
            }
            
        if "subtext" in task.get("type", "").lower():
            subtext = await self.tools[2].arun(
                content=state.project.content,
                scene_id=scene_id,
                dialogue=task.get("dialogue", {})
            )
            return {
                "messages": [AIMessage(content="Dialogue subtext analyzed")],
                "dialogue_updates": {"subtext": subtext}
            }
        
        dialogue = await self.tools[0].arun(
            content=state.project.content,
            scene_id=scene_id,
            characters=characters,
            parameters=task.get("parameters", {})
        )
        return {
            "messages": [AIMessage(content="Dialogue generated")],
            "dialogue_updates": {"dialogue": dialogue}
        }