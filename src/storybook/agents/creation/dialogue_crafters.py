from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.dialogue import (
    DialogueCreationTool,
    DialogueFlowTool,
    CharacterVoiceTool
)
from storybook.agents.base_agent import BaseAgent

class DialogueCrafters(BaseAgent):
    """Crafters responsible for creating engaging and character-appropriate dialogue."""
    
    def __init__(self):
        super().__init__(
            name="dialogue_crafters",
            tools=[
                DialogueCreationTool(),
                DialogueFlowTool(),
                CharacterVoiceTool()
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
        
        if "flow" in task.get("type", "").lower():
            flow = await self.tools[1].arun(
                content=state.project.content,
                scene_id=scene_id,
                context=state.project.content.get("scene_context", {})
            )
            return {
                "messages": [AIMessage(content="Dialogue flow analyzed")],
                "dialogue_updates": {"flow": flow}
            }
            
        if "voice" in task.get("type", "").lower():
            voice = await self.tools[2].arun(
                content=state.project.content,
                scene_id=scene_id,
                characters=task.get("characters", [])
            )
            return {
                "messages": [AIMessage(content="Character voice crafted")],
                "dialogue_updates": {"voice": voice}
            }
        
        dialogue = await self.tools[0].arun(
            content=state.project.content,
            scene_id=scene_id,
            parameters=task.get("parameters", {})
        )
        return {
            "messages": [AIMessage(content="Dialogue created")],
            "dialogue_updates": {"dialogue": dialogue}
        }
