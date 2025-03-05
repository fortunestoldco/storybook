from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.dialogue import (
    DialogueGenerationTool,
    DialogueStyleTool,
    DialogueRevisionTool,
    CharacterVoiceTool
)
from storybook.agents.base_agent import BaseAgent

class DialogueCrafter(BaseAgent):
    """Agent responsible for crafting and managing dialogue."""
    
    def __init__(self):
        super().__init__(
            name="dialogue_crafter",
            tools=[
                DialogueGenerationTool(),
                DialogueStyleTool(),
                DialogueRevisionTool(),
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
        
        if "voice" in task.get("type", "").lower():
            voice = await self.tools[3].invoke({
                "content": state.project.content,
                "character_id": task.get("character_id"),
                "voice_profile": task.get("voice_profile", {})
            })
            return {
                "messages": [AIMessage(content="Character voice updated")],
                "dialogue_updates": {"voice": voice}
            }
            
        if "revision" in task.get("type", "").lower():
            revision = await self.tools[2].invoke({
                "content": state.project.content,
                "section_id": task.get("section_id")
            })
            return {
                "messages": [AIMessage(content="Dialogue revision completed")],
                "dialogue_updates": {"revision": revision}
            }
        
        if "style" in task.get("type", "").lower():
            style = await self.tools[1].invoke({
                "content": state.project.content,
                "style_guide": task.get("style_guide", {})
            })
            return {
                "messages": [AIMessage(content="Dialogue style updated")],
                "dialogue_updates": {"style": style}
            }
        
        dialogue = await self.tools[0].invoke({
            "content": state.project.content,
            "characters": task.get("characters", []),
            "context": task.get("context", {})
        })
        return {
            "messages": [AIMessage(content="Dialogue generation completed")],
            "dialogue_updates": {"dialogue": dialogue}
        }
