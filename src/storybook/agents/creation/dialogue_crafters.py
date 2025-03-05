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
    """Collaborative dialogue crafting specialists."""
    
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
        
        if "flow" in task.get("type", "").lower():
            flow = await self.tools[1].invoke({
                "content": state.project.content,
                "scene_id": task.get("scene_id")
            })
            return {
                "messages": [AIMessage(content="Dialogue flow optimization completed")],
                "dialogue_updates": {"flow": flow}
            }
            
        if "voice" in task.get("type", "").lower():
            voice = await self.tools[2].invoke({
                "content": state.project.content,
                "character_id": task.get("character_id")
            })
            return {
                "messages": [AIMessage(content="Character voice application completed")],
                "dialogue_updates": {"voice": voice}
            }
        
        creation = await self.tools[0].invoke({
            "content": state.project.content,
            "scene_context": task.get("scene_context", {})
        })
        return {
            "messages": [AIMessage(content="Dialogue creation completed")],
            "dialogue_updates": {"creation": creation}
        }
