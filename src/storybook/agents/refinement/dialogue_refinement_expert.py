from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.dialogue import (
    DialoguePolishingTool,
    ConversationFlowTool,
    SubtextEnhancementTool
)
from storybook.agents.base_agent import BaseAgent

class DialogueRefinementExpert(BaseAgent):
    """Expert responsible for polishing and enhancing dialogue."""
    
    def __init__(self):
        super().__init__(
            name="dialogue_refinement_expert",
            tools=[
                DialoguePolishingTool(),
                ConversationFlowTool(),
                SubtextEnhancementTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Refine and enhance dialogue."""
        task = state.current_input.get("task", "")
        scene_id = task.get("scene_id", "")
        
        if "flow" in task.lower():
            flow = await self.tools[1].arun(
                scene=state.project.content.get("scenes", {}).get(scene_id, {}),
                characters=state.project.content.get("characters", {})
            )
            return {
                "messages": [AIMessage(content="Conversation flow optimized")],
                "dialogue_updates": {scene_id: {"flow": flow}}
            }
            
        if "subtext" in task.lower():
            subtext = await self.tools[2].arun(
                dialogue=task.get("dialogue", ""),
                character_relationships=state.project.content.get("relationship_graph", {})
            )
            return {
                "messages": [AIMessage(content="Dialogue subtext enhanced")],
                "dialogue_updates": {scene_id: {"subtext": subtext}}
            }
        
        # Default to dialogue polishing
        polished = await self.tools[0].arun(
            dialogue=task.get("dialogue", ""),
            character_voices=state.project.content.get("character_voices", {})
        )
        
        return {
            "messages": [AIMessage(content="Dialogue polished")],
            "dialogue_updates": {scene_id: {"polished": polished}}
        }