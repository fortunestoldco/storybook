from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.emotion import (
    EmotionalArcTool,
    EmotionalResonanceTool,
    EmotionalPacingTool
)
from storybook.agents.base_agent import BaseAgent

class EmotionalArcDesigner(BaseAgent):
    """Designer responsible for emotional arcs and resonance."""
    
    def __init__(self):
        super().__init__(
            name="emotional_arc_designer",
            tools=[
                EmotionalArcTool(),
                EmotionalResonanceTool(),
                EmotionalPacingTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        task = state.current_input.get("task", {})
        character_id = task.get("character_id")
        
        if not character_id:
            return {
                "messages": [AIMessage(content="No character ID provided")],
                "emotion_updates": {}
            }
        
        try:
            arc = await self.tools[0].arun(
                content=state.project.content,
                character_id=character_id
            )
            return {
                "messages": [AIMessage(content="Emotional arc designed")],
                "emotion_updates": {"arc": arc}
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error: {str(e)}")],
                "error": str(e)
            }