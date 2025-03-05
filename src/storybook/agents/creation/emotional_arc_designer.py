from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.emotion import (
    EmotionalArcTool,
    EmotionalPacingTool,
    EmotionalIntensityTool
)
from storybook.agents.base_agent import BaseAgent

class EmotionalArcDesigner(BaseAgent):
    """Designer responsible for emotional arcs and pacing."""
    
    def __init__(self):
        super().__init__(
            name="emotional_arc_designer",
            tools=[
                EmotionalArcTool(),
                EmotionalPacingTool(),
                EmotionalIntensityTool()
            ]
        )

    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process emotional arc design tasks."""
        task = state.current_input.get("task", {})
        
        if "intensity" in task.get("type", "").lower():
            intensity = await self.tools[2].invoke({
                "content": state.project.content,
                "intensity_target": task.get("intensity_target", 0.5)
            })
            return {
                "messages": [AIMessage(content="Emotional intensity updated")],
                "emotional_updates": {"intensity": intensity}
            }
            
        if "pacing" in task.get("type", "").lower():
            pacing = await self.tools[1].invoke({
                "content": state.project.content,
                "section_id": task.get("section_id")
            })
            return {
                "messages": [AIMessage(content="Emotional pacing updated")],
                "emotional_updates": {"pacing": pacing}
            }
        
        arc = await self.tools[0].invoke({
            "content": state.project.content,
            "arc_type": task.get("arc_type", "rising")
        })
        return {
            "messages": [AIMessage(content="Emotional arc updated")],
            "emotional_updates": {"arc": arc}
        }