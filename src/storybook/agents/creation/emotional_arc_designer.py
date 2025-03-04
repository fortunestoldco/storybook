from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.emotion import (
    EmotionalMapTool,
    ReaderResponseTool,
    EmotionalPacingTool
)
from storybook.agents.base_agent import BaseAgent

class EmotionalArcDesigner(BaseAgent):
    """Designer responsible for crafting emotional arcs and reader engagement."""
    
    def __init__(self):
        super().__init__(
            name="emotional_arc_designer",
            tools=[
                EmotionalMapTool(),
                ReaderResponseTool(),
                EmotionalPacingTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Design and maintain emotional arcs."""
        task = state.current_input.get("task", "")
        section_id = task.get("section_id", "")
        
        if "reader_response" in task.lower():
            response = await self.tools[1].arun(
                content=state.project.content,
                target_emotions=task.get("target_emotions", []),
                reader_profile=state.project.target_audience
            )
            return {
                "messages": [AIMessage(content="Reader response analysis complete")],
                "emotional_updates": {"reader_response": response}
            }
        
        if "pacing" in task.lower():
            pacing = await self.tools[2].arun(
                content=state.project.content,
                emotional_map=state.project.content.get("emotional_map", {}),
                target_pacing=state.project.style_preferences.get("emotional_pacing", {})
            )
            return {
                "messages": [AIMessage(content="Emotional pacing optimized")],
                "emotional_updates": {"pacing": pacing}
            }
        
        # Default to emotional mapping
        emotional_map = await self.tools[0].arun(
            content=state.project.content,
            characters=state.project.content.get("characters", {}),
            plot_points=state.project.content.get("plot_threads", [])
        )
        
        return {
            "messages": [AIMessage(content="Emotional map updated")],
            "emotional_updates": {"map": emotional_map}
        }