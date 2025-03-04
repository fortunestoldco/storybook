from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.rhythm import (
    ProseCadenceTool,
    SentenceVariationTool,
    PacingOptimizationTool
)
from storybook.agents.base_agent import BaseAgent

class RhythmCadenceOptimizer(BaseAgent):
    """Optimizer responsible for prose rhythm and flow."""
    
    def __init__(self):
        super().__init__(
            name="rhythm_cadence_optimizer",
            tools=[
                ProseCadenceTool(),
                SentenceVariationTool(),
                PacingOptimizationTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Optimize rhythm and cadence of prose."""
        task = state.current_input.get("task", "")
        section_id = task.get("section_id", "")
        
        if "variation" in task.lower():
            variation = await self.tools[1].arun(
                content=task.get("content", ""),
                target_style=state.project.style_preferences
            )
            return {
                "messages": [AIMessage(content="Sentence variation optimized")],
                "rhythm_updates": {section_id: {"variation": variation}}
            }
        
        if "pacing" in task.lower():
            pacing = await self.tools[2].arun(
                content=task.get("content", ""),
                scene_type=task.get("scene_type", ""),
                target_pace=task.get("target_pace", "")
            )
            return {
                "messages": [AIMessage(content="Pacing optimized")],
                "rhythm_updates": {section_id: {"pacing": pacing}}
            }
        
        # Default to cadence analysis
        cadence = await self.tools[0].arun(
            content=task.get("content", ""),
            style_preferences=state.project.style_preferences
        )
        
        return {
            "messages": [AIMessage(content="Prose cadence optimized")],
            "rhythm_updates": {section_id: {"cadence": cadence}}
        }