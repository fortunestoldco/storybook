from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.character import (
    ArcAnalysisTool,
    TransformationMapTool,
    ConsistencyCheckTool
)
from storybook.agents.base_agent import BaseAgent

class CharacterArcEvaluator(BaseAgent):
    """Evaluator responsible for analyzing and refining character arcs."""
    
    def __init__(self):
        super().__init__(
            name="character_arc_evaluator",
            tools=[
                ArcAnalysisTool(),
                TransformationMapTool(),
                ConsistencyCheckTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Evaluate and refine character arcs."""
        task = state.current_input.get("task", "")
        character_id = task.get("character_id", "")
        
        if "transformation" in task.lower():
            transformation = await self.tools[1].arun(
                character_id=character_id,
                character_data=state.project.content.get("characters", {}).get(character_id, {}),
                plot_points=state.project.content.get("plot_threads", [])
            )
            return {
                "messages": [AIMessage(content=f"Character transformation mapped: {character_id}")],
                "arc_updates": {character_id: {"transformation": transformation}}
            }
        
        if "consistency" in task.lower():
            consistency = await self.tools[2].arun(
                character_id=character_id,
                content=state.project.content,
                arc_data=state.project.content.get("character_arcs", {}).get(character_id, {})
            )
            return {
                "messages": [AIMessage(content=f"Character consistency verified: {character_id}")],
                "arc_updates": {character_id: {"consistency": consistency}}
            }
        
        # Default to arc analysis
        analysis = await self.tools[0].arun(
            character_id=character_id,
            content=state.project.content,
            character_data=state.project.content.get("characters", {}).get(character_id, {})
        )
        
        return {
            "messages": [AIMessage(content=f"Character arc analyzed: {character_id}")],
            "arc_updates": {character_id: {"analysis": analysis}}
        }