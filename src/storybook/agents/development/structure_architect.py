from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.structure import (
    StoryStructureTool,
    PacingAnalysisTool,
    ChapterOutlineTool
)
from storybook.agents.base_agent import BaseAgent

class StructureArchitect(BaseAgent):
    """Architect responsible for novel's structural design."""
    
    def __init__(self):
        super().__init__(
            name="structure_architect",
            tools=[
                StoryStructureTool(),
                PacingAnalysisTool(),
                ChapterOutlineTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Design and maintain novel structure."""
        task = state.current_input.get("task", "")
        
        if "outline" in task.lower():
            outline = await self.tools[2].arun(
                content=state.project.content,
                style_preferences=state.project.style_preferences
            )
            return {
                "messages": [AIMessage(content="Chapter outline updated")],
                "structure_updates": {"outline": outline}
            }
        
        if "pacing" in task.lower():
            pacing = await self.tools[1].arun(
                content=state.project.content,
                target_length=state.project.length_target
            )
            return {
                "messages": [AIMessage(content="Pacing analysis complete")],
                "structure_updates": {"pacing": pacing}
            }
        
        # Default to overall structure analysis
        structure = await self.tools[0].arun(
            content=state.project.content,
            outline=state.project.content.get("outline", {})
        )
        
        return {
            "messages": [AIMessage(content="Structure analysis complete")],
            "structure_updates": {"analysis": structure}
        }