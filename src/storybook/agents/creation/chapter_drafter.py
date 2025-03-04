from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.drafting import (
    ChapterStructureTool,
    SceneSequenceTool,
    NarrativeFlowTool
)
from storybook.agents.base_agent import BaseAgent

class ChapterDrafter(BaseAgent):
    """Agent responsible for drafting individual chapters."""
    
    def __init__(self):
        super().__init__(
            name="chapter_drafter",
            tools=[
                ChapterStructureTool(),
                SceneSequenceTool(),
                NarrativeFlowTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Draft and structure chapters."""
        task = state.current_input.get("task", "")
        chapter_id = task.get("chapter_id", "")
        
        if "sequence" in task.lower():
            sequence = await self.tools[1].arun(
                chapter_id=chapter_id,
                outline=state.project.content.get("outline", {}),
                scenes=state.project.content.get("scenes", {})
            )
            return {
                "messages": [AIMessage(content=f"Scene sequence updated for chapter: {chapter_id}")],
                "chapter_updates": {chapter_id: {"scene_sequence": sequence}}
            }
        
        if "flow" in task.lower():
            flow = await self.tools[2].arun(
                chapter_id=chapter_id,
                content=state.project.content,
                style=state.project.style_preferences
            )
            return {
                "messages": [AIMessage(content=f"Narrative flow optimized for chapter: {chapter_id}")],
                "chapter_updates": {chapter_id: {"narrative_flow": flow}}
            }
        
        # Default to chapter structure
        structure = await self.tools[0].arun(
            chapter_id=chapter_id,
            outline=state.project.content.get("outline", {}),
            plot_threads=state.project.content.get("plot_threads", [])
        )
        
        return {
            "messages": [AIMessage(content=f"Chapter structure created: {chapter_id}")],
            "chapter_updates": {chapter_id: {"structure": structure}}
        }