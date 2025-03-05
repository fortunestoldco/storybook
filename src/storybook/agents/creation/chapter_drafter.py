from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.chapter import (
    ChapterStructureTool,
    SceneSequenceTool,
    NarrativeFlowTool
)
from storybook.agents.base_agent import BaseAgent

class ChapterDrafter(BaseAgent):
    """Agent responsible for drafting chapters."""
    
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
        """Process chapter drafting tasks."""
        task = state.current_input.get("task", {})
        chapter_id = task.get("chapter_id")
        
        if "structure" in task.get("type", "").lower():
            structure = await self.tools[0].arun(
                content=state.project.content,
                chapter_id=chapter_id,
                outline=state.project.content.get("outline", {})
            )
            return {
                "messages": [AIMessage(content="Chapter structure created")],
                "chapter_updates": {"structure": structure}
            }
        
        if "sequence" in task.get("type", "").lower():
            sequence = await self.tools[1].arun(
                content=state.project.content,
                chapter_id=chapter_id,
                scenes=state.project.content.get("scenes", {})
            )
            return {
                "messages": [AIMessage(content="Scene sequence optimized")],
                "chapter_updates": {"sequence": sequence}
            }
        
        flow = await self.tools[2].arun(
            content=state.project.content,
            chapter_id=chapter_id
        )
        return {
            "messages": [AIMessage(content="Narrative flow analyzed")],
            "chapter_updates": {"flow": flow}
        }