from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.prose import (
    StyleRefinementTool,
    ImageryEnhancementTool,
    SentenceStructureTool
)
from storybook.agents.base_agent import BaseAgent

class ProseEnhancementSpecialist(BaseAgent):
    """Specialist responsible for improving prose quality."""
    
    def __init__(self):
        super().__init__(
            name="prose_enhancement_specialist",
            tools=[
                StyleRefinementTool(),
                ImageryEnhancementTool(),
                SentenceStructureTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Enhance and refine prose elements."""
        task = state.current_input.get("task", "")
        section_id = task.get("section_id", "")
        
        if "imagery" in task.lower():
            imagery = await self.tools[1].arun(
                content=task.get("content", ""),
                style_guide=state.project.style_preferences,
                scene_context=state.project.content.get("scenes", {}).get(section_id, {})
            )
            return {
                "messages": [AIMessage(content="Imagery enhanced")],
                "prose_updates": {section_id: {"imagery": imagery}}
            }
        
        if "sentence" in task.lower():
            structure = await self.tools[2].arun(
                content=task.get("content", ""),
                style_preferences=state.project.style_preferences,
                target_rhythm=task.get("target_rhythm", {})
            )
            return {
                "messages": [AIMessage(content="Sentence structure optimized")],
                "prose_updates": {section_id: {"structure": structure}}
            }
        
        # Default to style refinement
        style = await self.tools[0].arun(
            content=task.get("content", ""),
            style_guide=state.project.style_preferences,
            context=state.project.content.get("scenes", {}).get(section_id, {})
        )
        
        return {
            "messages": [AIMessage(content="Style refined")],
            "prose_updates": {section_id: {"style": style}}
        }
