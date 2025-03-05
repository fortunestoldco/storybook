from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.voice import (
    NarrativeVoiceTool,
    StyleConsistencyTool,
    ToneAnalysisTool
)
from storybook.agents.base_agent import BaseAgent

class VoiceConsistencyMonitor(BaseAgent):
    """Monitor responsible for maintaining consistent narrative voice."""
    
    def __init__(self):
        super().__init__(
            name="voice_consistency_monitor",
            tools=[
                NarrativeVoiceTool(),
                StyleConsistencyTool(),
                ToneAnalysisTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Monitor and maintain narrative voice consistency."""
        task = state.current_input.get("task", {})
        section_id = task.get("section_id", "")
        
        # Handle style consistency checks
        if "style" in task.get("type", "").lower():
            style_check = await self.tools[1].arun(
                content=state.project.content,
                section_id=section_id,
                style_guide=state.project.content.get("style_guide", {})
            )
            return {
                "messages": [AIMessage(content="Style consistency verified")],
                "voice_updates": {"style_check": style_check}
            }
        
        # Handle tone analysis
        if "tone" in task.get("type", "").lower():
            tone_analysis = await self.tools[2].arun(
                content=state.project.content,
                section_id=section_id,
                target_tone=state.project.content.get("tone_preferences", {})
            )
            return {
                "messages": [AIMessage(content="Tone analysis completed")],
                "voice_updates": {"tone_analysis": tone_analysis}
            }
        
        # Default to narrative voice check
        voice_check = await self.tools[0].arun(
            content=state.project.content,
            section_id=section_id,
            voice_guide=state.project.content.get("narrative_voice", {})
        )
        
        return {
            "messages": [AIMessage(content="Narrative voice consistency verified")],
            "voice_updates": {"voice_check": voice_check}
        }