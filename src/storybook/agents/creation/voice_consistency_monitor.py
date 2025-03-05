from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.voice import (
    NarrativeVoiceTool,
    VoiceConsistencyTool,
    ToneManagementTool
)
from storybook.agents.base_agent import BaseAgent

class VoiceConsistencyMonitor(BaseAgent):
    """Agent responsible for monitoring voice consistency."""
    
    def __init__(self):
        super().__init__(
            name="voice_consistency_monitor",
            tools=[
                NarrativeVoiceTool(),
                VoiceConsistencyTool(),
                ToneManagementTool()
            ]
        )
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process voice consistency tasks."""
        task = state.current_input.get("task", {})
        
        if "tone" in task.get("type", "").lower():
            tone = await self.tools[2].invoke({
                "content": state.project.content,
                "tone_profile": task.get("tone_profile", {})
            })
            return {
                "messages": [AIMessage(content="Tone management completed")],
                "voice_updates": {"tone": tone}
            }
        
        if "consistency" in task.get("type", "").lower():
            consistency = await self.tools[1].invoke({
                "content": state.project.content,
                "section_id": task.get("section_id")
            })
            return {
                "messages": [AIMessage(content="Voice consistency check completed")],
                "voice_updates": {"consistency": consistency}
            }
        
        voice = await self.tools[0].invoke({
            "content": state.project.content,
            "style_profile": task.get("style_profile", {})
        })
        return {
            "messages": [AIMessage(content="Narrative voice updated")],
            "voice_updates": {"voice": voice}
        }
