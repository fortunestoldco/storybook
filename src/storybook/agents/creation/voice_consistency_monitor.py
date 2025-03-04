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
        task = state.current_input.get("task", "")
        section_id = task.get("section_id", "")
        
        if "style" in task.lower():
            style_check = await self.tools[1].arun(
                content=state.project.content,
                section_id=section_id,